"""Command-line interface for rex."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rex import __version__
from rex.config import ProjectConfig, expand_alias, load_aliases
from rex.execution import DirectExecutor, ExecutionContext, Executor, SlurmExecutor, SlurmOptions
from rex.output import error
from rex.ssh import SSHExecutor, FileTransfer
from rex.utils import validate_job_name

# Global debug flag
DEBUG = False


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        prog="rex",
        description="Remote execution tool for Python and shell commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
        epilog="""
Examples:
  rex gpu train.py                    # run on login node
  rex gpu -s train.py                 # run via SLURM
  rex gpu -s -d train.py              # SLURM detached (sbatch)
  rex gpu -d -n exp1 train.py         # detach with custom name
  rex gpu --jobs                      # list jobs
  rex gpu --watch --last              # wait for most recent job
  rex gpu --push ./data ~/data        # upload directory
  rex gpu --pull ~/checkpoints ./     # download directory
  rex gpu --exec "ls -la ~/models"    # run shell command on login node
  rex gpu -s --exec "nvidia-smi"      # run shell command via SLURM
  rex gpu --sync                      # sync current project to remote
  rex gpu --connect                   # open persistent connection
  rex gpu --disconnect                # close connection when done
""",
    )

    # Version
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")

    # Target
    parser.add_argument("target", nargs="?", help="alias or user@host")

    # Mode flags
    parser.add_argument("-d", "--detach", action="store_true", help="Run in background")
    parser.add_argument("-s", "--slurm", action="store_true", help="Use SLURM")
    parser.add_argument("-n", "--name", help="Job name for detached jobs")
    parser.add_argument("-p", "--python", default="python3", help="Python interpreter")
    parser.add_argument("-g", "--gpus", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument(
        "-m", "--module", action="append", dest="modules", default=[], help="Module to load"
    )

    # SLURM options
    parser.add_argument("--partition", help="SLURM partition (overrides --gpu/--cpu)")
    parser.add_argument("--gres", help="SLURM GPU resources")
    parser.add_argument("--time", help="SLURM time limit")
    parser.add_argument("--cpus", type=int, help="SLURM CPUs per task")
    parser.add_argument("--gpu", action="store_true", help="Use GPU partition")
    parser.add_argument("--cpu", action="store_true", help="Use CPU partition (override)")

    # Commands (nargs="?" allows --cmd --last without specifying job)
    parser.add_argument("--jobs", action="store_true", help="List jobs")
    parser.add_argument("--status", nargs="?", const="--last", metavar="JOB", help="Check job status")
    parser.add_argument("--log", nargs="?", const="--last", metavar="JOB", help="Show job log")
    parser.add_argument("--kill", nargs="?", const="--last", metavar="JOB", help="Kill job")
    parser.add_argument("--watch", nargs="?", const="--last", metavar="JOB", help="Wait for job")
    parser.add_argument("--gpu-info", action="store_true", dest="gpu_info", help="Show GPU info")

    parser.add_argument("--push", nargs="+", metavar="PATH", help="Push files")
    parser.add_argument("--pull", nargs="+", metavar="PATH", help="Pull files")
    parser.add_argument("--sync", nargs="?", const=".", metavar="PATH", help="Sync project")

    parser.add_argument("--build", action="store_true", help="Build venv on remote")
    parser.add_argument("--exec", dest="exec_cmd", metavar="CMD", help="Execute shell command")
    parser.add_argument("--exec-login", dest="exec_login", metavar="CMD", help="Execute on login node")

    parser.add_argument("--connect", action="store_true", help="Open persistent connection")
    parser.add_argument("--disconnect", action="store_true", help="Close persistent connection")
    parser.add_argument("--connection", action="store_true", help="Show connection status")

    # Modifiers
    parser.add_argument("--last", action="store_true", help="Use most recent job")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-f", "--follow", action="store_true", help="Follow log")
    parser.add_argument("--no-install", action="store_true", help="Skip pip install")
    parser.add_argument("--wait", action="store_true", help="Wait for build")
    parser.add_argument("--clean", action="store_true", help="Clean build")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    # Script and args
    parser.add_argument("script", nargs="?", help="Python script to run")
    parser.add_argument("script_args", nargs="*", help="Script arguments")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_intermixed_args(argv)

    # Special case: --connection without target lists all
    if args.connection and not args.target:
        from rex.commands.connection import connection_status
        return connection_status(None)

    # Load configs
    aliases = load_aliases()
    project = ProjectConfig.find_and_load()

    # Resolve target
    target = args.target
    extra_args: list[str] = []

    if target:
        # Try alias expansion
        expansion = expand_alias(target, aliases)
        if expansion:
            target, extra_args = expansion
    elif project and project.host:
        # Use project config host
        target = project.host

    if not target:
        # Check if this is a command that doesn't need target
        if args.connection:
            from rex.commands.connection import connection_status
            return connection_status(None)
        parser.print_help()
        return 1

    # Apply extra args from alias (store partition separately so --gpu/--cpu can override)
    alias_partition = None
    for i, arg in enumerate(extra_args):
        if arg == "-p" and i + 1 < len(extra_args):
            args.python = extra_args[i + 1]
        elif arg == "-s" or arg == "--slurm":
            args.slurm = True
        elif arg == "--partition" and i + 1 < len(extra_args):
            alias_partition = extra_args[i + 1]
        elif arg == "--gres" and i + 1 < len(extra_args):
            args.gres = args.gres or extra_args[i + 1]
        elif arg == "--time" and i + 1 < len(extra_args):
            args.time = args.time or extra_args[i + 1]
        elif arg == "-m" or arg == "--module":
            if i + 1 < len(extra_args):
                args.modules.append(extra_args[i + 1])

    # Set debug mode
    global DEBUG
    DEBUG = args.debug

    # Create SSH executor
    ssh = SSHExecutor(target, verbose=args.debug)

    # Apply project defaults
    if project:
        if not args.modules and project.modules:
            args.modules = project.modules
        if not args.time and project.time:
            args.time = project.time
        if not args.cpus and project.cpus:
            args.cpus = project.cpus

    # Determine partition based on --gpu/--cpu flags
    # Priority: user --partition > --gpu/--cpu > alias partition > default_gpu > cpu
    partition = args.partition
    use_gpu = False
    if not partition:
        if args.gpu and project and project.gpu_partition:
            partition = project.gpu_partition
            use_gpu = True
        elif args.cpu and project and project.cpu_partition:
            partition = project.cpu_partition
        elif alias_partition:
            partition = alias_partition
            # Alias partition is CPU by default (user passes --gpu to override)
        elif project:
            if project.default_gpu:
                partition = project.gpu_partition
                use_gpu = True
            else:
                partition = project.cpu_partition

    # Only apply gres from config if using GPU partition
    gres = args.gres
    if not gres and use_gpu and project and project.gres:
        gres = project.gres

    # Create executor
    executor: Executor
    if args.slurm:
        slurm_opts = SlurmOptions(
            partition=partition,
            gres=gres,
            time=args.time,
            cpus=args.cpus,
        )
        executor = SlurmExecutor(ssh, slurm_opts)
    else:
        executor = DirectExecutor(ssh)

    # Create execution context
    ctx = ExecutionContext(
        target=target,
        python=args.python,
        gpus=args.gpus,
        modules=args.modules,
        code_dir=project.code_dir if project else None,
        run_dir=project.run_dir if project else None,
    )

    # Validate job name if provided
    if args.name:
        try:
            validate_job_name(args.name)
        except ValueError as e:
            error(str(e))

    # Dispatch commands
    if args.connect:
        from rex.commands.connection import connect
        return connect(target)

    if args.disconnect:
        from rex.commands.connection import disconnect
        return disconnect(target)

    if args.connection:
        from rex.commands.connection import connection_status
        return connection_status(target)

    if args.jobs:
        from rex.commands.jobs import list_jobs
        return list_jobs(executor, target, args.json)

    if args.status:
        from rex.commands.jobs import get_last_job, get_status
        job_id = args.status
        if args.last or job_id == "--last":
            job_id = get_last_job(ssh, target)
            if not job_id:
                error("No jobs found")
                return 1
        return get_status(executor, target, job_id, args.json)

    if args.log:
        from rex.commands.jobs import get_last_job, show_log
        job_id = args.log
        if args.last or job_id == "--last":
            job_id = get_last_job(ssh, target)
            if not job_id:
                error("No jobs found")
                return 1
        return show_log(ssh, target, job_id, args.follow)

    if args.kill:
        from rex.commands.jobs import get_last_job, kill_job
        job_id = args.kill
        if args.last or job_id == "--last":
            job_id = get_last_job(ssh, target)
            if not job_id:
                error("No jobs found")
                return 1
        return kill_job(executor, target, job_id)

    if args.watch:
        from rex.commands.jobs import get_last_job, watch_job
        job_id = args.watch
        if args.last or job_id == "--last":
            job_id = get_last_job(ssh, target)
            if not job_id:
                error("No jobs found")
                return 1
        return watch_job(executor, target, job_id, args.json)

    if args.gpu_info:
        from rex.commands.gpus import show_gpus, show_slurm_gpus
        if args.slurm:
            return show_slurm_gpus(ssh, args.partition)
        return show_gpus(ssh, target, args.json)

    if args.push:
        transfer = FileTransfer(target, ssh)
        push_local = Path(args.push[0])
        push_remote = args.push[1] if len(args.push) > 1 else None
        from rex.commands.transfer import push
        return push(transfer, push_local, push_remote)

    if args.pull:
        transfer = FileTransfer(target, ssh)
        pull_remote = args.pull[0]
        pull_local = Path(args.pull[1]) if len(args.pull) > 1 else None
        from rex.commands.transfer import pull
        return pull(transfer, pull_remote, pull_local)

    if args.sync is not None:
        transfer = FileTransfer(target, ssh)
        local_path = Path(args.sync) if args.sync != "." else None
        from rex.commands.transfer import sync
        return sync(
            transfer, ssh, project, local_path,
            code_dir=project.code_dir if project else None,
            python=args.python,
            no_install=args.no_install,
        )

    if args.build:
        if not project:
            error("No .rex.toml found")
            return 1
        from rex.commands.build import build
        return build(ssh, project, args.wait, args.clean, use_gpu=args.gpu)

    if args.exec_cmd:
        from rex.commands.exec import exec_command
        result = exec_command(executor, ctx, args.exec_cmd, args.detach, args.name)
        if isinstance(result, int):
            return result
        return 0

    if args.exec_login:
        # Always use direct executor for login node
        direct = DirectExecutor(ssh)
        from rex.commands.exec import exec_command
        result = exec_command(direct, ctx, args.exec_login, args.detach, args.name)
        if isinstance(result, int):
            return result
        return 0

    # Default: run Python script
    script = Path(args.script) if args.script else None
    script_args = args.script_args or []

    from rex.commands.run import run_python
    result = run_python(executor, ctx, script, script_args, args.detach, args.name)
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    sys.exit(main())
