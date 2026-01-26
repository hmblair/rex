"""Command-line interface for rex."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rex import __version__
from rex.config import GlobalConfig, HostConfig, ProjectConfig, ResolvedConfig
from rex.exceptions import RexError, ValidationError, ConfigError
from rex.execution import (
    DirectExecutor,
    ExecutionContext,
    Executor,
    SlurmExecutor,
    SlurmOptions,
)
from rex.output import error, setup_logging
from rex.ssh import SSHExecutor, FileTransfer
from rex.utils import (
    validate_job_name,
    validate_slurm_time,
    validate_memory,
    validate_gres,
    validate_cpus,
)

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
  rex gpu --exec "ls -la ~/models"    # run shell command
  rex gpu -s --exec "nvidia-smi"      # run shell command via SLURM
  rex gpu --exec-login "ls -la"       # run on login node (even with -s)
  rex gpu --read ~/data               # read file or list directory
  rex gpu --read                      # list code_dir
  rex gpu --sync                      # sync current project to remote
  rex gpu --connect                   # open persistent connection
  rex gpu --disconnect                # close connection when done
  rex gpu --manual                    # open interactive SSH session
""",
    )

    # Version
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # Target
    parser.add_argument("target", nargs="?", help="alias or user@host")

    # Mode flags
    parser.add_argument("-d", "--detach", action="store_true", help="Run in background")
    parser.add_argument("-s", "--slurm", action="store_true", help="Use SLURM")
    parser.add_argument("-n", "--name", help="Job name for detached jobs")
    parser.add_argument("-p", "--python", default="python3", help="Python interpreter")
    parser.add_argument(
        "-m",
        "--module",
        action="append",
        dest="modules",
        default=[],
        help="Module to load",
    )

    # SLURM options
    parser.add_argument("--partition", help="SLURM partition (overrides --gpu/--cpu)")
    parser.add_argument("--gres", help="SLURM GPU resources")
    parser.add_argument("--time", help="SLURM time limit")
    parser.add_argument("--cpus", type=int, help="SLURM CPUs per task")
    parser.add_argument("--mem", help="SLURM memory allocation (e.g., 4G, 16000M)")
    parser.add_argument("--constraint", help="SLURM node constraint")
    parser.add_argument("--prefer", help="SLURM node preference (soft constraint)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU partition")
    parser.add_argument(
        "--cpu", action="store_true", help="Use CPU partition (override)"
    )

    # Commands (nargs="?" allows --cmd --last without specifying job)
    parser.add_argument(
        "--jobs", action="store_true", help="List jobs (connected hosts if no target)"
    )
    parser.add_argument(
        "--since",
        type=int,
        metavar="MINS",
        help="Include finished jobs from last N minutes",
    )
    parser.add_argument(
        "--status", nargs="?", const="--last", metavar="JOB", help="Check job status"
    )
    parser.add_argument(
        "--log", nargs="?", const="--last", metavar="JOB", help="Show job log"
    )
    parser.add_argument(
        "--kill", nargs="?", const="--last", metavar="JOB", help="Kill job"
    )
    parser.add_argument(
        "--watch", nargs="*", metavar="JOB", help="Wait for job(s) to complete"
    )
    parser.add_argument(
        "--gpu-info", action="store_true", dest="gpu_info", help="Show GPU info"
    )

    parser.add_argument("--push", nargs="+", metavar="PATH", help="Push files")
    parser.add_argument("--pull", nargs="+", metavar="PATH", help="Pull files")
    parser.add_argument(
        "--sync", nargs="?", const=".", metavar="PATH", help="Sync project"
    )

    parser.add_argument("--build", action="store_true", help="Build venv on remote")
    parser.add_argument(
        "--exec", dest="exec_cmd", metavar="CMD", help="Execute shell command"
    )
    parser.add_argument(
        "--exec-code-dir",
        dest="exec_code_dir",
        metavar="CMD",
        help="Execute in code_dir",
    )
    parser.add_argument(
        "--exec-login", dest="exec_login", metavar="CMD", help="Execute on login node"
    )
    parser.add_argument(
        "--read", dest="read_path", nargs="?", const="", metavar="PATH",
        help="Read file or list directory (default: code_dir)"
    )

    parser.add_argument(
        "--connect", action="store_true", help="Open persistent connection"
    )
    parser.add_argument(
        "--disconnect", action="store_true", help="Close persistent connection"
    )
    parser.add_argument(
        "--connection", action="store_true", help="Show connection status"
    )
    parser.add_argument(
        "--manual", action="store_true", help="Open interactive SSH session"
    )

    # Modifiers
    parser.add_argument("--last", action="store_true", help="Use most recent job")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-f", "--follow", action="store_true", help="Follow log")
    parser.add_argument("--no-install", action="store_true", help="Skip pip install")
    parser.add_argument("--clean", action="store_true", help="Clean build")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    # Script and args
    parser.add_argument("script", nargs="?", help="Python script to run")
    parser.add_argument("script_args", nargs="*", help="Script arguments")

    return parser


def merge_configs(
    args: argparse.Namespace,
    project: ProjectConfig | None,
    host_config: HostConfig | None,
) -> tuple[
    str | None,
    str | None,
    str | None,
    int | None,
    str | None,
    str | None,
    str | None,
    list[str],
    bool,
    dict[str, str],
]:
    """Merge configs with priority: CLI > project > host_config > defaults.

    Returns (partition, gres, time, cpus, mem, constraint, prefer, modules, use_gpu, env).
    """
    # Get host config values or defaults
    hc = host_config or HostConfig()

    # Determine partition and use_gpu
    use_gpu = False
    partition = args.partition

    # Get partition sources
    proj_gpu_partition = project.gpu_partition if project else None
    proj_cpu_partition = project.cpu_partition if project else None
    host_gpu_partition = hc.gpu_partition
    host_cpu_partition = hc.cpu_partition

    # Resolve partition preference
    gpu_partition = proj_gpu_partition or host_gpu_partition
    cpu_partition = proj_cpu_partition or host_cpu_partition

    # Determine default_gpu
    default_gpu = False
    if project is not None and project.default_gpu is not None:
        default_gpu = project.default_gpu
    elif hc.default_gpu:
        default_gpu = hc.default_gpu

    if not partition:
        if args.gpu and gpu_partition:
            partition = gpu_partition
            use_gpu = True
        elif args.cpu and cpu_partition:
            partition = cpu_partition
        elif default_gpu and gpu_partition:
            partition = gpu_partition
            use_gpu = True
        else:
            partition = cpu_partition

    # Merge other SLURM options (CLI > project > host)
    def pick(
        cli_val: str | int | None,
        proj_val: str | int | None,
        host_val: str | int | None,
    ) -> str | int | None:
        if cli_val is not None:
            return cli_val
        if proj_val is not None:
            return proj_val
        return host_val

    gres = pick(
        args.gres, project.gres if project else None, hc.gres if use_gpu else None
    )
    time = pick(args.time, project.time if project else None, hc.time)
    cpus = pick(args.cpus, project.cpus if project else None, hc.cpus)
    mem = pick(args.mem, project.mem if project else None, hc.mem)
    # Host-level constraint/prefer are typically GPU-specific (e.g., GPU_SKU:H100),
    # so only apply them on GPU jobs. CLI and project-level values always apply.
    constraint = pick(
        args.constraint,
        project.constraint if project else None,
        hc.constraint if use_gpu else None,
    )
    prefer = pick(
        args.prefer, project.prefer if project else None, hc.prefer if use_gpu else None
    )

    # Merge modules (CLI > project > host)
    if args.modules:
        modules = args.modules
    elif project and project.modules is not None:
        modules = project.modules
    else:
        modules = hc.modules

    # Merge env (host < project, combined)
    env: dict[str, str] = {}
    env.update(hc.env)
    if project:
        env.update(project.env)

    # Cast to proper types for return (pick returns str | int | None but we know the actual types)
    return (
        str(partition) if partition else None,
        str(gres) if gres else None,
        str(time) if time else None,
        int(cpus) if cpus else None,
        str(mem) if mem else None,
        str(constraint) if constraint else None,
        str(prefer) if prefer else None,
        modules,
        use_gpu,
        env,
    )


def resolve_paths(
    project: ProjectConfig | None,
    host_config: HostConfig | None,
) -> tuple[str | None, str | None]:
    """Resolve code_dir and run_dir.

    Returns (code_dir, run_dir).

    If project specifies full path, use it.
    Otherwise: {host.code_dir}/{project.name}
    """
    if not project:
        return None, None

    hc = host_config or HostConfig()

    # Resolve code_dir
    if project.code_dir:
        code_dir = project.code_dir
    elif hc.code_dir:
        code_dir = f"{hc.code_dir}/{project.name}"
    else:
        code_dir = None

    # Resolve run_dir
    if project.run_dir:
        run_dir = project.run_dir
    elif hc.run_dir:
        run_dir = f"{hc.run_dir}/{project.name}"
    else:
        run_dir = None

    return code_dir, run_dir


def resolve_config(
    args: argparse.Namespace,
    project: ProjectConfig | None,
    host_config: HostConfig | None,
) -> ResolvedConfig:
    """Create fully resolved config from CLI args, project, and host config.

    Combines merge_configs() and resolve_paths() into a single ResolvedConfig.
    """
    partition, gres, time, cpus, mem, constraint, prefer, modules, _, env = (
        merge_configs(args, project, host_config)
    )
    code_dir, run_dir = resolve_paths(project, host_config)

    execution = ExecutionContext(
        python=args.python,
        modules=modules,
        code_dir=code_dir,
        run_dir=run_dir,
        env=env if env else None,
    )

    slurm = SlurmOptions(
        partition=partition,
        gres=gres,
        time=time,
        cpus=cpus,
        mem=mem,
        constraint=constraint,
        prefer=prefer,
    )

    return ResolvedConfig(
        name=project.name if project else None,
        root=project.root if project else None,
        execution=execution,
        slurm=slurm,
    )


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    try:
        return _main(argv)
    except RexError as e:
        error(e.message, exit_now=False)
        return e.exit_code
    except KeyboardInterrupt:
        return 130


def _validate_flag_conflicts(args: argparse.Namespace) -> None:
    """Validate that conflicting flags are not used together."""
    # Command flags - only one allowed
    commands = []
    if args.jobs:
        commands.append("--jobs")
    if args.status:
        commands.append("--status")
    if args.log:
        commands.append("--log")
    if args.kill:
        commands.append("--kill")
    if args.watch is not None:
        commands.append("--watch")
    if args.gpu_info:
        commands.append("--gpu-info")
    if args.push:
        commands.append("--push")
    if args.pull:
        commands.append("--pull")
    if args.sync is not None:
        commands.append("--sync")
    if args.build:
        commands.append("--build")
    if args.exec_cmd:
        commands.append("--exec")
    if args.exec_code_dir:
        commands.append("--exec-code-dir")
    if args.exec_login:
        commands.append("--exec-login")
    if args.read_path is not None:
        commands.append("--read")
    if args.connect:
        commands.append("--connect")
    if args.disconnect:
        commands.append("--disconnect")
    if args.connection:
        commands.append("--connection")
    if args.manual:
        commands.append("--manual")

    if len(commands) > 1:
        raise ValidationError(f"Conflicting commands: {', '.join(commands)}")

    # --gpu and --cpu are mutually exclusive
    if args.gpu and args.cpu:
        raise ValidationError("--gpu and --cpu are mutually exclusive")

    # --follow only valid with --log
    if args.follow and not args.log:
        raise ValidationError("--follow requires --log")

    # --clean only valid with --build
    if args.clean and not args.build:
        raise ValidationError("--clean requires --build")

    # --no-install only valid with --sync
    if args.no_install and args.sync is None:
        raise ValidationError("--no-install requires --sync")

    # --last only valid with job commands
    job_commands = args.status or args.log or args.kill or args.watch is not None
    if args.last and not job_commands:
        raise ValidationError("--last requires --status, --log, --kill, or --watch")

    # --since only valid with --jobs
    if args.since and not args.jobs:
        raise ValidationError("--since requires --jobs")


def _main(argv: list[str] | None = None) -> int:
    """Internal main function that may raise RexError."""
    parser = build_parser()
    args = parser.parse_intermixed_args(argv)

    # Validate flag conflicts early
    _validate_flag_conflicts(args)

    # Special case: --connection without target lists all
    if args.connection and not args.target:
        from rex.commands.connection import connection_status

        return connection_status(None)

    # Special case: --jobs without target lists all
    if args.jobs and not args.target:
        from rex.commands.jobs import list_all_jobs

        return list_all_jobs(GlobalConfig.load(), args.json, args.since or 0)

    # Load configs
    global_config = GlobalConfig.load()
    project = ProjectConfig.find_and_load()

    # Resolve target
    target = args.target
    alias_name: str | None = None

    if target:
        # Try alias expansion
        expanded = global_config.expand_alias(target)
        if expanded:
            alias_name = target
            target = expanded

    if not target:
        # Check if this is a command that doesn't need target
        if args.connection:
            from rex.commands.connection import connection_status

            return connection_status(None)
        parser.print_help()
        return 1

    # Get host config for the alias
    host_config = global_config.get_host_config(alias_name) if alias_name else None

    # Apply default_slurm from host config
    if host_config and host_config.default_slurm and not args.slurm:
        args.slurm = True

    # Set debug mode
    global DEBUG
    DEBUG = args.debug
    setup_logging(debug=args.debug)

    # Create SSH executor
    ssh = SSHExecutor(target, verbose=args.debug)

    # Resolve config (CLI > project > host)
    config = resolve_config(args, project, host_config)

    # Create executor
    executor: Executor
    if args.slurm:
        executor = SlurmExecutor(ssh, config.slurm)
    else:
        executor = DirectExecutor(ssh)

    # Use execution context from resolved config
    ctx = config.execution or ExecutionContext()

    # Validate job name if provided
    if args.name:
        try:
            validate_job_name(args.name)
        except ValueError as e:
            raise ValidationError(str(e))

    # Validate SLURM options
    try:
        slurm_opts = config.slurm
        if slurm_opts:
            if slurm_opts.time:
                validate_slurm_time(slurm_opts.time)
            if slurm_opts.mem:
                validate_memory(slurm_opts.mem)
            if slurm_opts.gres:
                validate_gres(slurm_opts.gres)
            if slurm_opts.cpus:
                validate_cpus(slurm_opts.cpus)
    except ValueError as e:
        raise ValidationError(str(e))

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

    if args.manual:
        from rex.commands.connection import manual_ssh

        return manual_ssh(ssh)

    # Verify SSH connection works before running any commands
    ssh.check_connection()

    if args.jobs:
        from rex.commands.jobs import list_jobs

        return list_jobs(executor, args.json, args.since or 0)

    if args.status:
        from rex.commands.jobs import get_last_job, get_status

        job_id = args.status
        if args.last or job_id == "--last":
            job_id = get_last_job(ssh, target)
            if not job_id:
                raise ValidationError("No jobs found")
        return get_status(executor, job_id, args.json)

    if args.log:
        from rex.commands.jobs import get_last_job, show_log

        job_id = args.log
        if args.last or job_id == "--last":
            job_id = get_last_job(ssh, target)
            if not job_id:
                raise ValidationError("No jobs found")
        return show_log(ssh, target, job_id, args.follow)

    if args.kill:
        from rex.commands.jobs import get_last_job, kill_job

        job_id = args.kill
        if args.last or job_id == "--last":
            job_id = get_last_job(ssh, target)
            if not job_id:
                raise ValidationError("No jobs found")
        return kill_job(executor, job_id)

    if args.watch is not None:
        from rex.commands.jobs import get_last_job, watch_jobs

        job_ids = args.watch
        if not job_ids or args.last:
            job_id = get_last_job(ssh, target)
            if not job_id:
                raise ValidationError("No jobs found")
            job_ids = [job_id]
        return watch_jobs(executor, job_ids, args.json)

    if args.gpu_info:
        from rex.commands.gpus import show_gpus, show_slurm_gpus

        if args.slurm:
            partition = config.slurm.partition if config.slurm else None
            return show_slurm_gpus(ssh, partition)
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

        return sync(transfer, ssh, config, local_path, no_install=args.no_install)

    if args.build:
        if not project:
            raise ConfigError("No .rex.toml found")
        from rex.commands.build import build

        result = build(executor, ctx, args.clean)
        if isinstance(result, int):
            return result
        return 0

    if args.exec_cmd:
        from rex.commands.exec import exec_command

        result = exec_command(executor, ctx, args.exec_cmd, args.detach, args.name)
        if isinstance(result, int):
            return result
        return 0

    if args.exec_code_dir:
        from dataclasses import replace
        from rex.commands.exec import exec_command

        # Use code_dir as working directory instead of run_dir
        code_ctx = replace(ctx, run_dir=ctx.code_dir)
        result = exec_command(
            executor, code_ctx, args.exec_code_dir, args.detach, args.name
        )
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

    if args.read_path is not None:
        # Always run on login node
        from rex.commands.read import read_remote

        path = args.read_path or ctx.code_dir
        if not path:
            raise ConfigError("No path specified and code_dir not configured")
        return read_remote(ssh, path)

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
