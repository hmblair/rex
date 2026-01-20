"""SLURM execution (srun/sbatch)."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from rex.exceptions import SSHError
from rex.execution.base import ExecutionContext, JobInfo, JobResult, JobStatus
from rex.execution.script import SbatchBuilder, ScriptBuilder, build_context_commands, get_log_path as _get_log_path
from rex.output import debug, error, success, warn
from rex.ssh.executor import SSHExecutor
from rex.utils import generate_job_name, generate_script_id


def _ssh_write(ssh: SSHExecutor, content: str, remote_path: str, chmod: str | None = None) -> None:
    """Write content to remote file via SSH.

    Raises SSHError on failure with user-friendly message.
    """
    cmd = f"cat > {remote_path}"
    if chmod:
        cmd += f" && chmod {chmod} {remote_path}"

    try:
        result = subprocess.run(
            ["ssh", *ssh._opts, ssh.target, cmd],
            input=content.encode(),
            capture_output=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode().strip() if result.stderr else ""
            raise SSHError(f"Failed to write to {remote_path}: {stderr or 'SSH error'}")
    except subprocess.CalledProcessError as e:
        raise SSHError(f"SSH connection failed while writing to {remote_path}") from e


@dataclass
class SlurmOptions:
    """SLURM-specific options."""

    partition: str | None = None
    gres: str | None = None
    time: str | None = None
    cpus: int | None = None
    mem: str | None = None
    constraint: str | None = None
    prefer: str | None = None


class SlurmExecutor:
    """SLURM-based execution (srun/sbatch).

    Uses ~/.rex directory for logs/scripts (shared filesystem).
    Uses squeue/scancel for job management.
    """

    # Single source of truth: (attribute_name, slurm_option_name)
    OPTION_MAPPINGS = [
        ("partition", "partition"),
        ("gres", "gres"),
        ("time", "time"),
        ("cpus", "cpus-per-task"),
        ("mem", "mem"),
        ("constraint", "constraint"),
        ("prefer", "prefer"),
    ]

    def __init__(self, ssh: SSHExecutor, options: SlurmOptions | None = None):
        self.ssh = ssh
        self.options = options or SlurmOptions()

    def _get_options(self) -> list[tuple[str, str]]:
        """Get all SLURM options as (key, value) pairs."""
        opts = []
        for attr, slurm_key in self.OPTION_MAPPINGS:
            value = getattr(self.options, attr)
            if value is not None:
                opts.append((slurm_key, str(value)))
        return opts

    def _build_slurm_opts(self) -> str:
        """Build command-line options string for srun."""
        opts = self._get_options()
        if not opts:
            return ""
        result = " " + " ".join(f"--{k}={v}" for k, v in opts)
        debug(f"[slurm] options:{result}")
        return result

    def _apply_options_to_builder(self, builder: SbatchBuilder) -> None:
        """Apply all SLURM options to an SbatchBuilder."""
        for key, value in self._get_options():
            builder.sbatch_option(key, value)

    def run_foreground(
        self, ctx: ExecutionContext, script_path: Path, args: list[str]
    ) -> int:
        """Run Python script via srun with streaming output."""
        debug(f"[slurm] run_foreground: {script_path}")
        script_id = generate_script_id()
        script_dir = ctx.run_dir or "$HOME/.rex"
        remote_py = f"{script_dir}/rex-run-{script_id}.py"
        remote_sh = f"{script_dir}/rex-run-{script_id}.sh"

        # Copy Python script to remote
        with open(script_path) as f:
            script_content = f.read()

        self.ssh.exec(f"mkdir -p {script_dir}")
        _ssh_write(self.ssh, script_content, remote_py)

        # Build wrapper script
        builder = ScriptBuilder().shebang(login=True)
        builder.apply_context(ctx)
        builder.run_python(ctx.python, remote_py, args)

        wrapper = builder.build()

        # Write wrapper to remote
        _ssh_write(self.ssh, wrapper, remote_sh, chmod="+x")

        # Execute via srun (force TTY for proper signal forwarding)
        slurm_opts = self._build_slurm_opts()
        exit_code = self.ssh.exec_streaming(
            f"srun{slurm_opts} {remote_sh}; _e=$?; rm -f {remote_py} {remote_sh}; exit $_e",
            tty=True,
        )

        return exit_code

    def run_detached(
        self,
        ctx: ExecutionContext,
        script_path: Path,
        args: list[str],
        job_name: str | None = None,
    ) -> JobInfo:
        """Run Python script via sbatch."""
        if job_name is None:
            job_name = generate_job_name()
        debug(f"[slurm] run_detached: {script_path} as {job_name}")

        # Get remote home for absolute paths
        code, stdout, _ = self.ssh.exec("echo $HOME")
        remote_home = stdout.strip()
        if not remote_home:
            warn("Failed to get remote home directory")
            return JobInfo(job_id=job_name, log_path="", is_slurm=True, slurm_id=None)
        remote_dir = f"{remote_home}/.rex"
        remote_script = f"{remote_dir}/rex-{job_name}.py"
        remote_sbatch = f"{remote_dir}/rex-{job_name}.sbatch"
        remote_log = f"{remote_dir}/rex-{job_name}.log"

        # Create directory and copy script
        self.ssh.exec("mkdir -p ~/.rex")
        with open(script_path) as f:
            script_content = f.read()
        _ssh_write(self.ssh, script_content, remote_script)

        # Build sbatch script
        builder = SbatchBuilder().shebang(login=True)
        builder.job_name(f"rex-{job_name}")
        builder.output(remote_log)
        builder.open_mode("append")
        self._apply_options_to_builder(builder)

        builder.rex_header(f"Script: {remote_script}")
        builder.apply_context(ctx, mkdir_run_dir=True)

        builder.blank_line()
        args_str = " ".join(f"'{a}'" for a in args) if args else ""
        python_cmd = f"{ctx.python} -u {remote_script}"
        if args_str:
            python_cmd += f" {args_str}"
        builder.run_command(python_cmd)

        builder.rex_footer()

        sbatch_content = builder.build()

        # Write sbatch script
        _ssh_write(self.ssh, sbatch_content, remote_sbatch)

        # Submit job
        code, stdout, stderr = self.ssh.exec(f"sbatch --parsable {remote_sbatch}")
        if code != 0 or not stdout.strip():
            err_msg = stderr.strip() or stdout.strip() or "unknown error"
            warn(f"sbatch failed: {err_msg}")
            return JobInfo(
                job_id=job_name,
                log_path=remote_log,
                is_slurm=True,
                slurm_id=None,
            )

        try:
            slurm_id = int(stdout.strip())
        except ValueError:
            warn(f"sbatch returned unexpected output: {stdout.strip()}")
            return JobInfo(
                job_id=job_name,
                log_path=remote_log,
                is_slurm=True,
                slurm_id=None,
            )

        # Write submission info to log (job will append when it starts)
        self.ssh.exec(
            f'echo "[rex] Submitted: $(date)" > {remote_log} && '
            f'echo "[rex] SLURM ID: {slurm_id}" >> {remote_log} && '
            f'echo "[rex] Status: pending" >> {remote_log} && '
            f'echo "---" >> {remote_log}'
        )

        target = self.ssh.target
        success(f"Submitted: {job_name} (SLURM {slurm_id})")
        print(f"Status: rex {target} --status {job_name}")
        print(f"Log:    rex {target} --log {job_name}")
        print(f"Kill:   rex {target} --kill {job_name}")
        return JobInfo(
            job_id=job_name,
            log_path=remote_log,
            is_slurm=True,
            slurm_id=slurm_id,
        )

    def exec_foreground(self, ctx: ExecutionContext, cmd: str) -> int:
        """Execute shell command via srun."""
        # Check for heredoc delimiter collision
        if "\nREXCMD\n" in f"\n{cmd}\n":
            error("Command contains 'REXCMD' as a line, which conflicts with internal delimiter")
            return 1

        debug(f"[slurm] exec_foreground: {cmd[:80]}{'...' if len(cmd) > 80 else ''}")
        script_id = generate_script_id()
        script_dir = ctx.run_dir or "$HOME/.rex"
        remote_script = f"{script_dir}/rex-exec-{script_id}.sh"
        remote_cmd = f"{script_dir}/rex-exec-{script_id}.cmd"

        # Build setup prefix
        context_cmds = build_context_commands(ctx)
        prefix_lines = ["#!/bin/bash -l"] + context_cmds + [f"source {remote_cmd}"]
        script_content = "\n".join(prefix_lines) + "\n"

        # Write command to separate file using heredoc (preserves all quoting)
        self.ssh.exec(f"mkdir -p {script_dir}")
        write_cmd = f"""cat > {remote_cmd} << 'REXCMD'
{cmd}
REXCMD"""
        code, _, stderr = self.ssh.exec(write_cmd)
        if code != 0:
            warn(f"Failed to write command: {stderr}")
            return code

        # Write wrapper script
        _ssh_write(self.ssh, script_content, remote_script, chmod="+x")

        # Execute via srun (force TTY for proper signal forwarding)
        slurm_opts = self._build_slurm_opts()
        exit_code = self.ssh.exec_streaming(
            f"srun{slurm_opts} {remote_script}; _e=$?; rm -f {remote_script} {remote_cmd}; exit $_e",
            tty=True,
        )

        return exit_code

    def exec_detached(
        self, ctx: ExecutionContext, cmd: str, job_name: str | None = None
    ) -> JobInfo:
        """Execute shell command via sbatch."""
        if job_name is None:
            job_name = generate_job_name()
        debug(f"[slurm] exec_detached: {cmd[:80]}{'...' if len(cmd) > 80 else ''} as {job_name}")

        # Get remote home
        code, stdout, _ = self.ssh.exec("echo $HOME")
        remote_home = stdout.strip()
        if not remote_home:
            warn("Failed to get remote home directory")
            return JobInfo(job_id=job_name, log_path="", is_slurm=True, slurm_id=None)
        remote_dir = f"{remote_home}/.rex"
        remote_sbatch = f"{remote_dir}/rex-{job_name}.sbatch"
        remote_log = f"{remote_dir}/rex-{job_name}.log"

        self.ssh.exec("mkdir -p ~/.rex")

        # Build sbatch script
        builder = SbatchBuilder().shebang(login=True)
        builder.job_name(f"rex-{job_name}")
        builder.output(remote_log)
        builder.open_mode("append")
        self._apply_options_to_builder(builder)

        builder.rex_header(f"Command: {cmd}")
        builder.apply_context(ctx)

        builder.blank_line()
        builder.run_command(cmd)

        builder.rex_footer()

        sbatch_content = builder.build()

        # Write and submit
        _ssh_write(self.ssh, sbatch_content, remote_sbatch)

        code, stdout, stderr = self.ssh.exec(f"sbatch --parsable {remote_sbatch}")
        if code != 0 or not stdout.strip():
            err_msg = stderr.strip() or stdout.strip() or "unknown error"
            warn(f"sbatch failed: {err_msg}")
            return JobInfo(
                job_id=job_name,
                log_path=remote_log,
                is_slurm=True,
                slurm_id=None,
            )

        try:
            slurm_id = int(stdout.strip())
        except ValueError:
            warn(f"sbatch returned unexpected output: {stdout.strip()}")
            return JobInfo(
                job_id=job_name,
                log_path=remote_log,
                is_slurm=True,
                slurm_id=None,
            )

        # Write submission info to log (job will append when it starts)
        self.ssh.exec(
            f'echo "[rex] Submitted: $(date)" > {remote_log} && '
            f'echo "[rex] SLURM ID: {slurm_id}" >> {remote_log} && '
            f'echo "[rex] Status: pending" >> {remote_log} && '
            f'echo "---" >> {remote_log}'
        )

        target = self.ssh.target
        success(f"Submitted: {job_name} (SLURM {slurm_id})")
        print(f"Log:    rex {target} --log {job_name}")
        print(f"Kill:   rex {target} --kill {job_name}")
        return JobInfo(
            job_id=job_name,
            log_path=remote_log,
            is_slurm=True,
            slurm_id=slurm_id,
        )

    def list_jobs(self, target: str) -> list[JobStatus]:
        """List all rex SLURM jobs."""
        code, stdout, _ = self.ssh.exec(
            "squeue -u $USER -o '%.10i %.30j %.8T %.10M' 2>/dev/null | grep rex"
        )

        jobs = []
        for line in stdout.strip().split("\n"):
            if not line or "rex" not in line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    slurm_id = int(parts[0])
                except ValueError:
                    continue  # Skip malformed lines
                name = parts[1]
                status = parts[2].lower()
                # Extract job_id from name (rex-<job_id>)
                job_id = name.replace("rex-", "") if name.startswith("rex-") else name
                jobs.append(JobStatus(
                    job_id=job_id,
                    status="running" if status in ("pending", "running") else status,
                    slurm_id=slurm_id,
                ))
        return jobs

    def get_status(self, target: str, job_id: str) -> JobStatus:
        """Get status of specific SLURM job."""
        code, stdout, _ = self.ssh.exec(
            f"squeue -u $USER -n rex-{job_id} -h -o %T 2>/dev/null"
        )

        state = stdout.strip()
        if state:
            return JobStatus(job_id=job_id, status=state.lower())
        return JobStatus(job_id=job_id, status="done")

    def get_log_path(self, target: str, job_id: str) -> str | None:
        """Get log file path."""
        return _get_log_path(self.ssh, job_id)

    def kill_job(self, target: str, job_id: str) -> bool:
        """Cancel SLURM job."""
        code, _, _ = self.ssh.exec(f"scancel -n rex-{job_id}")
        if code == 0:
            success(f"Cancelled job {job_id}")
            return True
        warn(f"Failed to cancel job {job_id}")
        return False

    def watch_job(
        self, target: str, job_id: str, poll_interval: int = 10
    ) -> JobResult:
        """Wait for SLURM job to complete."""
        max_failures = 3
        failures = 0

        while True:
            code, stdout, _ = self.ssh.exec(
                f"squeue -u $USER -n rex-{job_id} -h -o %T 2>/dev/null"
            )

            if code != 0:
                failures += 1
                if failures >= max_failures:
                    warn(f"Lost connection after {max_failures} attempts")
                    return JobResult(job_id=job_id, status="unknown", exit_code=1)
                warn(f"Connection failed (attempt {failures}/{max_failures}), retrying...")
                time.sleep(poll_interval)
                continue

            failures = 0
            state = stdout.strip()

            if not state:
                # Job no longer in queue - check sacct for final status
                code, stdout, _ = self.ssh.exec(
                    f"sacct -n -X --name=rex-{job_id} --format=State 2>/dev/null | head -1 | tr -d ' '"
                )
                sacct_status = stdout.strip()

                if sacct_status == "COMPLETED":
                    success(f"Job {job_id} completed: success")
                    return JobResult(job_id=job_id, status="success", exit_code=0)
                elif sacct_status in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                    warn(f"Job {job_id} finished: {sacct_status.lower()}")
                    return JobResult(job_id=job_id, status="failed", exit_code=1)
                else:
                    success(f"Job {job_id} completed")
                    return JobResult(job_id=job_id, status="done", exit_code=0)

            time.sleep(poll_interval)
