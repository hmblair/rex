"""SLURM execution (srun/sbatch)."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from rex.exceptions import SSHError
from rex.execution.base import (
    ExecutionContext, JobInfo, JobResult, JobStatus,
    list_job_meta_names, log_path as _log_path, read_job_meta, rex_dir,
    write_job_meta,
)
from rex.execution.script import SbatchBuilder, build_context_commands
from rex.output import debug, error, success, warn
from rex.ssh.executor import SSHExecutor
from rex.utils import generate_script_id


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

    Stages scripts and logs in run_dir/.rex (or $HOME/.rex if no run_dir).
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

    def __init__(self, ssh: SSHExecutor, options: SlurmOptions | None = None, run_dir: str | None = None):
        self.ssh = ssh
        self.options = options or SlurmOptions()
        self.run_dir = run_dir

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


    def exec_foreground(self, ctx: ExecutionContext, cmd: str) -> int:
        """Execute shell command via srun."""
        # Check for heredoc delimiter collision
        if "\nREXCMD\n" in f"\n{cmd}\n":
            error("Command contains 'REXCMD' as a line, which conflicts with internal delimiter")
            return 1

        debug(f"[slurm] exec_foreground: {cmd[:80]}{'...' if len(cmd) > 80 else ''}")
        script_id = generate_script_id()
        remote_dir = rex_dir(ctx.run_dir)
        remote_script = f"{remote_dir}/rex-exec-{script_id}.sh"
        remote_cmd = f"{remote_dir}/rex-exec-{script_id}.cmd"

        # Build setup prefix
        context_cmds = build_context_commands(ctx)
        prefix_lines = ["#!/bin/bash -l"] + context_cmds + [f"source {remote_cmd}"]
        script_content = "\n".join(prefix_lines) + "\n"

        # Write command to separate file using heredoc (preserves all quoting)
        self.ssh.exec(f"mkdir -p {remote_dir}")
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
        self, ctx: ExecutionContext, cmd: str, job_name: str
    ) -> JobInfo:
        """Execute shell command via sbatch."""
        debug(f"[slurm] exec_detached: {cmd[:80]}{'...' if len(cmd) > 80 else ''} as {job_name}")

        remote_dir = rex_dir(ctx.run_dir)
        remote_sbatch = f"{remote_dir}/rex-{job_name}.sbatch"
        remote_log = _log_path(job_name, ctx.run_dir)

        self.ssh.exec(f"mkdir -p {remote_dir}")

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

        write_job_meta(
            self.ssh, job_name, ctx.run_dir, remote_log, slurm_id=slurm_id
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

    def list_jobs(self, since_minutes: int = 0) -> list[JobStatus]:
        """List all rex SLURM jobs."""
        # Get active jobs from squeue
        code, stdout, _ = self.ssh.exec(
            "squeue -u $USER -o '%.10i %.30j %.12T %.10M' 2>/dev/null | grep rex"
        )

        jobs = []
        seen_ids: set[int] = set()

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
                    status=status,
                    slurm_id=slurm_id,
                    hostname=self.ssh.target,
                ))
                seen_ids.add(slurm_id)

        # Get recently finished jobs from sacct
        if since_minutes > 0:
            code, stdout, _ = self.ssh.exec(
                f"sacct -u $USER --starttime=now-{since_minutes}minutes "
                f"-o JobID,JobName,State --parsable2 2>/dev/null | grep -E '^[0-9]+\\|rex-'"
            )
            for line in stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        slurm_id = int(parts[0])
                    except ValueError:
                        continue
                    if slurm_id in seen_ids:
                        continue  # Skip if already in squeue results
                    name = parts[1]
                    status = parts[2].lower()
                    # Handle "cancelled by ..." status
                    if status.startswith("cancelled"):
                        status = "cancelled"
                    job_id = name.replace("rex-", "") if name.startswith("rex-") else name
                    jobs.append(JobStatus(
                        job_id=job_id,
                        status=status,
                        slurm_id=slurm_id,
                        hostname=self.ssh.target,
                    ))

        return jobs

    def _query_job_state(self, job_id: str) -> str:
        """Query SLURM for a job's current state.

        Checks squeue first (active jobs), then falls back to sacct
        (finished jobs). Returns a lowercase state string, or "completed"
        if neither source has information.
        """
        code, stdout, _ = self.ssh.exec(
            f"squeue -u $USER -n rex-{job_id} -h -o %T 2>/dev/null"
        )
        if code != 0:
            raise SSHError("squeue query failed")

        state = stdout.strip()
        if state:
            return state.lower()

        # Job not in squeue — check sacct for final status
        code, stdout, _ = self.ssh.exec(
            f"sacct -n -X --name=rex-{job_id} --format=State 2>/dev/null | head -1 | tr -d ' '"
        )
        sacct_status = stdout.strip().upper()
        if sacct_status in (
            "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            "COMPLETED", "PENDING", "RUNNING", "REQUEUED",
        ):
            return sacct_status.lower()

        return "completed"

    def get_status(self, job_id: str) -> JobStatus:
        """Get status of specific SLURM job."""
        try:
            state = self._query_job_state(job_id)
        except SSHError:
            return JobStatus(job_id=job_id, status="unknown")
        return JobStatus(job_id=job_id, status=state)

    def get_log_path(self, job_id: str) -> str | None:
        """Get log file path."""
        meta = read_job_meta(self.ssh, job_id, self.run_dir)
        return meta.get("log") if meta else None

    def kill_job(self, job_id: str) -> bool:
        """Cancel SLURM job."""
        code, _, _ = self.ssh.exec(f"scancel -n rex-{job_id}")
        if code == 0:
            success(f"Cancelled job {job_id}")
            return True
        warn(f"Failed to cancel job {job_id}")
        return False

    def watch_job(self, job_id: str, poll_interval: int = 10) -> JobResult:
        """Wait for SLURM job to complete."""
        max_failures = 3
        failures = 0

        while True:
            try:
                state = self._query_job_state(job_id)
            except SSHError:
                failures += 1
                if failures >= max_failures:
                    warn(f"Lost connection after {max_failures} attempts")
                    return JobResult(job_id=job_id, status="unknown", exit_code=1)
                warn(f"Connection failed (attempt {failures}/{max_failures}), retrying...")
                time.sleep(poll_interval)
                continue

            failures = 0

            if state in ("running", "pending", "requeued", "configuring"):
                time.sleep(poll_interval)
                continue

            if state == "completed":
                success(f"Job {job_id} completed")
                return JobResult(job_id=job_id, status="completed", exit_code=0)

            warn(f"Job {job_id} finished: {state}")
            return JobResult(job_id=job_id, status=state, exit_code=1)

    def show_log(self, job_id: str, follow: bool = False) -> int:
        """Show job output log."""
        meta = read_job_meta(self.ssh, job_id, self.run_dir)
        if not meta or "log" not in meta:
            error("Log not found", exit_now=False)
            return 1

        log = meta["log"]
        cmd = f'[ -f {log} ] || {{ echo "Log not found" >&2; exit 1; }}'

        if follow:
            pid = meta.get("pid")
            if pid:
                cmd += f'; if kill -0 {pid} 2>/dev/null; then tail -f --pid={pid} {log}; else cat {log}; fi'
            else:
                cmd += f'; cat {log}'
        else:
            cmd += f'; cat {log}'

        return self.ssh.exec_streaming(cmd, tty=follow)

    def last_job_id(self) -> str | None:
        """Get the most recent job ID."""
        names = list_job_meta_names(self.ssh, self.run_dir)
        return names[0] if names else None
