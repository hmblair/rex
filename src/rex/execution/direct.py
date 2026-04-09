"""Direct SSH execution (non-SLURM)."""

from __future__ import annotations

import time

from rex.execution.base import (
    BaseExecutor, ExecutionContext, JobInfo, JobResult, JobStatus,
    list_job_meta_names, log_path as _log_path, read_job_meta, rex_dir,
    write_job_meta,
)
from rex.execution.script import build_script
from rex.output import success, warn
from rex.ssh.executor import SSHExecutor
from rex.utils import generate_job_name


def _run_detached_nohup(
    ssh: SSHExecutor,
    bash_cmd: str,
    remote_log: str,
    job_name: str,
) -> JobInfo:
    """Run command detached via nohup and return JobInfo."""
    nohup_cmd = (
        f"nohup bash -c '{bash_cmd}' > {remote_log} 2>&1 & "
        f"pid=$!; disown $pid 2>/dev/null; sleep 0.5; echo $pid"
    )

    _, stdout, _ = ssh.exec(nohup_cmd)
    pid = int(stdout.strip()) if stdout.strip() else None

    write_job_meta(ssh, job_name, remote_log, pid=pid)

    target = ssh.target
    success(f"Detached: {job_name} (PID {pid})")
    print(f"Log:    rex {target} --log {job_name}")
    print(f"Kill:   rex {target} --kill {job_name}")

    return JobInfo(
        job_id=job_name,
        log_path=remote_log,
        is_slurm=False,
        pid=pid,
    )


class DirectExecutor(BaseExecutor):
    """Direct SSH execution (non-SLURM).

    Uses nohup for detached jobs, pgrep/kill for job management.
    """

    def __init__(self, ssh: SSHExecutor):
        super().__init__(ssh)

    def _write_script(self, remote_path: str, content: str) -> None:
        """Write a script to the remote host."""
        write_cmd = f"""cat > {remote_path} << 'REXSCRIPT'
{content}
REXSCRIPT
chmod +x {remote_path}"""
        code, _, stderr = self.ssh.exec(write_cmd)
        if code != 0:
            from rex.exceptions import ExecutionError
            raise ExecutionError(f"Failed to write script: {stderr}")

    def exec_foreground(self, ctx: ExecutionContext, cmd: str) -> int:
        """Execute shell command in foreground.

        Runs directly over SSH with output teed to a log file.
        Dies on disconnect — use exec_detached for persistent jobs.
        """
        job_name = generate_job_name()
        remote_dir = rex_dir(ctx.run_dir)
        remote_script = f"{remote_dir}/rex-{job_name}.sh"
        remote_log = _log_path(job_name, ctx.run_dir)

        self.ssh.exec(f"mkdir -p {remote_dir}")
        self._write_script(remote_script, build_script(ctx, cmd))

        exit_code = self.ssh.exec_streaming(
            f"{remote_script} 2>&1 | tee {remote_log}; "
            f"_e=${{PIPESTATUS[0]}}; rm -f {remote_script}; exit $_e",
            tty=True,
        )

        write_job_meta(self.ssh, job_name, remote_log)
        return exit_code

    def exec_detached(
        self, ctx: ExecutionContext, cmd: str, job_name: str
    ) -> JobInfo:
        """Execute shell command detached."""
        remote_dir = rex_dir(ctx.run_dir)
        remote_script = f"{remote_dir}/rex-{job_name}.sh"
        remote_log = _log_path(job_name, ctx.run_dir)

        self.ssh.exec(f"mkdir -p {remote_dir}")
        self._write_script(remote_script, build_script(ctx, cmd))

        return _run_detached_nohup(
            self.ssh, remote_script, remote_log, job_name
        )

    def _pid_from_meta(self, job_id: str) -> int | None:
        """Read PID from job metadata."""
        meta = read_job_meta(self.ssh, job_id)
        return meta.get("pid") if meta else None

    def list_jobs(self, since_minutes: int = 0) -> list[JobStatus]:
        """List all rex jobs on remote."""
        names = list_job_meta_names(self.ssh)
        jobs = []
        for name in names:
            meta = read_job_meta(self.ssh, name)
            if not meta:
                continue
            pid = meta.get("pid")
            if pid:
                code, _, _ = self.ssh.exec(f"kill -0 {pid} 2>/dev/null")
                status = "running" if code == 0 else "completed"
            else:
                status = "completed"
            jobs.append(JobStatus(
                job_id=name,
                status=status,
                pid=pid if status == "running" else None,
                hostname=self.ssh.target,
            ))
        return jobs

    def get_status(self, job_id: str) -> JobStatus:
        """Get status of specific job."""
        pid = self._pid_from_meta(job_id)
        if pid is None:
            return JobStatus(job_id=job_id, status="unknown")

        code, _, _ = self.ssh.exec(f"kill -0 {pid} 2>/dev/null")
        status = "running" if code == 0 else "completed"
        return JobStatus(
            job_id=job_id, status=status, pid=pid if status == "running" else None
        )

    def kill_job(self, job_id: str) -> bool:
        """Kill a running job."""
        pid = self._pid_from_meta(job_id)
        if pid is None:
            warn(f"Job {job_id} not found")
            return False

        code, _, _ = self.ssh.exec(f"kill {pid} 2>/dev/null")
        if code == 0:
            success(f"Killed job {job_id}")
            return True
        warn(f"Failed to kill job {job_id}")
        return False

    def watch_job(self, job_id: str, poll_interval: int = 5) -> JobResult:
        """Wait for job to complete."""
        max_failures = 3
        failures = 0

        while True:
            status = self.get_status(job_id)

            if status.status == "unknown":
                failures += 1
                if failures >= max_failures:
                    warn(f"Lost connection after {max_failures} attempts")
                    return JobResult(job_id=job_id, status="unknown", exit_code=1)
                warn(f"Connection failed (attempt {failures}/{max_failures}), retrying...")
                time.sleep(poll_interval)
                continue

            failures = 0

            if status.status == "completed":
                success(f"Job {job_id} completed")
                return JobResult(job_id=job_id, status="completed", exit_code=0)

            time.sleep(poll_interval)

