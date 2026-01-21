"""Job management commands."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from rex.execution.base import Executor, JobResult, JobStatus
from rex.output import colorize_status, error, info, success, warn
from rex.ssh.executor import SSHExecutor


def list_jobs(executor: Executor, json_output: bool = False) -> int:
    """List all rex jobs on remote."""
    jobs = executor.list_jobs()

    if json_output:
        output: list[dict[str, Any]] = []
        for job in jobs:
            item: dict[str, Any] = {"job": job.job_id, "status": job.status}
            if job.pid:
                item["pid"] = job.pid
            if job.slurm_id:
                item["slurm_id"] = job.slurm_id
            if job.hostname:
                item["hostname"] = job.hostname
            if job.description:
                item["description"] = job.description
            output.append(item)
        print(json.dumps(output, indent=2))
    else:
        for job in jobs:
            status_str = colorize_status(job.status)
            host = job.hostname or ""
            if job.pid:
                print(f"{job.job_id:<20} {colorize_status('running')} (PID {job.pid})  {host}  {job.description or ''}")
            elif job.slurm_id:
                print(f"{job.job_id:<20} {status_str} (SLURM {job.slurm_id})  {host}")
            else:
                print(f"{job.job_id:<20} {status_str:<20} {host}  {job.description or ''}")

    return 0


def get_status(
    executor: Executor, job_id: str, json_output: bool = False
) -> int:
    """Get status of specific job."""
    status = executor.get_status(job_id)

    if json_output:
        out: dict[str, Any] = {"job": status.job_id, "status": status.status}
        if status.pid:
            out["pid"] = status.pid
        if status.slurm_id:
            out["slurm_id"] = status.slurm_id
        print(json.dumps(out))
    else:
        if status.status == "unknown":
            warn("Could not determine job status")
        print(status.status)

    return 0 if status.status == "running" else 1


def show_log(
    ssh: SSHExecutor, target: str, job_id: str, follow: bool = False
) -> int:
    """Show job output log."""
    # Find log path
    cmd = (
        f'log=$(if [ -f ~/.rex/rex-{job_id}.log ]; then echo ~/.rex/rex-{job_id}.log; '
        f'elif [ -f /tmp/rex-{job_id}.log ]; then echo /tmp/rex-{job_id}.log; fi); '
        f'[ -n "$log" ] || {{ echo "Log not found" >&2; exit 1; }}'
    )

    if follow:
        cmd += '; tail -f "$log"'
    else:
        cmd += '; cat "$log"'

    return ssh.exec_streaming(cmd, tty=follow)


def kill_job(executor: Executor, job_id: str) -> int:
    """Kill a running job."""
    if executor.kill_job(job_id):
        return 0
    return 1


def watch_job(
    executor: Executor, job_id: str, json_output: bool = False
) -> int:
    """Wait for job to complete."""
    info(f"Watching job {job_id}...")
    result = executor.watch_job(job_id)

    if json_output:
        print(json.dumps({
            "job": result.job_id,
            "status": result.status,
            "exit_code": result.exit_code,
        }))

    return result.exit_code


def get_last_job(ssh: SSHExecutor, target: str) -> str | None:
    """Get most recent job ID from remote."""
    code, stdout, _ = ssh.exec(
        "ls -t ~/.rex/rex-*.log /tmp/rex-*.log 2>/dev/null | head -1 | "
        "sed 's|.*/rex-||; s|\\.log$||'"
    )
    return stdout.strip() if stdout.strip() else None
