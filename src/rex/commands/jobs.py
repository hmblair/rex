"""Job management commands."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any

from rex.execution.base import Executor, JobResult, JobStatus
from rex.output import colorize_status, error, info, success, warn
from rex.ssh.executor import SSHExecutor

if TYPE_CHECKING:
    from rex.config import GlobalConfig


def list_jobs(executor: Executor, json_output: bool = False, since_minutes: int = 0) -> int:
    """List all rex jobs on remote."""
    jobs = executor.list_jobs(since_minutes=since_minutes)

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
        if not jobs:
            return 0

        # Build row data and calculate column widths
        rows: list[tuple[str, str, str, str, str]] = []
        for job in jobs:
            if job.pid:
                info_str = f"PID {job.pid}"
                status = "running"
            elif job.slurm_id:
                info_str = f"SLURM {job.slurm_id}"
                status = job.status
            else:
                info_str = ""
                status = job.status
            rows.append((job.job_id, status, info_str, job.hostname or "", job.description or ""))

        # Calculate max widths
        id_width = max(len(r[0]) for r in rows)
        status_width = max(len(r[1]) for r in rows)
        info_width = max(len(r[2]) for r in rows)
        host_width = max(len(r[3]) for r in rows)

        # Print with aligned columns (pad plain text, then colorize)
        for job_id, status, info_str, host, desc in rows:
            padded_status = status.ljust(status_width)
            colored_status = colorize_status(padded_status)
            parts = [job_id.ljust(id_width), colored_status]
            if info_width > 0:
                parts.append(info_str.ljust(info_width))
            if host_width > 0:
                parts.append(host.ljust(host_width))
            if desc:
                parts.append(desc)
            print("  ".join(parts))

    return 0


def list_all_jobs(
    global_config: "GlobalConfig", json_output: bool = False, since_minutes: int = 0
) -> int:
    """List jobs across all connected hosts."""
    from rex.execution import DirectExecutor, SlurmExecutor
    from rex.ssh.connection import SSHConnection

    active = SSHConnection.list_active()
    if not active:
        warn("No active connections (use --connect first)")
        return 1

    # Build reverse lookup: target -> alias
    target_to_alias = {v: k for k, v in global_config.aliases.items()}

    all_results: dict[str, list[dict[str, Any]]] = {}

    for target, _socket in active:
        alias = target_to_alias.get(target, target)
        try:
            ssh = SSHExecutor(target, verbose=False)

            host_config = global_config.get_host_config(alias) if alias != target else None
            if host_config and host_config.default_slurm:
                executor: Executor = SlurmExecutor(ssh, None)
            else:
                executor = DirectExecutor(ssh)

            jobs = executor.list_jobs(since_minutes=since_minutes)
            if jobs:
                all_results[alias] = []
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
                    all_results[alias].append(item)
        except Exception:
            pass

    if json_output:
        print(json.dumps(all_results, indent=2))
    else:
        if not all_results:
            print("No jobs found")
            return 0

        for alias, jobs in all_results.items():
            print(f"{alias}:")
            for job in jobs:
                status = job["status"]
                parts = [f"  {job['job']}", colorize_status(status)]
                if "slurm_id" in job:
                    parts.append(f"SLURM {job['slurm_id']}")
                elif "pid" in job:
                    parts.append(f"PID {job['pid']}")
                if "hostname" in job:
                    parts.append(job["hostname"])
                if "description" in job:
                    parts.append(job["description"])
                print("  ".join(parts))

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
        print(colorize_status(status.status))

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
        # Get PID if job is running - tail --pid exits when process dies
        # Uses [.] character class so pgrep doesn't match itself
        cmd += f'; pid=$(pgrep -f "rex-{job_id}[.]py" 2>/dev/null | head -1)'
        cmd += '; if [ -n "$pid" ]; then tail -f --pid=$pid "$log"; else cat "$log"; fi'
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
