"""Execution protocol and data types."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from rex.ssh.executor import SSHExecutor


@dataclass
class ExecutionContext:
    """Shared context for execution."""

    modules: list[str] | None = None
    code_dir: str | None = None  # From project config
    run_dir: str | None = None  # Working directory for execution
    env: dict[str, str] | None = None  # Environment variables

    def __post_init__(self):
        if self.modules is None:
            self.modules = []
        if self.env is None:
            self.env = {}


def rex_dir(run_dir: str | None = None) -> str:
    """Return the remote .rex directory for scripts and logs.

    Uses run_dir/.rex if run_dir is set, otherwise ~/.rex.
    """
    if run_dir:
        return f"{run_dir}/.rex"
    return "~/.rex"


def log_path(job_name: str, run_dir: str | None = None) -> str:
    """Return the log file path for a given job name."""
    return f"{rex_dir(run_dir)}/rex-{job_name}.log"


def job_meta_dir(run_dir: str | None = None) -> str:
    """Return the directory for job metadata files."""
    return f"{rex_dir(run_dir)}/jobs"


def job_meta_path(job_name: str, run_dir: str | None = None) -> str:
    """Return the metadata file path for a given job name."""
    return f"{job_meta_dir(run_dir)}/{job_name}.json"


def write_job_meta(
    ssh: "SSHExecutor",
    job_name: str,
    run_dir: str | None,
    log: str,
    *,
    pid: int | None = None,
    slurm_id: int | None = None,
) -> None:
    """Write job metadata to the remote."""
    meta: dict[str, Any] = {"log": log}
    if pid is not None:
        meta["pid"] = pid
    if slurm_id is not None:
        meta["slurm_id"] = slurm_id

    meta_dir = job_meta_dir(run_dir)
    path = job_meta_path(job_name, run_dir)
    payload = json.dumps(meta)
    ssh.exec(f"mkdir -p {meta_dir} && echo '{payload}' > {path}")


def read_job_meta(
    ssh: "SSHExecutor", job_name: str, run_dir: str | None = None
) -> dict[str, Any] | None:
    """Read job metadata from the remote. Returns None if not found."""
    path = job_meta_path(job_name, run_dir)
    code, stdout, _ = ssh.exec(f"cat {path} 2>/dev/null")
    if code != 0 or not stdout.strip():
        return None
    try:
        return json.loads(stdout.strip())
    except json.JSONDecodeError:
        return None


def list_job_meta_names(
    ssh: "SSHExecutor", run_dir: str | None = None
) -> list[str]:
    """List all job names that have metadata files."""
    meta_d = job_meta_dir(run_dir)
    code, stdout, _ = ssh.exec(
        f"ls -t {meta_d}/*.json 2>/dev/null | sed 's|.*/||; s|\\.json$||'"
    )
    if code != 0 or not stdout.strip():
        return []
    return [name for name in stdout.strip().split("\n") if name]


@dataclass
class JobInfo:
    """Returned after launching a detached job."""

    job_id: str
    log_path: str
    is_slurm: bool
    slurm_id: int | None = None
    pid: int | None = None


@dataclass
class JobStatus:
    """Status of a job."""

    job_id: str
    status: str  # "running", "completed", "unknown"
    pid: int | None = None
    slurm_id: int | None = None
    description: str | None = None
    hostname: str | None = None


@dataclass
class JobResult:
    """Result of waiting for a job."""

    job_id: str
    status: str  # "completed", "failed", "unknown"
    exit_code: int


class Executor(Protocol):
    """Protocol for execution backends."""

    def exec_foreground(self, ctx: ExecutionContext, cmd: str) -> int:
        """Execute shell command in foreground."""
        ...

    def exec_detached(
        self, ctx: ExecutionContext, cmd: str, job_name: str
    ) -> JobInfo:
        """Execute shell command detached."""
        ...

    def list_jobs(self, since_minutes: int = 0) -> list[JobStatus]:
        """List all rex jobs.

        Args:
            since_minutes: Include finished jobs from the last N minutes (0 = active only).
        """
        ...

    def get_status(self, job_id: str) -> JobStatus:
        """Get status of specific job."""
        ...

    def get_log_path(self, job_id: str) -> str | None:
        """Get log file path for a job."""
        ...

    def kill_job(self, job_id: str) -> bool:
        """Kill a running job."""
        ...

    def watch_job(self, job_id: str, poll_interval: int = 5) -> JobResult:
        """Wait for job to complete, return final status."""
        ...

    def show_log(self, job_id: str, follow: bool = False) -> int:
        """Show job output log. Returns exit code."""
        ...

    def last_job_id(self) -> str | None:
        """Get the most recent job ID."""
        ...
