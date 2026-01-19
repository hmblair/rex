"""Execution protocol and data types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class ExecutionContext:
    """Shared context for execution."""

    target: str
    python: str = "python3"
    modules: list[str] | None = None
    code_dir: str | None = None  # From project config
    run_dir: str | None = None  # Working directory for execution
    env: dict[str, str] | None = None  # Environment variables

    def __post_init__(self):
        if self.modules is None:
            self.modules = []
        if self.env is None:
            self.env = {}


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
    status: str  # "running", "done", "unknown"
    pid: int | None = None
    slurm_id: int | None = None
    description: str | None = None


@dataclass
class JobResult:
    """Result of waiting for a job."""

    job_id: str
    status: str  # "done", "failed", "unknown"
    exit_code: int


class Executor(Protocol):
    """Protocol for execution backends."""

    def run_foreground(
        self, ctx: ExecutionContext, script_path: Path, args: list[str]
    ) -> int:
        """Run script in foreground, streaming output. Returns exit code."""
        ...

    def run_detached(
        self,
        ctx: ExecutionContext,
        script_path: Path,
        args: list[str],
        job_name: str,
    ) -> JobInfo:
        """Run script detached in background."""
        ...

    def exec_foreground(self, ctx: ExecutionContext, cmd: str) -> int:
        """Execute shell command in foreground."""
        ...

    def exec_detached(
        self, ctx: ExecutionContext, cmd: str, job_name: str
    ) -> JobInfo:
        """Execute shell command detached."""
        ...

    def list_jobs(self, target: str) -> list[JobStatus]:
        """List all rex jobs."""
        ...

    def get_status(self, target: str, job_id: str) -> JobStatus:
        """Get status of specific job."""
        ...

    def get_log_path(self, target: str, job_id: str) -> str | None:
        """Get log file path for a job."""
        ...

    def kill_job(self, target: str, job_id: str) -> bool:
        """Kill a running job."""
        ...

    def watch_job(
        self, target: str, job_id: str, poll_interval: int = 5
    ) -> JobResult:
        """Wait for job to complete, return final status."""
        ...
