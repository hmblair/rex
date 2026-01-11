"""Shell command execution."""

from __future__ import annotations

from rex.execution.base import ExecutionContext, Executor, JobInfo


def exec_command(
    executor: Executor,
    ctx: ExecutionContext,
    cmd: str,
    detach: bool,
    job_name: str | None = None,
) -> int | JobInfo:
    """Execute shell command.

    Returns exit code for foreground, JobInfo for detached.
    """
    if detach:
        return executor.exec_detached(ctx, cmd, job_name)
    else:
        return executor.exec_foreground(ctx, cmd)
