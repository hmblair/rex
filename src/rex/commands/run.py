"""Python script execution command."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from rex.execution.base import ExecutionContext, Executor, JobInfo
from rex.output import error
from rex.utils import generate_job_name


def run_python(
    executor: Executor,
    ctx: ExecutionContext,
    script: Path | None,
    args: list[str],
    detach: bool,
    job_name: str | None = None,
) -> int | JobInfo:
    """Execute Python script.

    If script is None, reads from stdin.
    Returns exit code for foreground, JobInfo for detached.
    """
    # Handle stdin input
    if script is None:
        if sys.stdin.isatty():
            error("No input file and stdin is a terminal. Provide a file or pipe input.")
            return 1

        # Read stdin to temp file
        content = sys.stdin.read()
        if not content.strip():
            error("No input file and stdin is empty. Provide a file or pipe input.")
            return 1

        if detach:
            error("-d requires a file (cannot detach piped input)")
            return 1

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            script = Path(f.name)

    # Validate script exists
    if not script.exists():
        error(f"Script not found: {script}")
        return 1

    if detach:
        name = job_name or generate_job_name()
        return executor.run_detached(ctx, script, args, name)
    else:
        return executor.run_foreground(ctx, script, args)
