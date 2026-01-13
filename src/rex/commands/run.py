"""Python script execution command."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from rex.exceptions import ValidationError
from rex.execution.base import ExecutionContext, Executor, JobInfo
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

    Raises:
        ValidationError: If input validation fails.
    """
    # Handle stdin input
    temp_script: Path | None = None
    if script is None:
        if sys.stdin.isatty():
            raise ValidationError(
                "No input file and stdin is a terminal. Provide a file or pipe input."
            )

        # Read stdin to temp file
        content = sys.stdin.read()
        if not content.strip():
            raise ValidationError(
                "No input file and stdin is empty. Provide a file or pipe input."
            )

        if detach:
            raise ValidationError("-d requires a file (cannot detach piped input)")

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            temp_script = Path(f.name)
            script = temp_script

    # Validate script exists
    if not script.exists():
        raise ValidationError(f"Script not found: {script}")

    try:
        if detach:
            name = job_name or generate_job_name()
            return executor.run_detached(ctx, script, args, name)
        else:
            return executor.run_foreground(ctx, script, args)
    finally:
        # Clean up temp file from stdin input
        if temp_script is not None:
            temp_script.unlink(missing_ok=True)
