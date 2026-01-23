"""Build command for remote venv setup."""

from __future__ import annotations

from rex.exceptions import ConfigError
from rex.execution.base import ExecutionContext, Executor, JobInfo
from rex.output import info
from rex.utils import generate_job_name


def _build_script(ctx: ExecutionContext, clean: bool = False) -> str:
    """Generate the build script content."""
    module_cmds = ""
    if ctx.modules:
        module_cmds = f"module load {' '.join(ctx.modules)}"

    clean_cmd = "rm -rf .venv" if clean else ""

    return f"""set -e
echo "=== Rex Build ==="
echo "Started: $(date)"

{module_cmds}

cd {ctx.code_dir}
{clean_cmd}

if [[ ! -d .venv ]]; then
    echo "=== Creating venv ==="
    python3 -m venv .venv
fi

echo "=== Installing package ==="
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --only-binary :all: -e .

echo "=== Build complete ==="
echo "Finished: $(date)"
"""


def build(
    executor: Executor,
    ctx: ExecutionContext,
    clean: bool = False,
) -> int | JobInfo:
    """Create/update venv on remote via exec_detached.

    Returns JobInfo so users can track with --log, --watch, --status, --kill.
    """
    if not ctx or not ctx.code_dir:
        raise ConfigError("code_dir not configured")

    info(f"Building in {ctx.code_dir}")

    job_name = f"build-{generate_job_name()}"
    script = _build_script(ctx, clean)

    return executor.exec_detached(ctx, script, job_name)
