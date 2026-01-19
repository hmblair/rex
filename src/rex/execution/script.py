"""Bash script builders for remote execution."""

from __future__ import annotations

import re
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rex.execution.base import ExecutionContext
    from rex.ssh.executor import SSHExecutor
    from typing_extensions import Self


def quote_with_expansion(value: str) -> str:
    """Quote a value for shell, allowing $VAR expansion if present.

    If the value contains shell variable references ($VAR or ${VAR}),
    uses double quotes and escapes dangerous characters while preserving
    variable expansion. Otherwise, uses shlex.quote() for full escaping.
    """
    if re.search(r'\$[A-Za-z_][A-Za-z0-9_]*|\$\{[^}]+\}', value):
        escaped = value.replace('\\', '\\\\')
        escaped = escaped.replace('"', '\\"')
        escaped = escaped.replace('`', '\\`')
        escaped = escaped.replace('$(', '\\$(')
        return f'"{escaped}"'
    return shlex.quote(value)


def get_log_path(ssh: "SSHExecutor", job_id: str) -> str | None:
    """Get log file path for a job.

    Checks ~/.rex/ first, then /tmp/.
    """
    cmd = (
        f'if [ -f ~/.rex/rex-{job_id}.log ]; then echo ~/.rex/rex-{job_id}.log; '
        f'elif [ -f /tmp/rex-{job_id}.log ]; then echo /tmp/rex-{job_id}.log; fi'
    )
    _, stdout, _ = ssh.exec(cmd)
    return stdout.strip() if stdout.strip() else None


def build_context_commands(
    ctx: "ExecutionContext",
    *,
    mkdir_run_dir: bool = False,
) -> list[str]:
    """Build shell commands for execution context setup.

    Returns a list of commands that can be joined with any separator.
    Handles: modules, env vars, venv activation, working directory.

    Args:
        ctx: Execution context with modules, env, code_dir, run_dir.
        mkdir_run_dir: If True, create run_dir before cd.
    """
    commands: list[str] = []

    if ctx.modules:
        commands.append(f"module load {' '.join(ctx.modules)}")

    for key, value in ctx.env.items():
        commands.append(f"export {key}={quote_with_expansion(value)}")

    if ctx.code_dir:
        commands.append(f"source {shlex.quote(ctx.code_dir + '/.venv/bin/activate')}")

    if ctx.run_dir:
        if mkdir_run_dir:
            commands.append(f"mkdir -p {shlex.quote(ctx.run_dir)} && cd {shlex.quote(ctx.run_dir)}")
        else:
            commands.append(f"cd {shlex.quote(ctx.run_dir)}")

    return commands


class ScriptBuilder:
    """Build bash wrapper scripts for execution."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def shebang(self, login: bool = False) -> Self:
        """Add shebang line."""
        if login:
            self._lines.insert(0, "#!/bin/bash -l")
        else:
            self._lines.insert(0, "#!/bin/bash")
        return self

    def module_load(self, modules: list[str]) -> Self:
        """Add module load commands."""
        if modules:
            self._lines.append(f"module load {' '.join(modules)}")
        return self

    def export(self, key: str, value: str) -> Self:
        """Add export statement."""
        self._lines.append(f"export {key}={quote_with_expansion(value)}")
        return self

    def cd(self, path: str) -> Self:
        """Add cd command."""
        self._lines.append(f"cd {shlex.quote(path)}")
        return self

    def source(self, path: str) -> Self:
        """Add source command (for venv activation)."""
        self._lines.append(f"source {shlex.quote(path)}")
        return self

    def run_python(
        self, python: str, script: str, args: list[str] | None = None
    ) -> Self:
        """Add python execution command."""
        cmd = f"{python} -u {shlex.quote(script)}"
        if args:
            cmd += " " + " ".join(shlex.quote(a) for a in args)
        self._lines.append(cmd)
        return self

    def run_command(self, cmd: str) -> Self:
        """Add arbitrary command."""
        self._lines.append(cmd)
        return self

    def apply_context(
        self,
        ctx: "ExecutionContext",
        *,
        mkdir_run_dir: bool = False,
    ) -> Self:
        """Apply execution context (modules, env, venv, cd).

        Args:
            ctx: Execution context.
            mkdir_run_dir: If True, create run_dir before cd.
        """
        if ctx.modules:
            self.module_load(ctx.modules)
        for key, value in ctx.env.items():
            self.export(key, value)
        if ctx.code_dir:
            self.source(f"{ctx.code_dir}/.venv/bin/activate")
        if ctx.run_dir:
            if mkdir_run_dir:
                self.run_command(f"mkdir -p {shlex.quote(ctx.run_dir)} && cd {shlex.quote(ctx.run_dir)}")
            else:
                self.cd(ctx.run_dir)
        return self

    def blank_line(self) -> Self:
        """Add blank line for readability."""
        self._lines.append("")
        return self

    def comment(self, text: str) -> Self:
        """Add comment."""
        self._lines.append(f"# {text}")
        return self

    def build(self) -> str:
        """Return complete script."""
        return "\n".join(self._lines) + "\n"


class SbatchBuilder(ScriptBuilder):
    """Build sbatch scripts with SLURM directives."""

    def __init__(self):
        super().__init__()
        self._sbatch_options: list[str] = []

    def sbatch_option(self, key: str, value: str) -> Self:
        """Add #SBATCH directive."""
        self._sbatch_options.append(f"#SBATCH --{key}={value}")
        return self

    def job_name(self, name: str) -> Self:
        """Set job name."""
        return self.sbatch_option("job-name", name)

    def output(self, path: str) -> Self:
        """Set output log path."""
        return self.sbatch_option("output", path)

    def open_mode(self, mode: str) -> Self:
        """Set output open mode (append or truncate)."""
        return self.sbatch_option("open-mode", mode)

    def partition(self, name: str) -> Self:
        """Set partition."""
        return self.sbatch_option("partition", name)

    def gres(self, spec: str) -> Self:
        """Set GPU resources."""
        return self.sbatch_option("gres", spec)

    def time(self, limit: str) -> Self:
        """Set time limit."""
        return self.sbatch_option("time", limit)

    def rex_header(self, description: str) -> Self:
        """Add standard rex job header."""
        self.blank_line()
        self.run_command('echo "[rex] Running: $(date)"')
        self.run_command('echo "[rex] Host: $(hostname)"')
        self.run_command(f'echo "[rex] {description}"')
        self.run_command('echo "---"')
        return self

    def rex_footer(self) -> Self:
        """Add standard rex job footer with exit code handling."""
        self.blank_line()
        self.run_command('_rex_exit=$?')
        self.run_command('echo "---"')
        self.run_command('echo "[rex] Finished: $(date)"')
        self.run_command('echo "[rex] Exit code: $_rex_exit"')
        self.run_command('exit $_rex_exit')
        return self

    def build(self) -> str:
        """Return complete sbatch script."""
        # Ensure shebang is first
        lines: list[str] = []
        for line in self._lines:
            if line.startswith("#!"):
                lines.insert(0, line)
            else:
                lines.append(line)

        # Insert sbatch options after shebang
        if self._sbatch_options:
            insert_pos = 1 if lines and lines[0].startswith("#!") else 0
            for i, opt in enumerate(self._sbatch_options):
                lines.insert(insert_pos + i, opt)

        return "\n".join(lines) + "\n"
