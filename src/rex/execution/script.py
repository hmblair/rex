"""Bash script builders for remote execution."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self


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
        self._lines.append(f"export {key}={shlex.quote(value)}")
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

    def partition(self, name: str) -> Self:
        """Set partition."""
        return self.sbatch_option("partition", name)

    def gres(self, spec: str) -> Self:
        """Set GPU resources."""
        return self.sbatch_option("gres", spec)

    def time(self, limit: str) -> Self:
        """Set time limit."""
        return self.sbatch_option("time", limit)

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
