"""Bash script builders for remote execution."""

from __future__ import annotations

from __future__ import annotations

import shlex


class ScriptBuilder:
    """Build bash wrapper scripts for execution."""

    def __init__(self):
        self._lines: list[str] = []

    def shebang(self, login: bool = False) -> ScriptBuilder:
        """Add shebang line."""
        if login:
            self._lines.insert(0, "#!/bin/bash -l")
        else:
            self._lines.insert(0, "#!/bin/bash")
        return self

    def module_load(self, modules: list[str]) -> ScriptBuilder:
        """Add module load commands."""
        if modules:
            self._lines.append(f"module load {' '.join(modules)}")
        return self

    def export(self, key: str, value: str) -> ScriptBuilder:
        """Add export statement."""
        self._lines.append(f"export {key}={shlex.quote(value)}")
        return self

    def cd(self, path: str) -> ScriptBuilder:
        """Add cd command."""
        self._lines.append(f"cd {shlex.quote(path)}")
        return self

    def source(self, path: str) -> ScriptBuilder:
        """Add source command (for venv activation)."""
        self._lines.append(f"source {shlex.quote(path)}")
        return self

    def run_python(
        self, python: str, script: str, args: list[str] | None = None
    ) -> ScriptBuilder:
        """Add python execution command."""
        cmd = f"{python} -u {shlex.quote(script)}"
        if args:
            cmd += " " + " ".join(shlex.quote(a) for a in args)
        self._lines.append(cmd)
        return self

    def run_command(self, cmd: str) -> ScriptBuilder:
        """Add arbitrary command."""
        self._lines.append(cmd)
        return self

    def blank_line(self) -> ScriptBuilder:
        """Add blank line for readability."""
        self._lines.append("")
        return self

    def comment(self, text: str) -> ScriptBuilder:
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

    def sbatch_option(self, key: str, value: str) -> SbatchBuilder:
        """Add #SBATCH directive."""
        self._sbatch_options.append(f"#SBATCH --{key}={value}")
        return self

    def job_name(self, name: str) -> SbatchBuilder:
        """Set job name."""
        return self.sbatch_option("job-name", name)

    def output(self, path: str) -> SbatchBuilder:
        """Set output log path."""
        return self.sbatch_option("output", path)

    def partition(self, name: str) -> SbatchBuilder:
        """Set partition."""
        return self.sbatch_option("partition", name)

    def gres(self, spec: str) -> SbatchBuilder:
        """Set GPU resources."""
        return self.sbatch_option("gres", spec)

    def time(self, limit: str) -> SbatchBuilder:
        """Set time limit."""
        return self.sbatch_option("time", limit)

    def build(self) -> str:
        """Return complete sbatch script."""
        # Ensure shebang is first
        lines = []
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
