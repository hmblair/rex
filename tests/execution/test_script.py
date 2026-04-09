"""Tests for script building utilities."""

import pytest

from rex.execution.base import ExecutionContext
from rex.execution.script import build_script, quote_with_expansion


class TestQuoteWithExpansion:
    """Tests for quote_with_expansion function."""

    def test_simple_value_uses_single_quotes(self):
        """Simple values without variables use shlex.quote (single quotes)."""
        assert quote_with_expansion("simple") == "simple"
        assert quote_with_expansion("value") == "value"

    def test_path_without_vars(self):
        """Paths without variables use single quotes."""
        assert quote_with_expansion("/path/to/bin") == "/path/to/bin"

    def test_value_with_spaces_uses_single_quotes(self):
        """Values with spaces are single-quoted."""
        assert quote_with_expansion("has spaces") == "'has spaces'"

    def test_dollar_var_uses_double_quotes(self):
        """$VAR syntax triggers double quoting for expansion."""
        assert quote_with_expansion("$PATH:/bin") == '"$PATH:/bin"'

    def test_brace_var_uses_double_quotes(self):
        """${VAR} syntax triggers double quoting for expansion."""
        assert quote_with_expansion("${HOME}/bin") == '"${HOME}/bin"'

    def test_multiple_vars(self):
        """Multiple variables in one value."""
        assert quote_with_expansion("$PATH:$HOME/bin") == '"$PATH:$HOME/bin"'

    def test_escapes_double_quotes(self):
        """Double quotes inside value are escaped."""
        result = quote_with_expansion('$VAR "quoted"')
        assert result == '"$VAR \\"quoted\\""'

    def test_escapes_backticks(self):
        """Backticks are escaped to prevent command substitution."""
        result = quote_with_expansion("$PATH:`whoami`")
        assert result == '"$PATH:\\`whoami\\`"'

    def test_escapes_dollar_paren(self):
        """$() command substitution is escaped."""
        result = quote_with_expansion("$PATH:$(whoami)")
        assert result == '"$PATH:\\$(whoami)"'

    def test_escapes_backslashes(self):
        """Backslashes are escaped."""
        result = quote_with_expansion("$PATH:\\bin")
        assert result == '"$PATH:\\\\bin"'

    def test_no_expansion_for_escaped_dollar(self):
        """Literal dollar sign without valid var name uses single quotes."""
        assert quote_with_expansion("costs $5") == "'costs $5'"

    def test_underscore_in_var_name(self):
        """Underscores in variable names are valid."""
        assert quote_with_expansion("$MY_VAR/path") == '"$MY_VAR/path"'

    def test_var_with_numbers(self):
        """Numbers in variable names (not first char) are valid."""
        assert quote_with_expansion("$VAR123/path") == '"$VAR123/path"'


class TestBuildScript:
    """Tests for build_script helper."""

    def test_includes_shebang(self):
        """Script starts with login shell shebang."""
        result = build_script(ExecutionContext(), "echo hello")
        assert result.startswith("#!/bin/bash -l\n")

    def test_includes_command(self):
        """Command appears in the script."""
        result = build_script(ExecutionContext(), "python train.py")
        assert "python train.py" in result

    def test_trailing_newline(self):
        """Script ends with a newline."""
        result = build_script(ExecutionContext(), "echo hello")
        assert result.endswith("\n")

    def test_empty_context(self):
        """No context lines when context is default."""
        result = build_script(ExecutionContext(), "echo hello")
        assert result == "#!/bin/bash -l\necho hello\n"

    def test_applies_modules(self):
        """Modules from context appear in script."""
        ctx = ExecutionContext(modules=["python/3.11", "cuda/12"])
        result = build_script(ctx, "echo hello")
        assert "module load python/3.11 cuda/12" in result

    def test_applies_env(self):
        """Environment variables are exported."""
        ctx = ExecutionContext(env={"MY_VAR": "value"})
        result = build_script(ctx, "echo hello")
        assert "export MY_VAR=" in result

    def test_applies_venv(self):
        """Venv activation added when code_dir is set."""
        ctx = ExecutionContext(code_dir="/projects/myexp")
        result = build_script(ctx, "echo hello")
        assert ".venv/bin/activate" in result

    def test_applies_run_dir(self):
        """cd to run_dir when set."""
        ctx = ExecutionContext(run_dir="/scratch/runs/exp1")
        result = build_script(ctx, "echo hello")
        assert "/scratch/runs/exp1" in result
