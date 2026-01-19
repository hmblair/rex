"""Tests for script building utilities."""

import pytest

from rex.execution.script import quote_with_expansion


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
