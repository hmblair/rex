"""Tests for direct (non-SLURM) execution."""

import pytest
from unittest.mock import MagicMock, patch

from rex.execution.direct import DirectExecutor, _shell_quote
from rex.execution.base import ExecutionContext


class TestShellQuote:
    """Tests for _shell_quote helper function."""

    def test_simple_string(self):
        """Simple strings are wrapped in single quotes."""
        assert _shell_quote("hello") == "'hello'"

    def test_string_with_spaces(self):
        """Strings with spaces are properly quoted."""
        assert _shell_quote("hello world") == "'hello world'"

    def test_string_with_single_quote(self):
        """Single quotes are escaped."""
        result = _shell_quote("it's")
        assert result == "'it'\\''s'"

    def test_double_quotes_preserved(self):
        """Double quotes are preserved."""
        result = _shell_quote('echo "hello"')
        assert result == '\'echo "hello"\''

    def test_dollar_sign_preserved(self):
        """Dollar signs are preserved."""
        result = _shell_quote("echo $HOME")
        assert result == "'echo $HOME'"

    def test_pipe_preserved(self):
        """Pipes are preserved."""
        result = _shell_quote("ls | grep foo")
        assert result == "'ls | grep foo'"


class TestDirectExecutorExecForeground:
    """Tests for DirectExecutor.exec_foreground with special characters."""

    @pytest.fixture
    def mock_ssh(self):
        """Create a mock SSH executor."""
        ssh = MagicMock()
        ssh.exec.return_value = (0, "", "")
        ssh.exec_script_streaming.return_value = 0
        return ssh

    def test_exec_foreground_double_quotes(self, mock_ssh):
        """exec_foreground preserves double quotes."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        executor.exec_foreground(ctx, 'echo "hello world"')

        # Check the script passed to exec_script_streaming
        script = mock_ssh.exec_script_streaming.call_args[0][0]
        assert '"hello world"' in script

    def test_exec_foreground_single_quotes(self, mock_ssh):
        """exec_foreground preserves single quotes."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        executor.exec_foreground(ctx, "echo 'hello world'")

        script = mock_ssh.exec_script_streaming.call_args[0][0]
        assert "hello world" in script

    def test_exec_foreground_dollar_variable(self, mock_ssh):
        """exec_foreground preserves dollar sign variables."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        executor.exec_foreground(ctx, "echo $HOME $USER")

        script = mock_ssh.exec_script_streaming.call_args[0][0]
        assert "$HOME" in script
        assert "$USER" in script

    def test_exec_foreground_pipe(self, mock_ssh):
        """exec_foreground preserves pipe characters."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        executor.exec_foreground(ctx, "ls -la | grep foo")

        script = mock_ssh.exec_script_streaming.call_args[0][0]
        assert "|" in script

    def test_exec_foreground_semicolon(self, mock_ssh):
        """exec_foreground preserves semicolons."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        executor.exec_foreground(ctx, "cd /tmp; ls; pwd")

        script = mock_ssh.exec_script_streaming.call_args[0][0]
        assert ";" in script

    def test_exec_foreground_ampersand(self, mock_ssh):
        """exec_foreground preserves ampersands."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        executor.exec_foreground(ctx, "cmd1 && cmd2 || cmd3")

        script = mock_ssh.exec_script_streaming.call_args[0][0]
        assert "&&" in script
        assert "||" in script

    def test_exec_foreground_backticks(self, mock_ssh):
        """exec_foreground preserves backticks."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        executor.exec_foreground(ctx, "echo `date`")

        script = mock_ssh.exec_script_streaming.call_args[0][0]
        assert "`" in script

    def test_exec_foreground_parentheses(self, mock_ssh):
        """exec_foreground preserves parentheses."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        executor.exec_foreground(ctx, "(cd /tmp && ls)")

        script = mock_ssh.exec_script_streaming.call_args[0][0]
        assert "(" in script
        assert ")" in script

    def test_exec_foreground_glob(self, mock_ssh):
        """exec_foreground preserves glob patterns."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        executor.exec_foreground(ctx, "ls *.py")

        script = mock_ssh.exec_script_streaming.call_args[0][0]
        assert "*" in script


class TestDirectExecutorExecDetached:
    """Tests for DirectExecutor.exec_detached with special characters."""

    @pytest.fixture
    def mock_ssh(self):
        """Create a mock SSH executor."""
        ssh = MagicMock()
        ssh.target = "user@host"
        ssh.exec.return_value = (0, "12345", "")
        return ssh

    def test_exec_detached_double_quotes(self, mock_ssh):
        """exec_detached preserves double quotes."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        executor.exec_detached(ctx, 'echo "hello world"', job_name="test")

        # Check the command passed to ssh.exec
        cmd = mock_ssh.exec.call_args[0][0]
        assert '"hello world"' in cmd

    def test_exec_detached_single_quotes(self, mock_ssh):
        """exec_detached escapes single quotes properly."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        executor.exec_detached(ctx, "echo 'hello world'", job_name="test")

        cmd = mock_ssh.exec.call_args[0][0]
        # Single quotes inside should be escaped
        assert "hello world" in cmd

    def test_exec_detached_dollar_variable(self, mock_ssh):
        """exec_detached preserves dollar sign variables."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        executor.exec_detached(ctx, "echo $HOME $USER", job_name="test")

        cmd = mock_ssh.exec.call_args[0][0]
        assert "$HOME" in cmd
        assert "$USER" in cmd

    def test_exec_detached_pipe(self, mock_ssh):
        """exec_detached preserves pipe characters."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        executor.exec_detached(ctx, "ls -la | grep foo", job_name="test")

        cmd = mock_ssh.exec.call_args[0][0]
        assert "|" in cmd

    def test_exec_detached_semicolon(self, mock_ssh):
        """exec_detached preserves semicolons."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        executor.exec_detached(ctx, "cd /tmp; ls; pwd", job_name="test")

        cmd = mock_ssh.exec.call_args[0][0]
        assert ";" in cmd

    def test_exec_detached_ampersand(self, mock_ssh):
        """exec_detached preserves ampersands."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        executor.exec_detached(ctx, "cmd1 && cmd2", job_name="test")

        cmd = mock_ssh.exec.call_args[0][0]
        assert "&&" in cmd

    def test_exec_detached_complex_command(self, mock_ssh):
        """exec_detached handles complex real-world commands."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext(target="user@host")

        cmd = '''for f in *.py; do echo "$f"; done'''
        executor.exec_detached(ctx, cmd, job_name="test")

        result_cmd = mock_ssh.exec.call_args[0][0]
        assert "for" in result_cmd
        assert "done" in result_cmd
