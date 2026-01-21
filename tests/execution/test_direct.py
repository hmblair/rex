"""Tests for direct (non-SLURM) execution."""

import pytest
from unittest.mock import MagicMock, patch

from rex.execution.direct import DirectExecutor
from rex.execution.base import ExecutionContext


class TestDirectExecutorExecForeground:
    """Tests for DirectExecutor.exec_foreground with special characters."""

    @pytest.fixture
    def mock_ssh(self):
        """Create a mock SSH executor."""
        ssh = MagicMock()
        ssh.exec.return_value = (0, "", "")
        ssh.exec_streaming.return_value = 0
        ssh.exec_script_streaming.return_value = 0
        return ssh

    def _get_heredoc_content(self, mock_ssh):
        """Extract command from heredoc passed to ssh.exec."""
        heredoc = mock_ssh.exec.call_args[0][0]
        return heredoc

    def test_exec_foreground_double_quotes(self, mock_ssh):
        """exec_foreground preserves double quotes."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_foreground(ctx, 'echo "hello world"')

        heredoc = self._get_heredoc_content(mock_ssh)
        assert '"hello world"' in heredoc

    def test_exec_foreground_single_quotes(self, mock_ssh):
        """exec_foreground preserves single quotes."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_foreground(ctx, "echo 'hello world'")

        heredoc = self._get_heredoc_content(mock_ssh)
        assert "'hello world'" in heredoc

    def test_exec_foreground_dollar_variable(self, mock_ssh):
        """exec_foreground preserves dollar sign variables."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_foreground(ctx, "echo $HOME $USER")

        heredoc = self._get_heredoc_content(mock_ssh)
        assert "$HOME" in heredoc
        assert "$USER" in heredoc

    def test_exec_foreground_pipe(self, mock_ssh):
        """exec_foreground preserves pipe characters."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_foreground(ctx, "ls -la | grep foo")

        heredoc = self._get_heredoc_content(mock_ssh)
        assert "|" in heredoc

    def test_exec_foreground_semicolon(self, mock_ssh):
        """exec_foreground preserves semicolons."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_foreground(ctx, "cd /tmp; ls; pwd")

        heredoc = self._get_heredoc_content(mock_ssh)
        assert ";" in heredoc

    def test_exec_foreground_ampersand(self, mock_ssh):
        """exec_foreground preserves ampersands."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_foreground(ctx, "cmd1 && cmd2 || cmd3")

        heredoc = self._get_heredoc_content(mock_ssh)
        assert "&&" in heredoc
        assert "||" in heredoc

    def test_exec_foreground_backticks(self, mock_ssh):
        """exec_foreground preserves backticks."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_foreground(ctx, "echo `date`")

        heredoc = self._get_heredoc_content(mock_ssh)
        assert "`" in heredoc

    def test_exec_foreground_parentheses(self, mock_ssh):
        """exec_foreground preserves parentheses."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_foreground(ctx, "(cd /tmp && ls)")

        heredoc = self._get_heredoc_content(mock_ssh)
        assert "(" in heredoc
        assert ")" in heredoc

    def test_exec_foreground_glob(self, mock_ssh):
        """exec_foreground preserves glob patterns."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_foreground(ctx, "ls *.py")

        heredoc = self._get_heredoc_content(mock_ssh)
        assert "*" in heredoc


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
        ctx = ExecutionContext()

        executor.exec_detached(ctx, 'echo "hello world"', job_name="test")

        # Check the command passed to ssh.exec
        cmd = mock_ssh.exec.call_args[0][0]
        assert '"hello world"' in cmd

    def test_exec_detached_single_quotes(self, mock_ssh):
        """exec_detached escapes single quotes properly."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "echo 'hello world'", job_name="test")

        cmd = mock_ssh.exec.call_args[0][0]
        # Single quotes inside should be escaped
        assert "hello world" in cmd

    def test_exec_detached_dollar_variable(self, mock_ssh):
        """exec_detached preserves dollar sign variables."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "echo $HOME $USER", job_name="test")

        cmd = mock_ssh.exec.call_args[0][0]
        assert "$HOME" in cmd
        assert "$USER" in cmd

    def test_exec_detached_pipe(self, mock_ssh):
        """exec_detached preserves pipe characters."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "ls -la | grep foo", job_name="test")

        cmd = mock_ssh.exec.call_args[0][0]
        assert "|" in cmd

    def test_exec_detached_semicolon(self, mock_ssh):
        """exec_detached preserves semicolons."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "cd /tmp; ls; pwd", job_name="test")

        cmd = mock_ssh.exec.call_args[0][0]
        assert ";" in cmd

    def test_exec_detached_ampersand(self, mock_ssh):
        """exec_detached preserves ampersands."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "cmd1 && cmd2", job_name="test")

        cmd = mock_ssh.exec.call_args[0][0]
        assert "&&" in cmd

    def test_exec_detached_complex_command(self, mock_ssh):
        """exec_detached handles complex real-world commands."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        cmd = '''for f in *.py; do echo "$f"; done'''
        executor.exec_detached(ctx, cmd, job_name="test")

        result_cmd = mock_ssh.exec.call_args[0][0]
        assert "for" in result_cmd
        assert "done" in result_cmd
