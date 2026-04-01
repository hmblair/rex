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

    def _find_exec_call(self, mock_ssh, keyword: str) -> str:
        """Find the first ssh.exec call containing the given keyword."""
        for args, _ in mock_ssh.exec.call_args_list:
            if keyword in args[0]:
                return args[0]
        raise AssertionError(f"No ssh.exec call containing '{keyword}'")

    def _get_script_content(self, mock_ssh) -> str:
        """Extract the script content from the heredoc write call."""
        return self._find_exec_call(mock_ssh, "REXSCRIPT")

    def test_exec_detached_double_quotes(self, mock_ssh):
        """exec_detached preserves double quotes."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, 'echo "hello world"', job_name="test")

        # Check the script content written via heredoc
        script = self._get_script_content(mock_ssh)
        assert '"hello world"' in script

    def test_exec_detached_single_quotes(self, mock_ssh):
        """exec_detached preserves single quotes."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "echo 'hello world'", job_name="test")

        script = self._get_script_content(mock_ssh)
        assert "'hello world'" in script

    def test_exec_detached_dollar_variable(self, mock_ssh):
        """exec_detached preserves dollar sign variables."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "echo $HOME $USER", job_name="test")

        script = self._get_script_content(mock_ssh)
        assert "$HOME" in script
        assert "$USER" in script

    def test_exec_detached_pipe(self, mock_ssh):
        """exec_detached preserves pipe characters."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "ls -la | grep foo", job_name="test")

        script = self._get_script_content(mock_ssh)
        assert "|" in script

    def test_exec_detached_semicolon(self, mock_ssh):
        """exec_detached preserves semicolons."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "cd /tmp; ls; pwd", job_name="test")

        script = self._get_script_content(mock_ssh)
        assert ";" in script

    def test_exec_detached_ampersand(self, mock_ssh):
        """exec_detached preserves ampersands."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "cmd1 && cmd2", job_name="test")

        script = self._get_script_content(mock_ssh)
        assert "&&" in script

    def test_exec_detached_complex_command(self, mock_ssh):
        """exec_detached handles complex real-world commands."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        cmd = '''for f in *.py; do echo "$f"; done'''
        executor.exec_detached(ctx, cmd, job_name="test")

        script = self._get_script_content(mock_ssh)
        assert "for" in script
        assert "done" in script

    def test_exec_detached_writes_script_file(self, mock_ssh):
        """exec_detached writes command to .sh file in rex_dir."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "echo hello", job_name="test-job")

        # Find the heredoc write call
        script_content = self._get_script_content(mock_ssh)
        assert "~/.rex/rex-test-job.sh" in script_content
        assert "#!/bin/bash -l" in script_content

        # Find the nohup call
        nohup_cmd = self._find_exec_call(mock_ssh, "nohup")
        assert "~/.rex/rex-test-job.sh" in nohup_cmd
