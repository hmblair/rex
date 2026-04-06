"""Tests for direct (non-SLURM) execution."""

import json
import signal

import pytest
from unittest.mock import MagicMock, patch, call

from rex.execution.direct import DirectExecutor
from rex.execution.base import ExecutionContext, JobInfo


class TestDirectExecutorExecForeground:
    """Tests for DirectExecutor.exec_foreground delegation."""

    @pytest.fixture
    def mock_ssh(self):
        """Create a mock SSH executor."""
        ssh = MagicMock()
        ssh.target = "user@host"
        ssh.exec.return_value = (0, "12345", "")
        ssh.exec_streaming.return_value = 0
        return ssh

    def test_exec_foreground_delegates_to_detached(self, mock_ssh):
        """exec_foreground submits a detached job then streams the log."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        with patch.object(executor, "exec_detached") as mock_detach, \
             patch.object(executor, "show_log", return_value=0) as mock_show:
            mock_detach.return_value = JobInfo(
                job_id="test-123", log_path="/tmp/test.log",
                is_slurm=False, pid=42
            )
            result = executor.exec_foreground(ctx, "echo hello")

        mock_detach.assert_called_once()
        assert mock_detach.call_args[0][0] is ctx
        assert mock_detach.call_args[0][1] == "echo hello"
        mock_show.assert_called_once_with("test-123", follow=True)
        assert result == 0

    def test_exec_foreground_returns_show_log_exit_code(self, mock_ssh):
        """exec_foreground returns the exit code from show_log."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        with patch.object(executor, "exec_detached") as mock_detach, \
             patch.object(executor, "show_log", return_value=42) as mock_show:
            mock_detach.return_value = JobInfo(
                job_id="test-123", log_path="/tmp/test.log",
                is_slurm=False, pid=42
            )
            result = executor.exec_foreground(ctx, "exit 42")

        assert result == 42

    def test_exec_foreground_installs_sigint_handler(self, mock_ssh):
        """exec_foreground installs a SIGINT handler that kills the job."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        captured_handler = None

        def capture_signal(sig, handler):
            nonlocal captured_handler
            if sig == signal.SIGINT and callable(handler):
                captured_handler = handler
            return signal.SIG_DFL

        with patch.object(executor, "exec_detached") as mock_detach, \
             patch.object(executor, "show_log", return_value=0), \
             patch("signal.signal", side_effect=capture_signal):
            mock_detach.return_value = JobInfo(
                job_id="test-123", log_path="/tmp/test.log",
                is_slurm=False, pid=42
            )
            executor.exec_foreground(ctx, "echo hello")

        assert captured_handler is not None


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
