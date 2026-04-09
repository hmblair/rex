"""Tests for exec_command dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock

from rex.execution.base import ExecutionContext, JobInfo
from rex.commands.exec import exec_command


class TestExecCommandDispatch:
    """Tests for exec_command foreground/detached dispatch."""

    def setup_method(self):
        self.executor = MagicMock()
        self.ctx = ExecutionContext()

    def test_foreground_calls_exec_foreground(self):
        """detach=False dispatches to executor.exec_foreground."""
        self.executor.exec_foreground.return_value = 0
        result = exec_command(self.executor, self.ctx, "echo hello", detach=False)

        self.executor.exec_foreground.assert_called_once_with(self.ctx, "echo hello")
        self.executor.exec_detached.assert_not_called()
        assert result == 0

    def test_detached_calls_exec_detached(self):
        """detach=True dispatches to executor.exec_detached."""
        job_info = JobInfo(job_id="test", log_path="/tmp/test.log", is_slurm=False)
        self.executor.exec_detached.return_value = job_info
        result = exec_command(self.executor, self.ctx, "echo hello", detach=True)

        self.executor.exec_detached.assert_called_once()
        self.executor.exec_foreground.assert_not_called()
        assert result is job_info

    def test_detached_uses_provided_name(self):
        """Job name is passed through when provided."""
        job_info = JobInfo(job_id="my-job", log_path="/tmp/test.log", is_slurm=False)
        self.executor.exec_detached.return_value = job_info
        exec_command(self.executor, self.ctx, "echo hello", detach=True, job_name="my-job")

        args = self.executor.exec_detached.call_args[0]
        assert args[2] == "my-job"

    def test_detached_generates_name_when_none(self):
        """Job name is auto-generated when not provided."""
        job_info = JobInfo(job_id="auto", log_path="/tmp/test.log", is_slurm=False)
        self.executor.exec_detached.return_value = job_info
        exec_command(self.executor, self.ctx, "echo hello", detach=True, job_name=None)

        args = self.executor.exec_detached.call_args[0]
        assert args[2] is not None  # auto-generated name
        assert len(args[2]) > 0

    def test_foreground_returns_exit_code(self):
        """Foreground returns the integer exit code directly."""
        self.executor.exec_foreground.return_value = 42
        result = exec_command(self.executor, self.ctx, "exit 42", detach=False)
        assert result == 42
