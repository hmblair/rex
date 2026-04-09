"""Contract tests: shared behavioral guarantees for all Executor implementations.

Parameterized over DirectExecutor and SlurmExecutor to catch cross-backend
inconsistencies — the class of bug that has driven most churn in this codebase.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call

from rex.execution.base import ExecutionContext, JobInfo
from rex.execution.direct import DirectExecutor
from rex.execution.slurm import SlurmExecutor, SlurmOptions


@pytest.fixture
def mock_ssh():
    """Create a mock SSH executor."""
    ssh = MagicMock()
    ssh.target = "user@host"
    ssh._opts = ["-o", "StrictHostKeyChecking=no"]
    ssh.exec.return_value = (0, "", "")
    ssh.exec_streaming.return_value = 0
    return ssh


@pytest.fixture(params=["direct", "slurm"])
def executor(request, mock_ssh):
    """Parameterized fixture returning both executor types."""
    if request.param == "direct":
        return DirectExecutor(mock_ssh)
    return SlurmExecutor(mock_ssh, SlurmOptions(partition="gpu"))


class TestExecForegroundContract:
    """Both backends must satisfy these foreground guarantees."""

    def test_returns_int(self, executor, mock_ssh):
        """exec_foreground returns an integer exit code."""
        ctx = ExecutionContext()
        result = executor.exec_foreground(ctx, "echo hello")
        assert isinstance(result, int)

    def test_returns_exit_code_from_streaming(self, executor, mock_ssh):
        """exec_foreground returns the exit code from exec_streaming."""
        mock_ssh.exec_streaming.return_value = 42
        ctx = ExecutionContext()
        result = executor.exec_foreground(ctx, "exit 42")
        assert result == 42

    def test_calls_exec_streaming(self, executor, mock_ssh):
        """exec_foreground streams output via SSH, not capture."""
        ctx = ExecutionContext()
        executor.exec_foreground(ctx, "echo hello")
        mock_ssh.exec_streaming.assert_called_once()

    def test_tees_to_log(self, executor, mock_ssh):
        """exec_foreground tees output to a log file."""
        ctx = ExecutionContext()
        executor.exec_foreground(ctx, "echo hello")
        streaming_cmd = mock_ssh.exec_streaming.call_args[0][0]
        assert "tee" in streaming_cmd

    def test_preserves_exit_code_through_pipe(self, executor, mock_ssh):
        """exec_foreground uses PIPESTATUS to preserve the real exit code."""
        ctx = ExecutionContext()
        executor.exec_foreground(ctx, "echo hello")
        streaming_cmd = mock_ssh.exec_streaming.call_args[0][0]
        assert "PIPESTATUS" in streaming_cmd

    def test_does_not_print_detached(self, executor, mock_ssh, capsys):
        """exec_foreground does not print 'Detached:' messages."""
        ctx = ExecutionContext()
        executor.exec_foreground(ctx, "echo hello")
        captured = capsys.readouterr()
        assert "Detached:" not in captured.out
        assert "Submitted:" not in captured.out

    def test_writes_metadata(self, executor, mock_ssh):
        """exec_foreground writes job metadata after completion."""
        ctx = ExecutionContext()
        with patch("rex.execution.direct.write_job_meta") as dm, \
             patch("rex.execution.slurm.write_job_meta") as sm:
            executor.exec_foreground(ctx, "echo hello")
            meta_mock = dm if isinstance(executor, DirectExecutor) else sm
            meta_mock.assert_called_once()

    def test_metadata_uses_fixed_location(self, executor, mock_ssh):
        """Job metadata always written via write_job_meta (-> ~/.rex/jobs/)."""
        ctx = ExecutionContext(run_dir="/some/project")
        with patch("rex.execution.direct.write_job_meta") as dm, \
             patch("rex.execution.slurm.write_job_meta") as sm:
            executor.exec_foreground(ctx, "echo hello")
            meta_mock = dm if isinstance(executor, DirectExecutor) else sm
            # write_job_meta always writes to ~/.rex/jobs/ internally
            meta_mock.assert_called_once()
            args = meta_mock.call_args[0]
            assert args[0] is mock_ssh  # first arg is ssh


class TestExecDetachedContract:
    """Both backends must satisfy these detached guarantees."""

    def test_returns_job_info(self, executor, mock_ssh):
        """exec_detached returns a JobInfo object."""
        if isinstance(executor, SlurmExecutor):
            mock_ssh.exec.return_value = (0, "12345", "")
        else:
            mock_ssh.exec.return_value = (0, "99999", "")
        ctx = ExecutionContext()
        result = executor.exec_detached(ctx, "echo hello", "test-job")
        assert isinstance(result, JobInfo)
        assert result.job_id == "test-job"

    def test_does_not_call_exec_streaming(self, executor, mock_ssh):
        """exec_detached does not stream output."""
        if isinstance(executor, SlurmExecutor):
            mock_ssh.exec.return_value = (0, "12345", "")
        else:
            mock_ssh.exec.return_value = (0, "99999", "")
        ctx = ExecutionContext()
        executor.exec_detached(ctx, "echo hello", "test-job")
        mock_ssh.exec_streaming.assert_not_called()

    def test_writes_metadata(self, executor, mock_ssh):
        """exec_detached writes job metadata."""
        if isinstance(executor, SlurmExecutor):
            mock_ssh.exec.return_value = (0, "12345", "")
        else:
            mock_ssh.exec.return_value = (0, "99999", "")
        ctx = ExecutionContext()
        with patch("rex.execution.direct.write_job_meta") as dm, \
             patch("rex.execution.slurm.write_job_meta") as sm:
            executor.exec_detached(ctx, "echo hello", "test-job")
            meta_mock = dm if isinstance(executor, DirectExecutor) else sm
            meta_mock.assert_called_once()


class TestContextContract:
    """Both backends apply ExecutionContext consistently."""

    def _all_calls(self, mock_ssh):
        """Collect all text sent over SSH."""
        parts = [str(c) for c in mock_ssh.exec.call_args_list]
        parts.append(str(mock_ssh.exec_streaming.call_args))
        return " ".join(parts)

    def test_modules_appear_in_command(self, executor, mock_ssh):
        """Modules from context appear in the script/command."""
        ctx = ExecutionContext(modules=["python/3.11", "cuda/12"])
        executor.exec_foreground(ctx, "echo hello")

        all_calls = self._all_calls(mock_ssh)
        assert "module load" in all_calls
        assert "python/3.11" in all_calls

    def test_env_vars_appear_in_command(self, executor, mock_ssh):
        """Environment variables from context are exported."""
        ctx = ExecutionContext(env={"MY_VAR": "hello"})
        executor.exec_foreground(ctx, "echo $MY_VAR")

        all_calls = self._all_calls(mock_ssh)
        assert "MY_VAR" in all_calls

    def test_run_dir_used_for_log_path(self, executor, mock_ssh):
        """Log path uses run_dir when set."""
        ctx = ExecutionContext(run_dir="/projects/myexp")
        executor.exec_foreground(ctx, "echo hello")

        streaming_cmd = mock_ssh.exec_streaming.call_args[0][0]
        assert "/projects/myexp/.rex/" in streaming_cmd
