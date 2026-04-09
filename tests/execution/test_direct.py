"""Tests for direct (non-SLURM) execution."""

import json

import pytest
from unittest.mock import MagicMock, patch, call

from rex.execution.direct import DirectExecutor
from rex.execution.base import ExecutionContext, JobInfo


class TestDirectExecutorExecForeground:
    """Tests for DirectExecutor.exec_foreground."""

    @pytest.fixture
    def mock_ssh(self):
        """Create a mock SSH executor."""
        ssh = MagicMock()
        ssh.target = "user@host"
        ssh.exec.return_value = (0, "", "")
        ssh.exec_streaming.return_value = 0
        return ssh

    def test_exec_foreground_streams_via_ssh(self, mock_ssh):
        """exec_foreground runs the script directly over SSH with tee."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        result = executor.exec_foreground(ctx, "echo hello")

        mock_ssh.exec_streaming.assert_called_once()
        streaming_cmd = mock_ssh.exec_streaming.call_args[0][0]
        assert "tee" in streaming_cmd
        assert "PIPESTATUS" in streaming_cmd
        assert result == 0

    def test_exec_foreground_returns_exit_code(self, mock_ssh):
        """exec_foreground returns the exit code from exec_streaming."""
        mock_ssh.exec_streaming.return_value = 42
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        result = executor.exec_foreground(ctx, "exit 42")

        assert result == 42

    def test_exec_foreground_writes_metadata(self, mock_ssh):
        """exec_foreground writes job metadata after completion."""
        executor = DirectExecutor(mock_ssh)
        ctx = ExecutionContext()

        with patch("rex.execution.direct.write_job_meta") as mock_meta:
            executor.exec_foreground(ctx, "echo hello")

        mock_meta.assert_called_once()
        args = mock_meta.call_args[0]
        assert args[0] is mock_ssh


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


class TestDirectWriteScript:
    """Tests for DirectExecutor._write_script."""

    @pytest.fixture
    def mock_ssh(self):
        ssh = MagicMock()
        ssh.target = "user@host"
        ssh.exec.return_value = (0, "", "")
        return ssh

    def test_creates_executable(self, mock_ssh):
        """_write_script writes via heredoc and sets +x."""
        executor = DirectExecutor(mock_ssh)
        executor._write_script("/tmp/test.sh", "#!/bin/bash\necho hi\n")

        write_call = mock_ssh.exec.call_args[0][0]
        assert "REXSCRIPT" in write_call
        assert "chmod +x" in write_call
        assert "#!/bin/bash" in write_call

    def test_raises_on_failure(self, mock_ssh):
        """_write_script raises ExecutionError when ssh.exec fails."""
        from rex.exceptions import ExecutionError
        mock_ssh.exec.return_value = (1, "", "permission denied")
        executor = DirectExecutor(mock_ssh)

        with pytest.raises(ExecutionError, match="Failed to write script"):
            executor._write_script("/tmp/test.sh", "#!/bin/bash\necho hi\n")


class TestDirectListJobs:
    """Tests for DirectExecutor.list_jobs."""

    @pytest.fixture
    def mock_ssh(self):
        ssh = MagicMock()
        ssh.target = "user@host"
        ssh.exec.return_value = (0, "", "")
        return ssh

    def test_reads_metadata_and_checks_pid(self, mock_ssh):
        """list_jobs reads metadata files and checks PID status."""
        with patch("rex.execution.direct.list_job_meta_names", return_value=["job-1", "job-2"]), \
             patch("rex.execution.direct.read_job_meta") as mock_read:
            mock_read.side_effect = [
                {"pid": 100, "log": "/tmp/1.log"},
                {"pid": 200, "log": "/tmp/2.log"},
            ]
            # job-1 running, job-2 completed
            mock_ssh.exec.side_effect = [
                (0, "", ""),   # kill -0 100: alive
                (1, "", ""),   # kill -0 200: dead
            ]
            executor = DirectExecutor(mock_ssh)
            jobs = executor.list_jobs()

        assert len(jobs) == 2
        assert jobs[0].status == "running"
        assert jobs[0].pid == 100
        assert jobs[1].status == "completed"
        assert jobs[1].pid is None

    def test_no_pid_means_completed(self, mock_ssh):
        """Jobs without a PID are treated as completed."""
        with patch("rex.execution.direct.list_job_meta_names", return_value=["job-1"]), \
             patch("rex.execution.direct.read_job_meta", return_value={"log": "/tmp/1.log"}):
            executor = DirectExecutor(mock_ssh)
            jobs = executor.list_jobs()

        assert len(jobs) == 1
        assert jobs[0].status == "completed"


class TestDirectGetStatus:
    """Tests for DirectExecutor.get_status."""

    @pytest.fixture
    def mock_ssh(self):
        ssh = MagicMock()
        ssh.target = "user@host"
        return ssh

    def test_running(self, mock_ssh):
        """get_status returns running when PID is alive."""
        mock_ssh.exec.return_value = (0, "", "")
        with patch("rex.execution.direct.read_job_meta", return_value={"pid": 42}):
            executor = DirectExecutor(mock_ssh)
            status = executor.get_status("job-1")

        assert status.status == "running"
        assert status.pid == 42

    def test_completed(self, mock_ssh):
        """get_status returns completed when PID is dead."""
        mock_ssh.exec.return_value = (1, "", "")
        with patch("rex.execution.direct.read_job_meta", return_value={"pid": 42}):
            executor = DirectExecutor(mock_ssh)
            status = executor.get_status("job-1")

        assert status.status == "completed"
        assert status.pid is None

    def test_no_metadata(self, mock_ssh):
        """get_status returns unknown when metadata is missing."""
        with patch("rex.execution.direct.read_job_meta", return_value=None):
            executor = DirectExecutor(mock_ssh)
            status = executor.get_status("job-1")

        assert status.status == "unknown"


class TestDirectKillJob:
    """Tests for DirectExecutor.kill_job."""

    @pytest.fixture
    def mock_ssh(self):
        ssh = MagicMock()
        ssh.target = "user@host"
        return ssh

    def test_sends_kill(self, mock_ssh):
        """kill_job sends kill signal to PID."""
        mock_ssh.exec.return_value = (0, "", "")
        with patch("rex.execution.direct.read_job_meta", return_value={"pid": 42}):
            executor = DirectExecutor(mock_ssh)
            result = executor.kill_job("job-1")

        assert result is True
        mock_ssh.exec.assert_called_once_with("kill 42 2>/dev/null")

    def test_missing_job(self, mock_ssh):
        """kill_job returns False when job not found."""
        with patch("rex.execution.direct.read_job_meta", return_value=None):
            executor = DirectExecutor(mock_ssh)
            result = executor.kill_job("nonexistent")

        assert result is False

    def test_kill_failure(self, mock_ssh):
        """kill_job returns False when kill fails."""
        mock_ssh.exec.return_value = (1, "", "")
        with patch("rex.execution.direct.read_job_meta", return_value={"pid": 42}):
            executor = DirectExecutor(mock_ssh)
            result = executor.kill_job("job-1")

        assert result is False


class TestDirectWatchJob:
    """Tests for DirectExecutor.watch_job."""

    @pytest.fixture
    def mock_ssh(self):
        ssh = MagicMock()
        ssh.target = "user@host"
        return ssh

    def test_polls_until_complete(self, mock_ssh):
        """watch_job returns when job completes."""
        with patch.object(DirectExecutor, "get_status") as mock_status:
            from rex.execution.base import JobStatus
            mock_status.side_effect = [
                JobStatus(job_id="job-1", status="running", pid=42),
                JobStatus(job_id="job-1", status="completed"),
            ]
            executor = DirectExecutor(mock_ssh)
            result = executor.watch_job("job-1", poll_interval=0)

        assert result.status == "completed"
        assert result.exit_code == 0

    def test_connection_failures(self, mock_ssh):
        """watch_job gives up after 3 consecutive failures."""
        with patch.object(DirectExecutor, "get_status") as mock_status:
            from rex.execution.base import JobStatus
            mock_status.return_value = JobStatus(job_id="job-1", status="unknown")
            executor = DirectExecutor(mock_ssh)
            result = executor.watch_job("job-1", poll_interval=0)

        assert result.status == "unknown"
        assert result.exit_code == 1
