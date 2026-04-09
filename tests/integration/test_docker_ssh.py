"""Integration tests using Docker SSH server.

These tests require Docker to be available and will be skipped otherwise.
"""

from __future__ import annotations

import subprocess
import time

import pytest

from rex.execution.base import ExecutionContext
from rex.execution.direct import DirectExecutor
from tests.integration.conftest import _DOCKER_AVAILABLE

pytestmark = pytest.mark.skipif(
    not _DOCKER_AVAILABLE,
    reason="Docker not available"
)


class TestSSHBasics:
    """Basic SSH connectivity tests."""

    def test_exec_simple_command(self, ssh_server):
        """Execute a simple command via SSH."""
        code, stdout, stderr = ssh_server.exec("echo hello")

        assert code == 0
        assert "hello" in stdout

    def test_exec_with_exit_code(self, ssh_server):
        """Command exit codes are returned correctly."""
        code, stdout, stderr = ssh_server.exec("exit 42")

        assert code == 42

    def test_exec_stderr(self, ssh_server):
        """Stderr is captured separately."""
        code, stdout, stderr = ssh_server.exec("echo error >&2")

        assert code == 0
        assert "error" in stderr

    def test_exec_multiline_output(self, ssh_server):
        """Multiline output is captured."""
        code, stdout, stderr = ssh_server.exec("echo line1; echo line2; echo line3")

        assert code == 0
        assert "line1" in stdout
        assert "line2" in stdout
        assert "line3" in stdout


class TestDirectExecutor:
    """Tests for DirectExecutor using real SSH."""

    def test_exec_foreground(self, ssh_server):
        """exec_foreground runs command and returns exit code."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        code = executor.exec_foreground(ctx, "echo test && exit 0")

        assert code == 0

    def test_exec_foreground_with_env(self, ssh_server):
        """exec_foreground applies environment variables."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext(env={"MY_VAR": "my_value"})

        code = executor.exec_foreground(ctx, "echo $MY_VAR")

        assert code == 0

    def test_exec_detached_creates_job(self, ssh_server):
        """exec_detached creates a background job and returns info."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        job_info = executor.exec_detached(ctx, "sleep 1", job_name="test-detach")

        assert job_info.job_id == "test-detach"
        assert job_info.pid is not None
        assert job_info.is_slurm is False
        # Log file should be created
        assert "rex-test-detach.log" in job_info.log_path


class TestShowLogFollow:
    """Tests for show_log with follow mode."""

    def test_follow_exits_when_job_completes(self, ssh_server):
        """Follow mode exits when the monitored job completes."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        # Start a job that writes output and exits
        executor.exec_detached(
            ctx,
            "for i in 1 2 3; do echo line$i; sleep 0.3; done; echo done",
            job_name="test-follow",
        )

        # Give it a moment to start
        time.sleep(0.2)

        # Run show_log with follow - should exit when job completes
        start = time.time()
        code = executor.show_log("test-follow", follow=True)
        elapsed = time.time() - start

        # Should have taken ~1 second (job duration), not hung forever
        assert elapsed < 5
        assert code == 0

    def test_follow_falls_back_to_cat_for_finished_job(self, ssh_server):
        """Follow mode uses cat when job is already finished."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        # Start a quick job that finishes immediately
        executor.exec_detached(ctx, "echo completed", job_name="test-follow-done")

        # Wait for job to finish
        time.sleep(1)

        # Follow should immediately return (cat instead of tail -f)
        start = time.time()
        code = executor.show_log("test-follow-done", follow=True)
        elapsed = time.time() - start

        assert elapsed < 2
        assert code == 0

    def test_no_follow_returns_immediately(self, ssh_server):
        """Non-follow mode returns immediately with current log content."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        # Start a long-running job
        executor.exec_detached(ctx, "echo started; sleep 60", job_name="test-nofollow")

        try:
            # Give it a moment to write output
            time.sleep(0.5)

            # Non-follow should return immediately
            start = time.time()
            code = executor.show_log("test-nofollow", follow=False)
            elapsed = time.time() - start

            assert elapsed < 2
            assert code == 0
        finally:
            executor.kill_job("test-nofollow")


class TestSpecialCharacters:
    """Test that special characters are handled correctly through SSH."""

    def test_quotes_preserved(self, ssh_server):
        """Quotes in commands work correctly."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        code = executor.exec_foreground(ctx, '''echo "hello 'world'"''')

        assert code == 0

    def test_variables_expanded(self, ssh_server):
        """Shell variables are expanded on remote."""
        code, stdout, _ = ssh_server.exec("echo $HOME")

        assert code == 0
        assert "/home/test" in stdout

    def test_pipes_work(self, ssh_server):
        """Pipe commands work correctly."""
        code, stdout, _ = ssh_server.exec("echo hello | tr 'h' 'H'")

        assert code == 0
        assert "Hello" in stdout

    def test_multiline_command(self, ssh_server):
        """Multi-statement commands work."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        code = executor.exec_foreground(ctx, "x=1; y=2; echo $((x+y))")

        assert code == 0


class TestForegroundSemantics:
    """Tests for foreground execution behavior."""

    def test_fast_command_does_not_hang(self, ssh_server):
        """A fast foreground command returns promptly, not hanging."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        start = time.time()
        code = executor.exec_foreground(ctx, "echo done")
        elapsed = time.time() - start

        assert code == 0
        assert elapsed < 5

    def test_foreground_returns_nonzero_exit_code(self, ssh_server):
        """Foreground propagates nonzero exit codes."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        code = executor.exec_foreground(ctx, "exit 3")

        assert code == 3

    def test_foreground_writes_log(self, ssh_server):
        """After foreground, the log file exists and show_log works."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        executor.exec_foreground(ctx, "echo logged-output")

        job_id = executor.last_job_id()
        assert job_id is not None
        code = executor.show_log(job_id, follow=False)
        assert code == 0


class TestJobLifecycle:
    """Tests for the full detached job lifecycle."""

    def test_detached_lifecycle(self, ssh_server):
        """exec_detached -> get_status -> wait -> completed."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        job_info = executor.exec_detached(ctx, "sleep 1; echo done", job_name="lifecycle-test")

        # Should be running initially
        status = executor.get_status(job_info.job_id)
        assert status.status in ("running", "completed")

        # Wait for completion
        time.sleep(2)

        status = executor.get_status(job_info.job_id)
        assert status.status == "completed"

    def test_detached_job_appears_in_list(self, ssh_server):
        """A detached job appears in list_jobs."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "sleep 2", job_name="list-test")

        try:
            jobs = executor.list_jobs()
            job_ids = [j.job_id for j in jobs]
            assert "list-test" in job_ids
        finally:
            executor.kill_job("list-test")

    def test_detached_job_kill(self, ssh_server):
        """A long-running detached job can be killed."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "sleep 60", job_name="kill-test")

        result = executor.kill_job("kill-test")
        assert result is True

        time.sleep(0.5)
        status = executor.get_status("kill-test")
        assert status.status == "completed"

    def test_detached_job_log(self, ssh_server):
        """A detached job's log can be retrieved."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "echo log-content; sleep 0.5", job_name="log-test")
        time.sleep(1)

        code = executor.show_log("log-test", follow=False)
        assert code == 0

    def test_last_job_id(self, ssh_server):
        """last_job_id returns the most recent job."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "echo first", job_name="first-job")
        time.sleep(0.1)
        executor.exec_detached(ctx, "echo second", job_name="second-job")

        last = executor.last_job_id()
        assert last == "second-job"


class TestContextApplication:
    """Tests that execution context is applied correctly over real SSH."""

    def test_env_vars_in_foreground(self, ssh_server):
        """Environment variables are visible to foreground commands."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext(env={"REX_TEST_VAR": "works"})

        code = executor.exec_foreground(ctx, "test \"$REX_TEST_VAR\" = works")
        assert code == 0

    def test_run_dir_in_foreground(self, ssh_server):
        """Foreground command runs in the specified run_dir."""
        executor = DirectExecutor(ssh_server)
        # Use /tmp since it always exists
        ctx = ExecutionContext(run_dir="/tmp")

        code, stdout, _ = ssh_server.exec("echo placeholder")
        # Run via foreground and check working directory
        code = executor.exec_foreground(ctx, "test \"$(pwd)\" = /tmp")
        assert code == 0
