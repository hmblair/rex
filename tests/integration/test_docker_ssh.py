"""Integration tests using Docker SSH server.

These tests require Docker to be available and will be skipped otherwise.
"""

from __future__ import annotations

import subprocess
import time

import pytest

from rex.execution.base import ExecutionContext
from rex.execution.direct import DirectExecutor
from rex.commands.jobs import show_log
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
        """exec_detached creates a background job."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        job_info = executor.exec_detached(ctx, "sleep 5", job_name="test-detach")

        assert job_info.job_id == "test-detach"
        assert job_info.pid is not None
        assert job_info.is_slurm is False

        # Cleanup
        executor.kill_job("test-detach")

    def test_list_jobs_shows_running(self, ssh_server):
        """list_jobs returns running jobs."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        # Start a job
        executor.exec_detached(ctx, "sleep 10", job_name="test-list")

        try:
            jobs = executor.list_jobs()

            job_ids = [j.job_id for j in jobs]
            assert "test-list" in job_ids

            job = next(j for j in jobs if j.job_id == "test-list")
            assert job.status == "running"
            assert job.pid is not None
        finally:
            executor.kill_job("test-list")

    def test_get_status_running(self, ssh_server):
        """get_status returns running for active job."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "sleep 10", job_name="test-status")

        try:
            status = executor.get_status("test-status")

            assert status.job_id == "test-status"
            assert status.status == "running"
            assert status.pid is not None
        finally:
            executor.kill_job("test-status")

    def test_get_status_done(self, ssh_server):
        """get_status returns done for completed job."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        # Start a quick job
        executor.exec_detached(ctx, "echo done", job_name="test-done")

        # Wait for it to complete
        time.sleep(1)

        status = executor.get_status("test-done")

        assert status.status == "done"

    def test_kill_job(self, ssh_server):
        """kill_job terminates a running job."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "sleep 60", job_name="test-kill")

        # Verify it's running
        status = executor.get_status("test-kill")
        assert status.status == "running"

        # Kill it
        result = executor.kill_job("test-kill")
        assert result is True

        # Verify it's stopped
        time.sleep(0.5)
        status = executor.get_status("test-kill")
        assert status.status == "done"

    def test_watch_job_waits_for_completion(self, ssh_server):
        """watch_job blocks until job completes."""
        executor = DirectExecutor(ssh_server)
        ctx = ExecutionContext()

        # Start a short job
        executor.exec_detached(ctx, "sleep 1; echo finished", job_name="test-watch")

        start = time.time()
        result = executor.watch_job("test-watch", poll_interval=1)
        elapsed = time.time() - start

        assert result.status == "done"
        assert elapsed >= 1  # Should have waited for job


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
        code = show_log(ssh_server, ssh_server.target, "test-follow", follow=True)
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
        code = show_log(ssh_server, ssh_server.target, "test-follow-done", follow=True)
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
            code = show_log(ssh_server, ssh_server.target, "test-nofollow", follow=False)
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
