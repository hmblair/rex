"""Tests for SLURM execution."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from rex.execution.base import ExecutionContext, JobInfo
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


def _find_exec_call(mock_ssh, keyword: str) -> str:
    """Find the first ssh.exec call containing the given keyword."""
    for args, _ in mock_ssh.exec.call_args_list:
        if keyword in args[0]:
            return args[0]
    raise AssertionError(f"No ssh.exec call containing '{keyword}'")


class TestSlurmExecForeground:
    """Tests for SlurmExecutor.exec_foreground."""

    def test_uses_srun(self, mock_ssh):
        """exec_foreground runs via srun."""
        executor = SlurmExecutor(mock_ssh, SlurmOptions(partition="gpu"))
        ctx = ExecutionContext()
        executor.exec_foreground(ctx, "echo hello")

        streaming_cmd = mock_ssh.exec_streaming.call_args[0][0]
        assert "srun" in streaming_cmd

    def test_includes_slurm_opts(self, mock_ssh):
        """exec_foreground passes SLURM options to srun."""
        opts = SlurmOptions(partition="gpu", gres="gpu:1", time="01:00:00")
        executor = SlurmExecutor(mock_ssh, opts)
        ctx = ExecutionContext()
        executor.exec_foreground(ctx, "echo hello")

        streaming_cmd = mock_ssh.exec_streaming.call_args[0][0]
        assert "--partition=gpu" in streaming_cmd
        assert "--gres=gpu:1" in streaming_cmd
        assert "--time=01:00:00" in streaming_cmd

    def test_returns_exit_code(self, mock_ssh):
        """exec_foreground returns the exit code from exec_streaming."""
        mock_ssh.exec_streaming.return_value = 7
        executor = SlurmExecutor(mock_ssh)
        ctx = ExecutionContext()
        result = executor.exec_foreground(ctx, "exit 7")
        assert result == 7

    def test_heredoc_collision_returns_error(self, mock_ssh):
        """exec_foreground returns 1 when command contains REXCMD delimiter."""
        executor = SlurmExecutor(mock_ssh)
        ctx = ExecutionContext()
        with patch("rex.execution.slurm.error") as mock_error:
            result = executor.exec_foreground(ctx, "echo before\nREXCMD\necho after")
        assert result == 1
        mock_ssh.exec_streaming.assert_not_called()
        mock_error.assert_called_once()

    def test_writes_separate_cmd_file(self, mock_ssh):
        """exec_foreground writes command to a .cmd file via heredoc."""
        executor = SlurmExecutor(mock_ssh)
        ctx = ExecutionContext()
        executor.exec_foreground(ctx, "python train.py --lr 0.01")

        # Find the heredoc write call for the command file
        cmd_write = None
        for args, _ in mock_ssh.exec.call_args_list:
            if "REXCMD" in args[0]:
                cmd_write = args[0]
                break
        assert cmd_write is not None
        assert "python train.py --lr 0.01" in cmd_write

    def test_cleans_up_scripts(self, mock_ssh):
        """exec_foreground cleans up script files after execution."""
        executor = SlurmExecutor(mock_ssh)
        ctx = ExecutionContext()
        executor.exec_foreground(ctx, "echo hello")

        streaming_cmd = mock_ssh.exec_streaming.call_args[0][0]
        assert "rm -f" in streaming_cmd


class TestSlurmExecDetached:
    """Tests for SlurmExecutor.exec_detached."""

    def test_uses_sbatch(self, mock_ssh):
        """exec_detached submits via sbatch --parsable."""
        mock_ssh.exec.return_value = (0, "12345", "")
        executor = SlurmExecutor(mock_ssh)
        ctx = ExecutionContext()
        executor.exec_detached(ctx, "echo hello", "test-job")

        sbatch_call = None
        for args, _ in mock_ssh.exec.call_args_list:
            if args[0].startswith("sbatch"):
                sbatch_call = args[0]
                break
        assert sbatch_call is not None
        assert "--parsable" in sbatch_call

    def test_returns_job_info_with_slurm_id(self, mock_ssh):
        """exec_detached parses SLURM ID from sbatch output."""
        mock_ssh.exec.return_value = (0, "67890", "")
        executor = SlurmExecutor(mock_ssh)
        ctx = ExecutionContext()
        result = executor.exec_detached(ctx, "echo hello", "test-job")

        assert isinstance(result, JobInfo)
        assert result.slurm_id == 67890
        assert result.is_slurm is True
        assert result.job_id == "test-job"

    def test_sets_job_name_in_sbatch(self, mock_ssh):
        """exec_detached sets job name as rex-{name} in sbatch script."""
        mock_ssh.exec.return_value = (0, "12345", "")
        executor = SlurmExecutor(mock_ssh)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "echo hello", "my-job")
        script = _find_exec_call(mock_ssh, "REXWRITE")
        assert "#SBATCH --job-name=rex-my-job" in script

    def test_includes_slurm_directives(self, mock_ssh):
        """exec_detached includes SLURM directives in sbatch script."""
        mock_ssh.exec.return_value = (0, "12345", "")
        opts = SlurmOptions(partition="gpu", gres="gpu:2", mem="32G")
        executor = SlurmExecutor(mock_ssh, opts)
        ctx = ExecutionContext()

        executor.exec_detached(ctx, "echo hello", "test-job")
        script = _find_exec_call(mock_ssh, "REXWRITE")
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --gres=gpu:2" in script
        assert "#SBATCH --mem=32G" in script

    def test_applies_context(self, mock_ssh):
        """exec_detached applies execution context to sbatch script."""
        mock_ssh.exec.return_value = (0, "12345", "")
        executor = SlurmExecutor(mock_ssh)
        ctx = ExecutionContext(
            modules=["python/3.11"],
            env={"WANDB_PROJECT": "myexp"},
        )

        executor.exec_detached(ctx, "python train.py", "test-job")
        script = _find_exec_call(mock_ssh, "REXWRITE")
        assert "module load python/3.11" in script
        assert "WANDB_PROJECT" in script

    def test_handles_sbatch_failure(self, mock_ssh):
        """exec_detached returns JobInfo with no slurm_id on failure."""
        def side_effect(cmd):
            if cmd.startswith("sbatch"):
                return (1, "", "sbatch: error: invalid partition")
            return (0, "", "")

        mock_ssh.exec.side_effect = side_effect
        executor = SlurmExecutor(mock_ssh)
        ctx = ExecutionContext()
        result = executor.exec_detached(ctx, "echo hello", "test-job")

        assert result.slurm_id is None
        assert result.job_id == "test-job"


class TestSlurmListJobs:
    """Tests for SlurmExecutor.list_jobs."""

    def test_parses_squeue_output(self, mock_ssh):
        """list_jobs correctly parses squeue output."""
        squeue_output = (
            "     12345 rex-train-001          RUNNING       5:00\n"
            "     12346 rex-eval-002           PENDING       0:00\n"
        )
        mock_ssh.exec.return_value = (0, squeue_output, "")
        executor = SlurmExecutor(mock_ssh)
        jobs = executor.list_jobs()

        assert len(jobs) == 2
        assert jobs[0].job_id == "train-001"
        assert jobs[0].status == "running"
        assert jobs[0].slurm_id == 12345
        assert jobs[1].job_id == "eval-002"
        assert jobs[1].status == "pending"

    def test_with_since_queries_sacct(self, mock_ssh):
        """list_jobs queries sacct when since_minutes > 0."""
        # First call: squeue (no active jobs)
        # Second call: sacct
        mock_ssh.exec.side_effect = [
            (0, "", ""),
            (0, "12345|rex-old-job|COMPLETED\n", ""),
        ]
        executor = SlurmExecutor(mock_ssh)
        jobs = executor.list_jobs(since_minutes=30)

        assert len(jobs) == 1
        assert jobs[0].job_id == "old-job"
        assert jobs[0].status == "completed"

    def test_deduplicates_by_slurm_id(self, mock_ssh):
        """list_jobs deduplicates jobs appearing in both squeue and sacct."""
        mock_ssh.exec.side_effect = [
            (0, "     12345 rex-job-001          RUNNING       5:00\n", ""),
            (0, "12345|rex-job-001|RUNNING\n", ""),
        ]
        executor = SlurmExecutor(mock_ssh)
        jobs = executor.list_jobs(since_minutes=30)

        assert len(jobs) == 1


class TestSlurmGetStatus:
    """Tests for SlurmExecutor.get_status."""

    def test_checks_squeue_first(self, mock_ssh):
        """get_status checks squeue before sacct."""
        mock_ssh.exec.return_value = (0, "RUNNING", "")
        executor = SlurmExecutor(mock_ssh)
        status = executor.get_status("test-job")

        assert status.status == "running"
        # Should only call squeue, not sacct
        assert mock_ssh.exec.call_count == 1

    def test_falls_back_to_sacct(self, mock_ssh):
        """get_status checks sacct when job not in squeue."""
        mock_ssh.exec.side_effect = [
            (0, "", ""),          # squeue: empty
            (0, "COMPLETED", ""), # sacct: found
        ]
        executor = SlurmExecutor(mock_ssh)
        status = executor.get_status("test-job")

        assert status.status == "completed"
        assert mock_ssh.exec.call_count == 2

    def test_maps_slurm_states(self, mock_ssh):
        """get_status maps SLURM states to lowercase."""
        for slurm_state, expected in [
            ("RUNNING", "running"),
            ("PENDING", "pending"),
            ("FAILED", "failed"),
            ("CANCELLED", "cancelled"),
            ("TIMEOUT", "timeout"),
        ]:
            mock_ssh.exec.reset_mock()
            mock_ssh.exec.return_value = (0, slurm_state, "")
            executor = SlurmExecutor(mock_ssh)
            status = executor.get_status("test-job")
            assert status.status == expected, f"Failed for {slurm_state}"

    def test_returns_unknown_on_ssh_failure(self, mock_ssh):
        """get_status returns unknown when SSH fails."""
        from rex.exceptions import SSHError
        mock_ssh.exec.return_value = (1, "", "connection refused")
        executor = SlurmExecutor(mock_ssh)
        status = executor.get_status("test-job")

        assert status.status == "unknown"


class TestSlurmKillJob:
    """Tests for SlurmExecutor.kill_job."""

    def test_uses_scancel(self, mock_ssh):
        """kill_job cancels via scancel -n."""
        mock_ssh.exec.return_value = (0, "", "")
        executor = SlurmExecutor(mock_ssh)
        result = executor.kill_job("test-job")

        assert result is True
        mock_ssh.exec.assert_called_once_with("scancel -n rex-test-job")

    def test_returns_false_on_failure(self, mock_ssh):
        """kill_job returns False when scancel fails."""
        mock_ssh.exec.return_value = (1, "", "error")
        executor = SlurmExecutor(mock_ssh)
        result = executor.kill_job("test-job")

        assert result is False


class TestSlurmWatchJob:
    """Tests for SlurmExecutor.watch_job."""

    def test_polls_until_complete(self, mock_ssh):
        """watch_job polls until job completes."""
        mock_ssh.exec.side_effect = [
            (0, "RUNNING", ""),
            (0, "", ""), (0, "COMPLETED", ""),  # squeue empty, sacct completed
        ]
        executor = SlurmExecutor(mock_ssh)
        result = executor.watch_job("test-job", poll_interval=0)

        assert result.status == "completed"
        assert result.exit_code == 0

    def test_returns_failure_status(self, mock_ssh):
        """watch_job returns failure status for failed jobs."""
        mock_ssh.exec.side_effect = [
            (0, "", ""), (0, "FAILED", ""),
        ]
        executor = SlurmExecutor(mock_ssh)
        result = executor.watch_job("test-job", poll_interval=0)

        assert result.status == "failed"
        assert result.exit_code == 1

    def test_handles_connection_failures(self, mock_ssh):
        """watch_job retries on connection failures up to 3 times."""
        from rex.exceptions import SSHError
        mock_ssh.exec.side_effect = [
            (1, "", ""),  # fail 1
            (1, "", ""),  # fail 2
            (1, "", ""),  # fail 3
        ]
        executor = SlurmExecutor(mock_ssh)
        result = executor.watch_job("test-job", poll_interval=0)

        assert result.status == "unknown"
        assert result.exit_code == 1
