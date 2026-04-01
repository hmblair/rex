"""Tests for job commands."""

import json

import pytest

from rex.commands.jobs import show_log


def _mock_ssh_with_meta(mocker, meta: dict):
    """Create a mock SSH that returns job meta on first exec, then works for streaming."""
    mock_ssh = mocker.Mock()
    mock_ssh.exec.return_value = (0, json.dumps(meta), "")
    mock_ssh.exec_streaming.return_value = 0
    return mock_ssh


class TestShowLog:
    """Tests for show_log command."""

    def test_follow_with_pid_uses_tail(self, mocker):
        """Follow mode uses tail --pid when meta has a PID."""
        mock_ssh = _mock_ssh_with_meta(mocker, {"log": "/tmp/test.log", "pid": 42})

        show_log(mock_ssh, "host", "abc123", follow=True)

        cmd = mock_ssh.exec_streaming.call_args[0][0]
        assert "tail -f --pid=42" in cmd
        assert "/tmp/test.log" in cmd

    def test_follow_without_pid_uses_cat(self, mocker):
        """Follow mode uses cat when meta has no PID (SLURM job)."""
        mock_ssh = _mock_ssh_with_meta(mocker, {"log": "/tmp/test.log", "slurm_id": 999})

        show_log(mock_ssh, "host", "abc123", follow=True)

        cmd = mock_ssh.exec_streaming.call_args[0][0]
        assert "cat /tmp/test.log" in cmd

    def test_follow_passes_tty_true(self, mocker):
        """Follow mode passes tty=True to exec_streaming."""
        mock_ssh = _mock_ssh_with_meta(mocker, {"log": "/tmp/test.log", "pid": 42})

        show_log(mock_ssh, "host", "abc123", follow=True)

        assert mock_ssh.exec_streaming.call_args[1]["tty"] is True

    def test_no_follow_uses_cat(self, mocker):
        """Non-follow mode uses cat."""
        mock_ssh = _mock_ssh_with_meta(mocker, {"log": "/tmp/test.log", "pid": 42})

        show_log(mock_ssh, "host", "abc123", follow=False)

        cmd = mock_ssh.exec_streaming.call_args[0][0]
        assert "cat /tmp/test.log" in cmd
        assert "tail" not in cmd

    def test_no_follow_passes_tty_false(self, mocker):
        """Non-follow mode passes tty=False to exec_streaming."""
        mock_ssh = _mock_ssh_with_meta(mocker, {"log": "/tmp/test.log"})

        show_log(mock_ssh, "host", "abc123", follow=False)

        assert mock_ssh.exec_streaming.call_args[1]["tty"] is False

    def test_returns_exit_code(self, mocker):
        """Returns exit code from exec_streaming."""
        mock_ssh = _mock_ssh_with_meta(mocker, {"log": "/tmp/test.log"})
        mock_ssh.exec_streaming.return_value = 42

        result = show_log(mock_ssh, "host", "abc123", follow=False)

        assert result == 42

    def test_missing_meta_returns_error(self, mocker):
        """Returns 1 when no job meta exists."""
        mock_ssh = mocker.Mock()
        mock_ssh.exec.return_value = (1, "", "")

        result = show_log(mock_ssh, "host", "abc123", follow=False)

        assert result == 1
