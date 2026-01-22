"""Tests for job commands."""

import pytest

from rex.commands.jobs import show_log


class TestShowLog:
    """Tests for show_log command."""

    def test_follow_uses_tail_with_pid(self, mocker):
        """Follow mode uses tail --pid to exit when job completes."""
        mock_ssh = mocker.Mock()
        mock_ssh.exec_streaming.return_value = 0

        show_log(mock_ssh, "host", "abc123", follow=True)

        cmd = mock_ssh.exec_streaming.call_args[0][0]
        # Should look up PID via pgrep
        assert 'pgrep -f "rex-abc123[.]py"' in cmd
        # Should use tail --pid when job is running
        assert "tail -f --pid=$pid" in cmd
        # Should fall back to cat when job is done
        assert "else cat" in cmd

    def test_follow_passes_tty_true(self, mocker):
        """Follow mode passes tty=True to exec_streaming."""
        mock_ssh = mocker.Mock()
        mock_ssh.exec_streaming.return_value = 0

        show_log(mock_ssh, "host", "abc123", follow=True)

        assert mock_ssh.exec_streaming.call_args[1]["tty"] is True

    def test_no_follow_uses_cat(self, mocker):
        """Non-follow mode uses cat without PID tracking."""
        mock_ssh = mocker.Mock()
        mock_ssh.exec_streaming.return_value = 0

        show_log(mock_ssh, "host", "abc123", follow=False)

        cmd = mock_ssh.exec_streaming.call_args[0][0]
        assert "cat" in cmd
        assert "tail -f" not in cmd
        assert "pgrep" not in cmd

    def test_no_follow_passes_tty_false(self, mocker):
        """Non-follow mode passes tty=False to exec_streaming."""
        mock_ssh = mocker.Mock()
        mock_ssh.exec_streaming.return_value = 0

        show_log(mock_ssh, "host", "abc123", follow=False)

        assert mock_ssh.exec_streaming.call_args[1]["tty"] is False

    def test_checks_both_log_locations(self, mocker):
        """Log lookup checks both ~/.rex and /tmp locations."""
        mock_ssh = mocker.Mock()
        mock_ssh.exec_streaming.return_value = 0

        show_log(mock_ssh, "host", "abc123", follow=False)

        cmd = mock_ssh.exec_streaming.call_args[0][0]
        assert "~/.rex/rex-abc123.log" in cmd
        assert "/tmp/rex-abc123.log" in cmd

    def test_returns_exit_code(self, mocker):
        """Returns exit code from exec_streaming."""
        mock_ssh = mocker.Mock()
        mock_ssh.exec_streaming.return_value = 42

        result = show_log(mock_ssh, "host", "abc123", follow=False)

        assert result == 42
