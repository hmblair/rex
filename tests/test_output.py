"""Tests for rex output functions."""

import logging
import pytest
import sys
from unittest.mock import patch, MagicMock

from rex.output import (
    _supports_color,
    _colorize,
    error,
    warn,
    info,
    success,
    debug,
    setup_logging,
    get_logger,
    RED,
    YELLOW,
    CYAN,
    GREEN,
    NC,
)


class TestSupportsColor:
    """Tests for _supports_color function."""

    def test_returns_true_when_tty(self, mocker):
        """Returns True when stderr is a TTY."""
        mock_stderr = MagicMock()
        mock_stderr.isatty.return_value = True
        mocker.patch.object(sys, "stderr", mock_stderr)
        assert _supports_color() is True

    def test_returns_false_when_not_tty(self, mocker):
        """Returns False when stderr is not a TTY."""
        mock_stderr = MagicMock()
        mock_stderr.isatty.return_value = False
        mocker.patch.object(sys, "stderr", mock_stderr)
        assert _supports_color() is False

    def test_returns_false_when_no_isatty(self, mocker):
        """Returns False when stderr has no isatty method."""
        mock_stderr = MagicMock(spec=[])  # No isatty
        mocker.patch.object(sys, "stderr", mock_stderr)
        assert _supports_color() is False


class TestColorize:
    """Tests for _colorize function."""

    def test_adds_color_when_tty(self, mocker):
        """Wraps text in color codes when TTY."""
        mock_stderr = MagicMock()
        mock_stderr.isatty.return_value = True
        mocker.patch.object(sys, "stderr", mock_stderr)

        result = _colorize(RED, "test")
        assert result == f"{RED}test{NC}"

    def test_no_color_when_not_tty(self, mocker):
        """Returns plain text when not TTY."""
        mock_stderr = MagicMock()
        mock_stderr.isatty.return_value = False
        mocker.patch.object(sys, "stderr", mock_stderr)

        result = _colorize(RED, "test")
        assert result == "test"


class TestWarn:
    """Tests for warn function."""

    def test_prints_to_stderr(self, capsys, mocker):
        """warn() prints to stderr."""
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = False
        mocker.patch.object(sys, "stdout", mock_stdout)

        warn("test warning")
        captured = capsys.readouterr()
        assert "warning: test warning" in captured.err
        assert captured.out == ""

    def test_does_not_exit(self, mocker):
        """warn() does not exit."""
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = False
        mocker.patch.object(sys, "stdout", mock_stdout)
        mock_exit = mocker.patch.object(sys, "exit")

        warn("test warning")
        mock_exit.assert_not_called()


class TestInfo:
    """Tests for info function."""

    def test_prints_to_stderr(self, capsys, mocker):
        """info() prints to stderr."""
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = False
        mocker.patch.object(sys, "stdout", mock_stdout)

        info("test info")
        captured = capsys.readouterr()
        assert "test info" in captured.err
        assert captured.out == ""


class TestSuccess:
    """Tests for success function."""

    def test_prints_to_stderr(self, capsys, mocker):
        """success() prints to stderr."""
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = False
        mocker.patch.object(sys, "stdout", mock_stdout)

        success("test success")
        captured = capsys.readouterr()
        assert "test success" in captured.err
        assert captured.out == ""


class TestError:
    """Tests for error function."""

    def test_prints_to_stderr(self, capsys, mocker):
        """error() prints to stderr before exiting."""
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = False
        mocker.patch.object(sys, "stdout", mock_stdout)

        with pytest.raises(SystemExit):
            error("test error")

        captured = capsys.readouterr()
        assert "error: test error" in captured.err

    def test_exits_with_code_1(self, mocker):
        """error() exits with code 1."""
        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = False
        mocker.patch.object(sys, "stdout", mock_stdout)

        with pytest.raises(SystemExit) as exc_info:
            error("test error")
        assert exc_info.value.code == 1


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_sets_debug_level_when_debug_true(self):
        """setup_logging sets DEBUG level when debug=True."""
        setup_logging(debug=True)
        logger = get_logger()
        assert logger.level == logging.DEBUG

    def test_sets_warning_level_when_debug_false(self):
        """setup_logging sets WARNING level when debug=False."""
        setup_logging(debug=False)
        logger = get_logger()
        assert logger.level == logging.WARNING

    def test_adds_handler(self):
        """setup_logging adds a handler."""
        # Clear any existing handlers
        logger = get_logger()
        logger.handlers.clear()

        setup_logging(debug=False)
        assert len(logger.handlers) >= 1


class TestDebug:
    """Tests for debug function."""

    def test_logs_at_debug_level(self, caplog):
        """debug() logs at DEBUG level."""
        setup_logging(debug=True)
        with caplog.at_level(logging.DEBUG, logger="rex"):
            debug("test debug message")
        assert "test debug message" in caplog.text

    def test_not_shown_when_debug_disabled(self, caplog):
        """debug() not shown when debug mode disabled."""
        setup_logging(debug=False)
        with caplog.at_level(logging.WARNING, logger="rex"):
            debug("hidden message")
        assert "hidden message" not in caplog.text


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger(self):
        """get_logger returns a Logger instance."""
        logger = get_logger()
        assert isinstance(logger, logging.Logger)

    def test_returns_rex_logger(self):
        """get_logger returns the 'rex' logger."""
        logger = get_logger()
        assert logger.name == "rex"
