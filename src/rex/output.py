"""Colored output utilities and logging."""

from __future__ import annotations

import logging
import sys

# ANSI color codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"
BOLD = "\033[1m"
NC = "\033[0m"  # No color / reset

# Module logger
_logger = logging.getLogger("rex")


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if not _supports_color():
            return msg
        if record.levelno >= logging.ERROR:
            return f"{RED}{msg}{NC}"
        elif record.levelno >= logging.WARNING:
            return f"{YELLOW}{msg}{NC}"
        elif record.levelno <= logging.DEBUG:
            return f"{DIM}{msg}{NC}"
        return msg


def setup_logging(debug: bool = False) -> None:
    """Configure logging for rex.

    Args:
        debug: If True, set log level to DEBUG. Otherwise, WARNING.
    """
    level = logging.DEBUG if debug else logging.WARNING
    _logger.setLevel(level)

    # Only add handler if none exist
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(ColoredFormatter("%(message)s"))
        _logger.addHandler(handler)
    else:
        # Update existing handler level
        for handler in _logger.handlers:
            handler.setLevel(level)


def get_logger() -> logging.Logger:
    """Get the rex logger for use in other modules."""
    return _logger


def debug(msg: str) -> None:
    """Log debug message (only shown with --debug flag)."""
    _logger.debug(msg)


def _supports_color() -> bool:
    """Check if terminal supports color."""
    if not hasattr(sys.stderr, "isatty"):
        return False
    if not sys.stderr.isatty():
        return False
    return True


def _colorize(color: str, text: str) -> str:
    """Wrap text in color codes if terminal supports it."""
    if _supports_color():
        return f"{color}{text}{NC}"
    return text


def error(msg: str, *, exit_now: bool = True) -> None:
    """Print error message and optionally exit.

    Args:
        msg: The error message to print.
        exit_now: If True (default), exit with code 1 after printing.
    """
    print(_colorize(RED, f"error: {msg}"), file=sys.stderr)
    if exit_now:
        sys.exit(1)


def warn(msg: str) -> None:
    """Print warning message."""
    print(_colorize(YELLOW, f"warning: {msg}"), file=sys.stderr)


def info(msg: str) -> None:
    """Print info message."""
    print(_colorize(CYAN, msg), file=sys.stderr)


def success(msg: str) -> None:
    """Print success message."""
    print(_colorize(GREEN, msg), file=sys.stderr)
