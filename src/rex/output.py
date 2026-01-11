"""Colored output utilities."""

import sys

# ANSI color codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
NC = "\033[0m"  # No color / reset


def _supports_color() -> bool:
    """Check if terminal supports color."""
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    return True


def _colorize(color: str, text: str) -> str:
    """Wrap text in color codes if terminal supports it."""
    if _supports_color():
        return f"{color}{text}{NC}"
    return text


def error(msg: str) -> None:
    """Print error message and exit."""
    print(_colorize(RED, f"error: {msg}"), file=sys.stderr)
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
