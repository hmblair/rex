"""rex - Remote execution tool for Python and shell commands."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("rex")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"  # Fallback for editable installs without scm
