"""Read files or list directories on remote server."""

from __future__ import annotations

from rex.output import error
from rex.ssh import SSHExecutor


def read_remote(ssh: SSHExecutor, path: str) -> int:
    """Read a file or list a directory on the remote server.

    If path is a directory, lists its contents.
    If path is a file, displays the file contents.
    Always runs on the login node.

    Returns exit code.
    """
    # Check if path exists and determine type
    code, stdout, stderr = ssh.exec(f'test -e {_quote(path)} && echo exists')
    if code != 0 or stdout.strip() != "exists":
        error(f"Path not found: {path}", exit_now=False)
        return 1

    # Check if it's a directory
    code, stdout, _ = ssh.exec(f'test -d {_quote(path)} && echo dir')
    is_dir = stdout.strip() == "dir"

    if is_dir:
        # List directory contents
        return ssh.exec_streaming(f'ls -la {_quote(path)}')
    else:
        # Display file contents
        return ssh.exec_streaming(f'cat {_quote(path)}')


def _quote(s: str) -> str:
    """Quote string for shell."""
    escaped = s.replace("'", "'\\''")
    return f"'{escaped}'"
