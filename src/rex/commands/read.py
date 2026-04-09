"""Read files or list directories on remote server."""

from __future__ import annotations

from rex.output import error
from rex.ssh import SSHExecutor
from rex.utils import shell_quote


def read_remote(ssh: SSHExecutor, path: str) -> int:
    """Read a file or list a directory on the remote server.

    If path is a directory, lists its contents.
    If path is a file, displays the file contents.
    Always runs on the login node.

    Returns exit code.
    """
    # Check if path exists and determine type
    code, stdout, stderr = ssh.exec(f'test -e {shell_quote(path)} && echo exists')
    if code != 0 or stdout.strip() != "exists":
        error(f"Path not found: {path}", exit_now=False)
        return 1

    # Check if it's a directory
    code, stdout, _ = ssh.exec(f'test -d {shell_quote(path)} && echo dir')
    is_dir = stdout.strip() == "dir"

    if is_dir:
        return ssh.exec_streaming(f'ls -la {shell_quote(path)}')
    else:
        return ssh.exec_streaming(f'cat {shell_quote(path)}')
