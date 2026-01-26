"""Connection management commands."""

from __future__ import annotations

import subprocess

from rex.exceptions import SSHError
from rex.output import error
from rex.ssh.connection import SSHConnection
from rex.ssh.executor import SSHExecutor


def manual_ssh(ssh: SSHExecutor) -> int:
    """Open interactive SSH session to target."""
    # Run ssh with TTY allocation for interactive shell
    result = subprocess.run(["ssh", *ssh._opts, "-t", ssh.target])
    return result.returncode


def connect(target: str) -> int:
    """Establish persistent SSH connection."""
    conn = SSHConnection(target)
    try:
        conn.connect()
        return 0
    except SSHError as e:
        error(e.message, exit_now=False)
        return 1


def disconnect(target: str) -> int:
    """Close persistent SSH connection."""
    conn = SSHConnection(target)
    conn.disconnect()
    return 0


def connection_status(target: str | None = None) -> int:
    """Show connection status.

    If target is None, lists all active connections.
    """
    if target is None:
        active = SSHConnection.list_active()
        if not active:
            print("No active connections")
            return 1
        for t, socket in active:
            print(f"connected: {t}")
        return 0
    else:
        conn = SSHConnection(target)
        if conn.is_connected():
            print(f"connected: {target}")
            return 0
        else:
            print(f"disconnected: {target}")
            return 1
