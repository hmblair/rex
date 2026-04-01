"""File transfer commands."""

from __future__ import annotations

from pathlib import Path

from rex.config.resolved import ResolvedConfig
from rex.exceptions import TransferError
from rex.output import error
from rex.ssh.transfer import FileTransfer


def push(
    transfer: FileTransfer,
    local: Path,
    remote: str | None = None,
) -> int:
    """Push file/directory to remote."""
    try:
        transfer.push(local, remote)
        return 0
    except TransferError as e:
        error(e.message, exit_now=False)
        return e.exit_code


def pull(
    transfer: FileTransfer,
    remote: str,
    local: Path | None = None,
) -> int:
    """Pull file/directory from remote."""
    try:
        transfer.pull(remote, local)
        return 0
    except TransferError as e:
        error(e.message, exit_now=False)
        return e.exit_code


def sync(
    transfer: FileTransfer,
    config: ResolvedConfig,
    local_path: Path | None = None,
) -> int:
    """Sync project to remote.

    Uses code_dir and root from resolved config.
    """
    # Determine local path
    if local_path is None:
        if config.root:
            local_path = config.root
        else:
            local_path = Path.cwd()

    local_path = local_path.resolve()

    # Determine remote path from resolved config
    ctx = config.execution
    remote_path = ctx.code_dir if ctx else None

    # Sync
    try:
        transfer.sync(local_path, remote_path, excludes=config.sync_excludes)
    except TransferError as e:
        error(e.message, exit_now=False)
        return e.exit_code

    return 0
