"""File transfer commands."""

from __future__ import annotations

from pathlib import Path

from rex.config.resolved import ResolvedConfig
from rex.exceptions import TransferError
from rex.output import error, info, success
from rex.ssh.executor import SSHExecutor
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
        return 1


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
        return 1


def sync(
    transfer: FileTransfer,
    ssh: SSHExecutor,
    config: ResolvedConfig,
    local_path: Path | None = None,
    no_install: bool = False,
) -> int:
    """Sync project to remote.

    Uses code_dir and root from resolved config.
    If no_install is False and pyproject.toml exists, runs pip install -e.
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
        transfer.sync(local_path, remote_path)
    except TransferError as e:
        error(e.message, exit_now=False)
        return 1

    # Pip install if applicable
    if not no_install and remote_path is None:
        # Only auto-install if not using project config
        pyproject = local_path / "pyproject.toml"
        setup_py = local_path / "setup.py"

        if pyproject.exists() or setup_py.exists():
            info("Installing package...")
            # Get actual remote path
            code, stdout, _ = ssh.exec("echo $HOME")
            remote_home = stdout.strip()
            from rex.utils import map_to_remote
            actual_remote = map_to_remote(local_path, remote_home)

            python = ctx.python if ctx else "python3"
            code, _, _ = ssh.exec(
                f"cd {actual_remote} && {python} -m pip install -e . -q"
            )
            if code == 0:
                success(f"Installed {local_path.name}")
            else:
                from rex.output import warn
                warn("pip install failed")

    return 0
