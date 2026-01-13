"""File transfer operations (rsync, scp)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from rex.exceptions import TransferError
from rex.output import info, success
from rex.ssh.executor import SSHExecutor
from rex.utils import map_to_remote

# Default rsync exclusions for Python projects
PYTHON_EXCLUDES = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".git",
    ".pytest_cache",
    "*.egg-info",
    "build",
    "dist",
    ".tox",
    ".eggs",
    ".mypy_cache",
    ".ruff_cache",
    "*.so",
    ".venv",
    "venv",
]


class FileTransfer:
    """File transfer operations via rsync/scp."""

    def __init__(self, target: str, executor: SSHExecutor):
        self.target = target
        self.executor = executor

    def push(self, local: Path, remote: str | None = None) -> None:
        """Upload file/directory to remote.

        If remote is None, mirrors local path structure under remote $HOME.

        Raises:
            TransferError: If the transfer fails.
        """
        local = local.resolve()
        if not local.exists():
            raise TransferError(f"Path not found: {local}")

        if remote is None:
            # Get remote home and map path
            code, stdout, _ = self.executor.exec("echo $HOME")
            if code != 0 or not stdout.strip():
                raise TransferError("Failed to get remote home directory")
            remote = map_to_remote(local, stdout.strip())

        # Create remote parent directory
        remote_parent = str(Path(remote).parent)
        code, _, _ = self.executor.exec(f"mkdir -p {_shell_quote(remote_parent)}")
        if code != 0:
            raise TransferError("Failed to create remote directory")

        info(f"Pushing to {self.target}:{remote}")

        if local.is_dir():
            args = ["rsync", "-avz", "--progress", f"{local}/", f"{self.target}:{remote}/"]
        else:
            args = ["rsync", "-avz", "--progress", str(local), f"{self.target}:{remote}"]

        result = subprocess.run(args)
        if result.returncode != 0:
            raise TransferError("Push failed")

        success(f"Pushed {local.name}")

    def pull(self, remote: str, local: Path | None = None) -> None:
        """Download file/directory from remote (supports globs).

        If local is None, downloads to current directory.

        Raises:
            TransferError: If the transfer fails.
        """
        if local is None:
            local = Path.cwd()
        local = local.resolve()

        # Create local directory
        local.mkdir(parents=True, exist_ok=True)

        info(f"Pulling from {self.target}:{remote}")

        args = ["rsync", "-avz", "--progress", f"{self.target}:{remote}", f"{local}/"]
        result = subprocess.run(args)

        if result.returncode != 0:
            raise TransferError("Pull failed")

        success(f"Pulled to {local}")

    def sync(
        self,
        local: Path,
        remote: str | None = None,
        *,
        excludes: list[str] | None = None,
        delete: bool = True,
    ) -> None:
        """Rsync project to remote with Python project defaults.

        Raises:
            TransferError: If the sync fails.
        """
        local = local.resolve()
        if not local.is_dir():
            raise TransferError(f"Directory not found: {local}")

        if remote is None:
            # Get remote home and map path
            code, stdout, _ = self.executor.exec("echo $HOME")
            if code != 0 or not stdout.strip():
                raise TransferError("Failed to get remote home directory")
            remote = map_to_remote(local, stdout.strip())

        # Create remote parent directory
        remote_parent = str(Path(remote).parent)
        code, _, _ = self.executor.exec(f"mkdir -p {_shell_quote(remote_parent)}")
        if code != 0:
            raise TransferError("Failed to create remote directory")

        info(f"Syncing to {self.target}:{remote}")

        # Build rsync args
        if excludes is None:
            excludes = PYTHON_EXCLUDES

        args = ["rsync", "-avz"]
        if delete:
            args.append("--delete")
        for ex in excludes:
            args.extend(["--exclude", ex])
        args.extend([f"{local}/", f"{self.target}:{remote}/"])

        result = subprocess.run(args)
        if result.returncode != 0:
            raise TransferError("Sync failed")

        success(f"Synced {local.name}")


def _shell_quote(s: str) -> str:
    """Quote string for shell."""
    escaped = s.replace("'", "'\\''")
    return f"'{escaped}'"
