"""File transfer commands."""

from pathlib import Path

from rex.config.project import ProjectConfig
from rex.output import info, success
from rex.ssh.executor import SSHExecutor
from rex.ssh.transfer import FileTransfer


def push(
    transfer: FileTransfer,
    local: Path,
    remote: str | None = None,
) -> int:
    """Push file/directory to remote."""
    if transfer.push(local, remote):
        return 0
    return 1


def pull(
    transfer: FileTransfer,
    remote: str,
    local: Path | None = None,
) -> int:
    """Pull file/directory from remote."""
    if transfer.pull(remote, local):
        return 0
    return 1


def sync(
    transfer: FileTransfer,
    ssh: SSHExecutor,
    project: ProjectConfig | None,
    local_path: Path | None = None,
    code_dir: str | None = None,
    python: str = "python3",
    no_install: bool = False,
) -> int:
    """Sync project to remote.

    If project config exists, uses code_dir from config.
    If no_install is False and pyproject.toml exists, runs pip install -e.
    """
    # Determine local path
    if local_path is None:
        if project:
            local_path = project.root
        else:
            local_path = Path.cwd()

    local_path = local_path.resolve()

    # Determine remote path
    remote_path = code_dir
    if remote_path is None and project:
        remote_path = project.code_dir

    # Sync
    if not transfer.sync(local_path, remote_path):
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

            code, _, _ = ssh.exec(
                f"cd {actual_remote} && {python} -m pip install -e . -q"
            )
            if code == 0:
                success(f"Installed {local_path.name}")
            else:
                from rex.output import warn
                warn("pip install failed")

    return 0
