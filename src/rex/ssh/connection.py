"""SSH connection multiplexing via ControlMaster."""

from __future__ import annotations

import subprocess
from pathlib import Path

from rex.output import error, info, success, warn

SOCKET_DIR = Path.home() / ".ssh" / "controlmasters"


class SSHConnection:
    """Manages SSH ControlMaster connections."""

    def __init__(self, target: str):
        self.target = target
        self.socket_path = self._socket_path()

    def _socket_path(self) -> Path:
        """Get socket path for this target.

        Uses ~/.ssh/controlmasters/{user}--{host} format.
        The @ is replaced with -- to avoid confusing SSH option parser.
        """
        return SOCKET_DIR / self.target.replace("@", "--")

    def is_connected(self) -> bool:
        """Check if multiplexed connection is active."""
        if not self.socket_path.exists():
            return False

        result = subprocess.run(
            ["ssh", "-O", "check", "-o", f"ControlPath={self.socket_path}", self.target],
            capture_output=True,
        )
        return result.returncode == 0

    def connect(self) -> bool:
        """Establish persistent connection.

        Returns True on success, False on failure.
        """
        if self.is_connected():
            info(f"Already connected to {self.target}")
            return True

        # Remove stale socket if exists
        if self.socket_path.exists():
            self.socket_path.unlink()

        # Create socket directory
        SOCKET_DIR.mkdir(parents=True, exist_ok=True)
        SOCKET_DIR.chmod(0o700)

        info(f"Connecting to {self.target}...")

        result = subprocess.run(
            [
                "ssh",
                "-o", "ControlMaster=yes",
                "-o", f"ControlPath={self.socket_path}",
                "-o", "ControlPersist=yes",
                "-o", "ConnectTimeout=10",
                "-o", "ServerAliveInterval=60",
                "-o", "ServerAliveCountMax=3",
                "-fN",
                self.target,
            ],
        )

        if result.returncode != 0:
            error(f"Failed to connect to {self.target}")
            return False

        success(f"Connected to {self.target}")
        return True

    def disconnect(self) -> bool:
        """Close persistent connection.

        Returns True on success, False if no connection.
        """
        if not self.socket_path.exists():
            warn(f"No active connection to {self.target}")
            return False

        subprocess.run(
            ["ssh", "-O", "exit", "-o", f"ControlPath={self.socket_path}", self.target],
            capture_output=True,
        )

        if self.socket_path.exists():
            self.socket_path.unlink()

        success(f"Disconnected from {self.target}")
        return True

    @classmethod
    def list_active(cls) -> list[tuple[str, str]]:
        """List all active connections.

        Returns list of (target, socket_path) tuples.
        """
        active: list[tuple[str, str]] = []
        if not SOCKET_DIR.exists():
            return active

        for socket in SOCKET_DIR.iterdir():
            if not socket.is_socket():
                continue

            # Reconstruct target: -- becomes @
            target = socket.name.replace("--", "@")

            # Verify connection is alive
            result = subprocess.run(
                ["ssh", "-O", "check", "-o", f"ControlPath={socket}", target],
                capture_output=True,
            )

            if result.returncode == 0:
                active.append((target, str(socket)))
            else:
                # Stale socket, remove it
                socket.unlink()

        return active
