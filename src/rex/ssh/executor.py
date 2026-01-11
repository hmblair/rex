"""SSH command execution."""

import subprocess
import sys
from pathlib import Path

SOCKET_DIR = Path.home() / ".ssh" / "controlmasters"


class SSHExecutor:
    """Execute commands on remote host via SSH."""

    def __init__(self, target: str, verbose: bool = False):
        self.target = target
        self.verbose = verbose
        self._opts = self._build_opts()

    def _socket_path(self) -> Path:
        """Get socket path for this target."""
        return SOCKET_DIR / self.target.replace("@", "--")

    def _build_opts(self) -> list[str]:
        """Build SSH options list."""
        opts = []

        if self.verbose:
            opts.append("-v")

        opts.extend([
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=60",
            "-o", "ServerAliveCountMax=3",
        ])

        socket = self._socket_path()
        opts.extend([
            "-o", f"ControlPath={socket}",
            "-o", "ControlMaster=auto",
        ])

        return opts

    def exec(self, cmd: str) -> tuple[int, str, str]:
        """Execute simple command and capture output.

        Returns (exit_code, stdout, stderr).
        """
        # Use bash --norc --noprofile to skip slow startup scripts
        wrapped = f'bash --norc --noprofile -c {_shell_quote(cmd)}'

        result = subprocess.run(
            ["ssh", *self._opts, self.target, wrapped],
            capture_output=True,
            text=True,
        )

        return (result.returncode, result.stdout, result.stderr)

    def exec_streaming(self, cmd: str, *, tty: bool | None = None) -> int:
        """Execute command with streaming output.

        Inherits parent's stdio for real-time output.
        tty=None means auto-detect from sys.stdin.isatty().

        Returns exit code.
        """
        if tty is None:
            tty = sys.stdin.isatty()

        wrapped = f'bash --norc --noprofile -c {_shell_quote(cmd)}'

        ssh_args = ["ssh", *self._opts]
        if tty:
            ssh_args.append("-t")
        ssh_args.extend([self.target, wrapped])

        result = subprocess.run(ssh_args)
        return result.returncode

    def exec_script(
        self,
        script: str,
        *,
        tty: bool = False,
        login_shell: bool = False,
    ) -> int:
        """Execute script via file-based method (safest for complex scripts).

        Writes script to temp file on remote, executes, cleans up.
        All in a single SSH session.

        Returns exit code.
        """
        shell = "bash -l" if login_shell else "bash"

        # Pattern: write to temp, execute, cleanup in one session
        wrapper = (
            f"{shell} -c 'script=$(mktemp) && "
            f'cat > "$script" && chmod +x "$script" && "$script"; '
            f'e=$?; rm -f "$script"; exit $e\''
        )

        ssh_args = ["ssh", *self._opts]
        if tty:
            ssh_args.append("-t")
        ssh_args.extend([self.target, wrapper])

        result = subprocess.run(ssh_args, input=script.encode())
        return result.returncode

    def exec_script_streaming(
        self,
        script: str,
        *,
        tty: bool | None = None,
        login_shell: bool = False,
    ) -> int:
        """Execute script with streaming output.

        Like exec_script but inherits stdio for interactive use.
        """
        if tty is None:
            tty = sys.stdin.isatty()

        shell = "bash -l" if login_shell else "bash"

        # Write script to temp file, execute, cleanup
        wrapper = (
            f"{shell} -c 'script=$(mktemp) && "
            f'cat > "$script" && chmod +x "$script" && "$script"; '
            f'e=$?; rm -f "$script"; exit $e\''
        )

        ssh_args = ["ssh", *self._opts]
        if tty:
            ssh_args.append("-t")
        ssh_args.extend([self.target, wrapper])

        # Use Popen for streaming stdin
        proc = subprocess.Popen(
            ssh_args,
            stdin=subprocess.PIPE,
            stdout=None,  # Inherit
            stderr=None,  # Inherit
        )
        proc.communicate(input=script.encode())
        return proc.returncode


def _shell_quote(s: str) -> str:
    """Quote string for shell, handling single quotes."""
    # Replace ' with '\''
    escaped = s.replace("'", "'\\''")
    return f"'{escaped}'"
