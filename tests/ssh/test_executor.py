"""Tests for SSH executor."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rex.ssh.executor import SSHExecutor, _shell_quote, SOCKET_DIR


class TestShellQuote:
    """Tests for _shell_quote helper function."""

    def test_simple_string(self):
        """Simple strings are wrapped in single quotes."""
        assert _shell_quote("hello") == "'hello'"

    def test_string_with_spaces(self):
        """Strings with spaces are properly quoted."""
        assert _shell_quote("hello world") == "'hello world'"

    def test_string_with_single_quote(self):
        """Single quotes are escaped."""
        result = _shell_quote("it's")
        assert result == "'it'\\''s'"

    def test_empty_string(self):
        """Empty string returns empty quotes."""
        assert _shell_quote("") == "''"

    def test_special_characters(self):
        """Special shell characters are safely quoted."""
        result = _shell_quote("echo $HOME; rm -rf /")
        assert result == "'echo $HOME; rm -rf /'"


class TestSSHExecutorInit:
    """Tests for SSHExecutor initialization."""

    def test_basic_init(self):
        """Basic initialization sets target."""
        executor = SSHExecutor("user@host")
        assert executor.target == "user@host"
        assert executor.verbose is False

    def test_verbose_init(self):
        """Verbose mode can be enabled."""
        executor = SSHExecutor("user@host", verbose=True)
        assert executor.verbose is True
        assert "-v" in executor._opts

    def test_socket_path(self):
        """Socket path replaces @ with --."""
        executor = SSHExecutor("user@host.example.com")
        socket = executor._socket_path()
        assert socket.parent == SOCKET_DIR
        assert socket.name == "user--host.example.com"

    def test_opts_include_control_path(self):
        """Options include ControlPath for multiplexing."""
        executor = SSHExecutor("user@host")
        opts_str = " ".join(executor._opts)
        assert "ControlPath=" in opts_str
        assert "ControlMaster=auto" in opts_str

    def test_opts_include_timeouts(self):
        """Options include connection timeouts."""
        executor = SSHExecutor("user@host")
        opts_str = " ".join(executor._opts)
        assert "ConnectTimeout=10" in opts_str
        assert "ServerAliveInterval=60" in opts_str


class TestSSHExecutorExec:
    """Tests for SSHExecutor.exec method."""

    def test_exec_returns_tuple(self, mocker):
        """exec() returns (returncode, stdout, stderr) tuple."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="output\n",
            stderr=""
        )

        executor = SSHExecutor("user@host")
        code, stdout, stderr = executor.exec("echo hello")

        assert code == 0
        assert stdout == "output\n"
        assert stderr == ""

    def test_exec_captures_failure(self, mocker):
        """exec() captures non-zero exit codes."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="error message"
        )

        executor = SSHExecutor("user@host")
        code, stdout, stderr = executor.exec("false")

        assert code == 1
        assert stderr == "error message"

    def test_exec_wraps_in_bash(self, mocker):
        """exec() wraps command in bash -c."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        executor = SSHExecutor("user@host")
        executor.exec("echo $HOME")

        args = mock_run.call_args[0][0]
        # Command should be wrapped
        assert "bash" in " ".join(args)
        assert "--norc" in " ".join(args)


class TestSSHExecutorExecStreaming:
    """Tests for SSHExecutor.exec_streaming method."""

    def test_exec_streaming_returns_code(self, mocker):
        """exec_streaming() returns exit code."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        executor = SSHExecutor("user@host")
        # Mock stdin.isatty()
        mocker.patch("sys.stdin.isatty", return_value=False)

        code = executor.exec_streaming("echo hello")
        assert code == 0

    def test_exec_streaming_with_tty(self, mocker):
        """exec_streaming() adds -t when tty=True."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        executor = SSHExecutor("user@host")
        executor.exec_streaming("echo hello", tty=True)

        args = mock_run.call_args[0][0]
        assert "-t" in args

    def test_exec_streaming_without_tty(self, mocker):
        """exec_streaming() omits -t when tty=False."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        executor = SSHExecutor("user@host")
        executor.exec_streaming("echo hello", tty=False)

        args = mock_run.call_args[0][0]
        assert "-t" not in args


class TestSSHExecutorExecScript:
    """Tests for SSHExecutor.exec_script method."""

    def test_exec_script_returns_code(self, mocker):
        """exec_script() returns exit code."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        executor = SSHExecutor("user@host")
        code = executor.exec_script("#!/bin/bash\necho hello")

        assert code == 0

    def test_exec_script_sends_input(self, mocker):
        """exec_script() sends script as input."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        executor = SSHExecutor("user@host")
        script = "#!/bin/bash\necho hello"
        executor.exec_script(script)

        # Check input was passed
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["input"] == script.encode()

    def test_exec_script_with_login_shell(self, mocker):
        """exec_script() uses bash -l for login shell."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        executor = SSHExecutor("user@host")
        executor.exec_script("echo hello", login_shell=True)

        args = mock_run.call_args[0][0]
        # Should use bash -l
        cmd_str = " ".join(args)
        assert "bash -l" in cmd_str


class TestSSHExecutorExecScriptStreaming:
    """Tests for SSHExecutor.exec_script_streaming method."""

    def test_exec_script_streaming_returns_code(self, mocker):
        """exec_script_streaming() returns exit code."""
        mock_popen = MagicMock()
        mock_popen.communicate.return_value = (b"", b"")
        mock_popen.returncode = 0
        mocker.patch("subprocess.Popen", return_value=mock_popen)
        mocker.patch("sys.stdin.isatty", return_value=False)

        executor = SSHExecutor("user@host")
        code = executor.exec_script_streaming("echo hello")

        assert code == 0

    def test_exec_script_streaming_uses_popen(self, mocker):
        """exec_script_streaming() uses Popen for streaming."""
        mock_popen = MagicMock()
        mock_popen.communicate.return_value = (b"", b"")
        mock_popen.returncode = 0
        mock_popen_class = mocker.patch("subprocess.Popen", return_value=mock_popen)
        mocker.patch("sys.stdin.isatty", return_value=False)

        executor = SSHExecutor("user@host")
        executor.exec_script_streaming("echo hello")

        mock_popen_class.assert_called_once()
        mock_popen.communicate.assert_called_once()
