"""Tests for SSH executor."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rex.exceptions import SSHError
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

    def test_double_quotes(self):
        """Double quotes are preserved inside single quotes."""
        result = _shell_quote('echo "hello world"')
        assert result == '\'echo "hello world"\''

    def test_mixed_quotes(self):
        """Mixed single and double quotes are handled."""
        result = _shell_quote("echo \"it's working\"")
        assert result == "'echo \"it'\\''s working\"'"

    def test_dollar_sign_variable(self):
        """Dollar signs are preserved (not expanded)."""
        result = _shell_quote("echo $HOME $USER")
        assert result == "'echo $HOME $USER'"

    def test_backticks(self):
        """Backticks are preserved."""
        result = _shell_quote("echo `date`")
        assert result == "'echo `date`'"

    def test_pipe(self):
        """Pipe characters are preserved."""
        result = _shell_quote("ls -la | grep foo")
        assert result == "'ls -la | grep foo'"

    def test_semicolon(self):
        """Semicolons are preserved."""
        result = _shell_quote("cmd1; cmd2; cmd3")
        assert result == "'cmd1; cmd2; cmd3'"

    def test_ampersand(self):
        """Ampersands are preserved."""
        result = _shell_quote("cmd1 && cmd2 || cmd3")
        assert result == "'cmd1 && cmd2 || cmd3'"

    def test_backslash(self):
        """Backslashes are preserved."""
        result = _shell_quote("echo \\n\\t")
        assert result == "'echo \\n\\t'"

    def test_parentheses(self):
        """Parentheses are preserved."""
        result = _shell_quote("(cd /tmp && ls)")
        assert result == "'(cd /tmp && ls)'"

    def test_brackets(self):
        """Brackets are preserved."""
        result = _shell_quote("[[ -f /tmp/test ]] && echo yes")
        assert result == "'[[ -f /tmp/test ]] && echo yes'"

    def test_glob_characters(self):
        """Glob characters are preserved."""
        result = _shell_quote("ls *.py **/*.txt")
        assert result == "'ls *.py **/*.txt'"

    def test_multiple_single_quotes(self):
        """Multiple single quotes are all escaped."""
        result = _shell_quote("echo 'one' 'two' 'three'")
        assert result == "'echo '\\''one'\\'' '\\''two'\\'' '\\''three'\\'''"

    def test_newline(self):
        """Newlines are preserved."""
        result = _shell_quote("echo first\necho second")
        assert result == "'echo first\necho second'"

    def test_complex_command(self):
        """Complex real-world command is properly quoted."""
        cmd = '''for f in *.py; do echo "$f"; done'''
        result = _shell_quote(cmd)
        assert result == "'" + cmd + "'"


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


class TestSSHExecutorExecSpecialChars:
    """Tests for exec() with special characters."""

    def test_exec_with_double_quotes(self, mocker):
        """exec() preserves double quotes in command."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        executor = SSHExecutor("user@host")
        executor.exec('echo "hello world"')

        args = mock_run.call_args[0][0]
        cmd_str = " ".join(args)
        assert '"hello world"' in cmd_str

    def test_exec_with_single_quotes(self, mocker):
        """exec() preserves single quotes in command."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        executor = SSHExecutor("user@host")
        executor.exec("echo 'hello world'")

        args = mock_run.call_args[0][0]
        cmd_str = " ".join(args)
        # Single quotes should be escaped within the outer quotes
        assert "hello world" in cmd_str

    def test_exec_with_dollar_variable(self, mocker):
        """exec() preserves dollar sign variables."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        executor = SSHExecutor("user@host")
        executor.exec("echo $HOME $USER")

        args = mock_run.call_args[0][0]
        cmd_str = " ".join(args)
        assert "$HOME" in cmd_str
        assert "$USER" in cmd_str

    def test_exec_with_pipe(self, mocker):
        """exec() preserves pipe characters."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        executor = SSHExecutor("user@host")
        executor.exec("ls -la | grep foo | wc -l")

        args = mock_run.call_args[0][0]
        cmd_str = " ".join(args)
        assert "|" in cmd_str

    def test_exec_with_semicolon(self, mocker):
        """exec() preserves semicolons."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        executor = SSHExecutor("user@host")
        executor.exec("cd /tmp; ls; pwd")

        args = mock_run.call_args[0][0]
        cmd_str = " ".join(args)
        assert ";" in cmd_str

    def test_exec_with_ampersand(self, mocker):
        """exec() preserves ampersands."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        executor = SSHExecutor("user@host")
        executor.exec("cmd1 && cmd2 || cmd3")

        args = mock_run.call_args[0][0]
        cmd_str = " ".join(args)
        assert "&&" in cmd_str
        assert "||" in cmd_str

    def test_exec_with_backticks(self, mocker):
        """exec() preserves backticks."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        executor = SSHExecutor("user@host")
        executor.exec("echo `date`")

        args = mock_run.call_args[0][0]
        cmd_str = " ".join(args)
        assert "`" in cmd_str

    def test_exec_with_backslash(self, mocker):
        """exec() preserves backslashes."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        executor = SSHExecutor("user@host")
        executor.exec("echo 'line1\\nline2'")

        args = mock_run.call_args[0][0]
        cmd_str = " ".join(args)
        assert "\\" in cmd_str

    def test_exec_with_parentheses(self, mocker):
        """exec() preserves parentheses for subshells."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        executor = SSHExecutor("user@host")
        executor.exec("(cd /tmp && ls)")

        args = mock_run.call_args[0][0]
        cmd_str = " ".join(args)
        assert "(" in cmd_str
        assert ")" in cmd_str

    def test_exec_with_glob(self, mocker):
        """exec() preserves glob patterns."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        executor = SSHExecutor("user@host")
        executor.exec("ls *.py")

        args = mock_run.call_args[0][0]
        cmd_str = " ".join(args)
        assert "*" in cmd_str

    def test_exec_with_complex_command(self, mocker):
        """exec() handles complex real-world commands."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        executor = SSHExecutor("user@host")
        cmd = '''for f in *.py; do echo "$f"; done'''
        executor.exec(cmd)

        args = mock_run.call_args[0][0]
        cmd_str = " ".join(args)
        assert "for" in cmd_str
        assert "done" in cmd_str
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


class TestSSHExecutorCheckConnection:
    """Tests for SSHExecutor.check_connection method."""

    def test_check_connection_with_valid_socket(self, mocker, tmp_path):
        """check_connection() succeeds with valid ControlMaster socket."""
        # Create a mock socket file
        socket_dir = tmp_path / ".ssh" / "controlmasters"
        socket_dir.mkdir(parents=True)
        socket_file = socket_dir / "user--host"
        socket_file.touch()

        mocker.patch("rex.ssh.executor.SOCKET_DIR", socket_dir)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        executor = SSHExecutor("user@host")
        # Should not raise
        executor.check_connection()

        # Should have called ssh -O check
        args = mock_run.call_args[0][0]
        assert "-O" in args
        assert "check" in args

    def test_check_connection_without_socket_success(self, mocker, tmp_path):
        """check_connection() succeeds when no socket but SSH works."""
        socket_dir = tmp_path / ".ssh" / "controlmasters"
        socket_dir.mkdir(parents=True)

        mocker.patch("rex.ssh.executor.SOCKET_DIR", socket_dir)
        mock_run = mocker.patch("subprocess.run")
        # First call (socket check) won't happen, second call (connection test) succeeds
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        executor = SSHExecutor("user@host")
        # Should not raise
        executor.check_connection()

    def test_check_connection_permission_denied(self, mocker, tmp_path):
        """check_connection() raises SSHError on permission denied."""
        socket_dir = tmp_path / ".ssh" / "controlmasters"
        socket_dir.mkdir(parents=True)

        mocker.patch("rex.ssh.executor.SOCKET_DIR", socket_dir)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(
            returncode=255,
            stderr="Permission denied (publickey,password)."
        )

        executor = SSHExecutor("user@host")
        with pytest.raises(SSHError) as exc_info:
            executor.check_connection()

        assert "Permission denied" in str(exc_info.value)
        assert "--connect" in str(exc_info.value)

    def test_check_connection_hostname_not_found(self, mocker, tmp_path):
        """check_connection() raises SSHError for unknown hostname."""
        socket_dir = tmp_path / ".ssh" / "controlmasters"
        socket_dir.mkdir(parents=True)

        mocker.patch("rex.ssh.executor.SOCKET_DIR", socket_dir)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(
            returncode=255,
            stderr="ssh: Could not resolve hostname badhost: Name or service not known"
        )

        executor = SSHExecutor("user@badhost")
        with pytest.raises(SSHError) as exc_info:
            executor.check_connection()

        assert "Could not resolve hostname" in str(exc_info.value)

    def test_check_connection_stale_socket_removed(self, mocker, tmp_path):
        """check_connection() removes stale socket and retries."""
        socket_dir = tmp_path / ".ssh" / "controlmasters"
        socket_dir.mkdir(parents=True)
        socket_file = socket_dir / "user--host"
        socket_file.touch()

        mocker.patch("rex.ssh.executor.SOCKET_DIR", socket_dir)
        mock_run = mocker.patch("subprocess.run")
        # First call: socket check fails (stale)
        # Second call: connection test succeeds
        mock_run.side_effect = [
            MagicMock(returncode=1),  # ssh -O check fails
            MagicMock(returncode=0, stderr=""),  # connection test succeeds
        ]

        executor = SSHExecutor("user@host")
        executor.check_connection()

        # Stale socket should be removed
        assert not socket_file.exists()

    def test_check_connection_timeout(self, mocker, tmp_path):
        """check_connection() raises SSHError on connection timeout."""
        socket_dir = tmp_path / ".ssh" / "controlmasters"
        socket_dir.mkdir(parents=True)

        mocker.patch("rex.ssh.executor.SOCKET_DIR", socket_dir)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(
            returncode=255,
            stderr="ssh: connect to host example.com port 22: Connection timed out"
        )

        executor = SSHExecutor("user@example.com")
        with pytest.raises(SSHError) as exc_info:
            executor.check_connection()

        assert "timed out" in str(exc_info.value)
