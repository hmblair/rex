"""Shared test fixtures for rex."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_subprocess(mocker):
    """Mock subprocess.run for SSH/rsync commands."""
    mock = mocker.patch("subprocess.run")
    mock.return_value = MagicMock(returncode=0, stdout="", stderr="")
    return mock


@pytest.fixture
def mock_subprocess_popen(mocker):
    """Mock subprocess.Popen for streaming commands."""
    mock_popen = MagicMock()
    mock_popen.communicate.return_value = (b"", b"")
    mock_popen.returncode = 0
    mocker.patch("subprocess.Popen", return_value=mock_popen)
    return mock_popen


@pytest.fixture
def mock_ssh_executor(mocker):
    """Mock SSHExecutor for testing without real SSH."""
    from rex.ssh.executor import SSHExecutor

    executor = MagicMock(spec=SSHExecutor)
    executor.target = "user@host"
    executor._opts = []
    executor.exec.return_value = (0, "", "")
    executor.exec_streaming.return_value = 0
    executor.exec_script.return_value = 0
    executor.exec_script_streaming.return_value = 0
    return executor


@pytest.fixture
def mock_file_transfer(mock_ssh_executor):
    """Mock FileTransfer for testing without real rsync."""
    from rex.ssh.transfer import FileTransfer

    transfer = FileTransfer("user@host", mock_ssh_executor)
    return transfer


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project with .rex.toml."""
    config = tmp_path / ".rex.toml"
    config.write_text(
        """
host = "user@cluster"
code_dir = "~/project"
modules = ["python/3.11"]
cpu_partition = "cpu"
gpu_partition = "gpu"
"""
    )
    return tmp_path


@pytest.fixture
def temp_script(tmp_path):
    """Create a temporary Python script."""
    script = tmp_path / "test_script.py"
    script.write_text('print("hello")')
    return script


@pytest.fixture
def temp_alias_file(tmp_path):
    """Create a temporary alias config file."""
    alias_file = tmp_path / "rex_aliases"
    alias_file.write_text(
        """
# Test aliases
gpu = user@gpu.cluster -p python3.11 -s --partition gpu
cpu = user@cpu.cluster
"""
    )
    return alias_file


@pytest.fixture
def mock_stdin_piped(mocker):
    """Mock sys.stdin as piped (non-TTY) with content."""
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = False
    mock_stdin.read.return_value = 'print("hello from stdin")'
    mocker.patch("sys.stdin", mock_stdin)
    return mock_stdin


@pytest.fixture
def mock_stdin_tty(mocker):
    """Mock sys.stdin as interactive TTY."""
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    mocker.patch("sys.stdin", mock_stdin)
    return mock_stdin


@pytest.fixture
def mock_home_dir(mocker, tmp_path):
    """Mock Path.home() to return a temp directory."""
    mocker.patch.object(Path, "home", return_value=tmp_path)
    return tmp_path


@pytest.fixture
def capture_stderr(capsys):
    """Helper to capture stderr for testing output functions."""
    return capsys
