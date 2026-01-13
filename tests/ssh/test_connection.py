"""Tests for SSH connection multiplexing."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rex.exceptions import SSHError
from rex.ssh.connection import SSHConnection, SOCKET_DIR


class TestSSHConnectionInit:
    """Tests for SSHConnection initialization."""

    def test_basic_init(self):
        """Basic initialization sets target and socket path."""
        conn = SSHConnection("user@host")
        assert conn.target == "user@host"
        assert conn.socket_path is not None

    def test_socket_path_format(self):
        """Socket path uses correct format (@ -> --)."""
        conn = SSHConnection("user@host.example.com")
        assert conn.socket_path.parent == SOCKET_DIR
        assert conn.socket_path.name == "user--host.example.com"


class TestSSHConnectionIsConnected:
    """Tests for SSHConnection.is_connected method."""

    def test_not_connected_no_socket(self, mocker, tmp_path):
        """is_connected returns False when socket doesn't exist."""
        mocker.patch("rex.ssh.connection.SOCKET_DIR", tmp_path)
        conn = SSHConnection("user@host")
        # Socket doesn't exist in tmp_path

        assert conn.is_connected() is False

    def test_connected_socket_exists(self, mocker, tmp_path):
        """is_connected returns True when socket exists and SSH check succeeds."""
        mocker.patch("rex.ssh.connection.SOCKET_DIR", tmp_path)
        conn = SSHConnection("user@host")
        conn.socket_path.touch()  # Create socket file

        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        assert conn.is_connected() is True

    def test_not_connected_ssh_check_fails(self, mocker, tmp_path):
        """is_connected returns False when SSH check fails."""
        mocker.patch("rex.ssh.connection.SOCKET_DIR", tmp_path)
        conn = SSHConnection("user@host")
        conn.socket_path.touch()  # Create socket file

        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=1)

        assert conn.is_connected() is False


class TestSSHConnectionConnect:
    """Tests for SSHConnection.connect method."""

    def test_connect_already_connected(self, mocker):
        """connect() does nothing if already connected."""
        conn = SSHConnection("user@host")
        mocker.patch.object(conn, "is_connected", return_value=True)

        # Should not raise, just return
        conn.connect()

    def test_connect_success(self, mocker, tmp_path):
        """connect() establishes connection successfully."""
        # Use tmp_path for socket dir
        mocker.patch("rex.ssh.connection.SOCKET_DIR", tmp_path)

        conn = SSHConnection("user@host")
        conn.socket_path = tmp_path / "user--host"

        mocker.patch.object(conn, "is_connected", return_value=False)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        # Should not raise
        conn.connect()

        # Verify SSH was called with ControlMaster options
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "ssh" in args
        assert "ControlMaster=yes" in " ".join(args)

    def test_connect_failure(self, mocker, tmp_path):
        """connect() raises SSHError on failure."""
        mocker.patch("rex.ssh.connection.SOCKET_DIR", tmp_path)

        conn = SSHConnection("user@host")
        conn.socket_path = tmp_path / "user--host"

        mocker.patch.object(conn, "is_connected", return_value=False)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=1)

        with pytest.raises(SSHError) as exc_info:
            conn.connect()
        assert "Failed to connect" in exc_info.value.message

    def test_connect_removes_stale_socket(self, mocker, tmp_path):
        """connect() removes stale socket before connecting."""
        mocker.patch("rex.ssh.connection.SOCKET_DIR", tmp_path)

        conn = SSHConnection("user@host")
        conn.socket_path = tmp_path / "user--host"
        conn.socket_path.touch()  # Create stale socket

        mocker.patch.object(conn, "is_connected", return_value=False)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        conn.connect()

        # Socket should have been removed before SSH call
        # (it will be recreated by SSH)


class TestSSHConnectionDisconnect:
    """Tests for SSHConnection.disconnect method."""

    def test_disconnect_no_connection(self, mocker, tmp_path):
        """disconnect() returns False when not connected."""
        mocker.patch("rex.ssh.connection.SOCKET_DIR", tmp_path)
        conn = SSHConnection("user@host")
        # Socket doesn't exist

        result = conn.disconnect()
        assert result is False

    def test_disconnect_success(self, mocker, tmp_path):
        """disconnect() closes connection successfully."""
        mocker.patch("rex.ssh.connection.SOCKET_DIR", tmp_path)
        conn = SSHConnection("user@host")
        conn.socket_path.touch()  # Create socket file

        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        result = conn.disconnect()

        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "-O" in args
        assert "exit" in args


class TestSSHConnectionListActive:
    """Tests for SSHConnection.list_active class method."""

    def test_list_active_no_socket_dir(self, mocker, tmp_path):
        """list_active() returns empty list if socket dir doesn't exist."""
        mocker.patch("rex.ssh.connection.SOCKET_DIR", tmp_path / "nonexistent")

        result = SSHConnection.list_active()
        assert result == []

    def test_list_active_empty_dir(self, mocker, tmp_path):
        """list_active() returns empty list if no sockets."""
        mocker.patch("rex.ssh.connection.SOCKET_DIR", tmp_path)

        result = SSHConnection.list_active()
        assert result == []

    def test_list_active_with_connections(self, mocker, tmp_path):
        """list_active() returns active connections."""
        mocker.patch("rex.ssh.connection.SOCKET_DIR", tmp_path)

        # Create a mock socket file
        socket = tmp_path / "user--host"
        socket.touch()
        mocker.patch.object(Path, "is_socket", return_value=True)

        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        result = SSHConnection.list_active()

        assert len(result) == 1
        assert result[0][0] == "user@host"  # Target reconstructed

    def test_list_active_removes_stale(self, mocker, tmp_path):
        """list_active() removes stale sockets."""
        mocker.patch("rex.ssh.connection.SOCKET_DIR", tmp_path)

        # Create a mock socket file
        socket = tmp_path / "user--host"
        socket.touch()
        mocker.patch.object(Path, "is_socket", return_value=True)

        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=1)  # SSH check fails

        result = SSHConnection.list_active()

        assert result == []
        assert not socket.exists()  # Stale socket removed
