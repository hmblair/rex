"""Tests for file transfer operations."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from rex.exceptions import TransferError
from rex.ssh.transfer import FileTransfer, PYTHON_EXCLUDES, _shell_quote


class TestShellQuote:
    """Tests for _shell_quote function."""

    def test_simple_string(self):
        """Simple string gets quoted."""
        assert _shell_quote("hello") == "'hello'"

    def test_string_with_spaces(self):
        """String with spaces gets quoted."""
        assert _shell_quote("hello world") == "'hello world'"

    def test_string_with_single_quotes(self):
        """Single quotes are properly escaped."""
        assert _shell_quote("it's") == "'it'\\''s'"

    def test_empty_string(self):
        """Empty string gets quoted."""
        assert _shell_quote("") == "''"


class TestFileTransferPush:
    """Tests for FileTransfer.push method."""

    def test_push_file_not_found(self, mock_ssh_executor, tmp_path):
        """Push raises TransferError if file doesn't exist."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        nonexistent = tmp_path / "nonexistent.txt"

        with pytest.raises(TransferError) as exc_info:
            transfer.push(nonexistent)
        assert "Path not found" in exc_info.value.message

    def test_push_remote_home_failure(self, mock_ssh_executor, tmp_path):
        """Push raises TransferError if remote home lookup fails."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        mock_ssh_executor.exec.return_value = (1, "", "error")

        with pytest.raises(TransferError) as exc_info:
            transfer.push(test_file)
        assert "Failed to get remote home directory" in exc_info.value.message

    def test_push_mkdir_failure(self, mock_ssh_executor, tmp_path):
        """Push raises TransferError if mkdir fails."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # First call succeeds (echo $HOME), second fails (mkdir)
        mock_ssh_executor.exec.side_effect = [
            (0, "/home/user\n", ""),
            (1, "", "mkdir error"),
        ]

        with pytest.raises(TransferError) as exc_info:
            transfer.push(test_file)
        assert "Failed to create remote directory" in exc_info.value.message

    def test_push_rsync_failure(self, mock_ssh_executor, tmp_path, mocker):
        """Push raises TransferError if rsync fails."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        mock_ssh_executor.exec.return_value = (0, "/home/user\n", "")
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=1)

        with pytest.raises(TransferError) as exc_info:
            transfer.push(test_file)
        assert "Push failed" in exc_info.value.message

    def test_push_success(self, mock_ssh_executor, tmp_path, mocker, capsys):
        """Push succeeds with valid file."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        mock_ssh_executor.exec.return_value = (0, "/home/user\n", "")
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        # Should not raise
        transfer.push(test_file)

        # Check rsync was called
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "rsync"
        assert "user@host:" in args[-1]

    def test_push_with_explicit_remote(self, mock_ssh_executor, tmp_path, mocker):
        """Push uses explicit remote path when provided."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        mock_ssh_executor.exec.return_value = (0, "", "")  # mkdir only
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        transfer.push(test_file, remote="/custom/path")

        # Should not call echo $HOME (first exec is mkdir)
        args = mock_run.call_args[0][0]
        assert "user@host:/custom/path" in args[-1]

    def test_push_directory(self, mock_ssh_executor, tmp_path, mocker):
        """Push handles directories with trailing slash."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        test_dir = tmp_path / "mydir"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")

        mock_ssh_executor.exec.return_value = (0, "/home/user\n", "")
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        transfer.push(test_dir)

        args = mock_run.call_args[0][0]
        # Directory should have trailing slash in source
        assert str(test_dir) + "/" in args[-2]


class TestFileTransferPull:
    """Tests for FileTransfer.pull method."""

    def test_pull_rsync_failure(self, mock_ssh_executor, tmp_path, mocker):
        """Pull raises TransferError if rsync fails."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=1)

        with pytest.raises(TransferError) as exc_info:
            transfer.pull("~/remote/file.txt", tmp_path)
        assert "Pull failed" in exc_info.value.message

    def test_pull_success(self, mock_ssh_executor, tmp_path, mocker):
        """Pull succeeds with valid remote path."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        transfer.pull("~/remote/file.txt", tmp_path)

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "rsync"
        assert "user@host:~/remote/file.txt" in args[-2]

    def test_pull_creates_local_dir(self, mock_ssh_executor, tmp_path, mocker):
        """Pull creates local directory if it doesn't exist."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        new_dir = tmp_path / "new" / "nested"
        transfer.pull("~/file.txt", new_dir)

        assert new_dir.exists()

    def test_pull_default_local(self, mock_ssh_executor, mocker):
        """Pull uses cwd when local is None."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        transfer.pull("~/file.txt")

        mock_run.assert_called_once()


class TestFileTransferSync:
    """Tests for FileTransfer.sync method."""

    def test_sync_directory_not_found(self, mock_ssh_executor, tmp_path):
        """Sync raises TransferError if directory doesn't exist."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(TransferError) as exc_info:
            transfer.sync(nonexistent)
        assert "Directory not found" in exc_info.value.message

    def test_sync_not_a_directory(self, mock_ssh_executor, tmp_path):
        """Sync raises TransferError if path is a file, not directory."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        with pytest.raises(TransferError) as exc_info:
            transfer.sync(test_file)
        assert "Directory not found" in exc_info.value.message

    def test_sync_remote_home_failure(self, mock_ssh_executor, tmp_path):
        """Sync raises TransferError if remote home lookup fails."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        test_dir = tmp_path / "project"
        test_dir.mkdir()

        mock_ssh_executor.exec.return_value = (1, "", "error")

        with pytest.raises(TransferError) as exc_info:
            transfer.sync(test_dir)
        assert "Failed to get remote home directory" in exc_info.value.message

    def test_sync_rsync_failure(self, mock_ssh_executor, tmp_path, mocker):
        """Sync raises TransferError if rsync fails."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        test_dir = tmp_path / "project"
        test_dir.mkdir()

        mock_ssh_executor.exec.return_value = (0, "/home/user\n", "")
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=1)

        with pytest.raises(TransferError) as exc_info:
            transfer.sync(test_dir)
        assert "Sync failed" in exc_info.value.message

    def test_sync_success_with_defaults(self, mock_ssh_executor, tmp_path, mocker):
        """Sync uses default Python excludes."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        test_dir = tmp_path / "project"
        test_dir.mkdir()

        mock_ssh_executor.exec.return_value = (0, "/home/user\n", "")
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        transfer.sync(test_dir)

        args = mock_run.call_args[0][0]
        # Check some Python excludes are present
        assert "--exclude" in args
        assert "__pycache__" in args
        assert ".git" in args

    def test_sync_with_custom_excludes(self, mock_ssh_executor, tmp_path, mocker):
        """Sync uses custom excludes when provided."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        test_dir = tmp_path / "project"
        test_dir.mkdir()

        mock_ssh_executor.exec.return_value = (0, "/home/user\n", "")
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        transfer.sync(test_dir, excludes=["*.tmp", "cache"])

        args = mock_run.call_args[0][0]
        assert "*.tmp" in args
        assert "cache" in args
        # Default excludes should NOT be present
        assert "__pycache__" not in args

    def test_sync_delete_option(self, mock_ssh_executor, tmp_path, mocker):
        """Sync includes --delete by default."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        test_dir = tmp_path / "project"
        test_dir.mkdir()

        mock_ssh_executor.exec.return_value = (0, "/home/user\n", "")
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        transfer.sync(test_dir)

        args = mock_run.call_args[0][0]
        assert "--delete" in args

    def test_sync_no_delete(self, mock_ssh_executor, tmp_path, mocker):
        """Sync can disable --delete."""
        transfer = FileTransfer("user@host", mock_ssh_executor)
        test_dir = tmp_path / "project"
        test_dir.mkdir()

        mock_ssh_executor.exec.return_value = (0, "/home/user\n", "")
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0)

        transfer.sync(test_dir, delete=False)

        args = mock_run.call_args[0][0]
        assert "--delete" not in args


class TestPythonExcludes:
    """Tests for PYTHON_EXCLUDES constant."""

    def test_common_excludes_present(self):
        """Common Python excludes are in the list."""
        assert "__pycache__" in PYTHON_EXCLUDES
        assert ".git" in PYTHON_EXCLUDES
        assert ".venv" in PYTHON_EXCLUDES
        assert "*.pyc" in PYTHON_EXCLUDES
        assert ".mypy_cache" in PYTHON_EXCLUDES
        assert ".pytest_cache" in PYTHON_EXCLUDES
