"""Tests for transfer command with ResolvedConfig."""

import pytest
from pathlib import Path

from rex.config.resolved import ResolvedConfig
from rex.execution.base import ExecutionContext
from rex.execution.slurm import SlurmOptions


def make_config(
    name: str | None = None,
    root: Path | None = None,
    code_dir: str | None = None,
    python: str = "python3",
) -> ResolvedConfig:
    """Create a ResolvedConfig for testing."""
    return ResolvedConfig(
        name=name,
        root=root,
        execution=ExecutionContext(
            python=python,
            modules=[],
            code_dir=code_dir,
            run_dir=None,
            env={},
        ),
        slurm=SlurmOptions(),
    )


class TestSyncWithResolvedConfig:
    """Tests for sync command using ResolvedConfig."""

    def test_sync_uses_execution_code_dir(self, tmp_path, mocker):
        """Sync uses code_dir from config.execution."""
        config = make_config(code_dir="/remote/project")

        mock_transfer = mocker.Mock()
        mock_ssh = mocker.Mock()

        from rex.commands.transfer import sync

        sync(mock_transfer, mock_ssh, config, local_path=tmp_path)

        mock_transfer.sync.assert_called_once_with(tmp_path, "/remote/project")

    def test_sync_uses_config_root_as_default_local(self, tmp_path, mocker):
        """Sync uses config.root when local_path not specified."""
        config = make_config(root=tmp_path, code_dir="/remote/project")

        mock_transfer = mocker.Mock()
        mock_ssh = mocker.Mock()

        from rex.commands.transfer import sync

        sync(mock_transfer, mock_ssh, config, local_path=None)

        mock_transfer.sync.assert_called_once_with(tmp_path, "/remote/project")

    def test_sync_falls_back_to_cwd(self, mocker):
        """Sync uses cwd when config.root is None and local_path not specified."""
        config = make_config(root=None, code_dir="/remote/project")

        mock_transfer = mocker.Mock()
        mock_ssh = mocker.Mock()

        from rex.commands.transfer import sync

        sync(mock_transfer, mock_ssh, config, local_path=None)

        # Should use resolved cwd
        call_args = mock_transfer.sync.call_args[0]
        assert call_args[0] == Path.cwd()
        assert call_args[1] == "/remote/project"

    def test_sync_local_path_overrides_root(self, tmp_path, mocker):
        """Explicit local_path overrides config.root."""
        other_path = tmp_path / "other"
        other_path.mkdir()

        config = make_config(root=tmp_path, code_dir="/remote/project")

        mock_transfer = mocker.Mock()
        mock_ssh = mocker.Mock()

        from rex.commands.transfer import sync

        sync(mock_transfer, mock_ssh, config, local_path=other_path)

        mock_transfer.sync.assert_called_once_with(other_path, "/remote/project")

    def test_sync_with_none_code_dir(self, tmp_path, mocker):
        """Sync passes None remote_path when code_dir is None."""
        config = make_config(root=tmp_path, code_dir=None)

        mock_transfer = mocker.Mock()
        mock_ssh = mocker.Mock()
        mock_ssh.exec.return_value = (0, "/home/user", "")

        from rex.commands.transfer import sync

        sync(mock_transfer, mock_ssh, config, local_path=tmp_path, no_install=True)

        mock_transfer.sync.assert_called_once_with(tmp_path, None)

    def test_sync_pip_install_uses_config_python(self, tmp_path, mocker):
        """Pip install uses python from config.execution.python."""
        # Create pyproject.toml to trigger install
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        config = make_config(root=tmp_path, code_dir=None, python="python3.11")

        mock_transfer = mocker.Mock()
        mock_ssh = mocker.Mock()
        mock_ssh.exec.return_value = (0, "/home/user", "")

        from rex.commands.transfer import sync

        sync(mock_transfer, mock_ssh, config, local_path=tmp_path, no_install=False)

        # Find the pip install call
        pip_call = None
        for call in mock_ssh.exec.call_args_list:
            if "pip install" in call[0][0]:
                pip_call = call[0][0]
                break

        assert pip_call is not None
        assert "python3.11 -m pip install" in pip_call

    def test_sync_no_install_skips_pip(self, tmp_path, mocker):
        """no_install=True skips pip install."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        config = make_config(root=tmp_path, code_dir=None)

        mock_transfer = mocker.Mock()
        mock_ssh = mocker.Mock()

        from rex.commands.transfer import sync

        sync(mock_transfer, mock_ssh, config, local_path=tmp_path, no_install=True)

        # Should not call ssh.exec for pip install
        for call in mock_ssh.exec.call_args_list:
            assert "pip install" not in call[0][0]

    def test_sync_returns_zero_on_success(self, tmp_path, mocker):
        """Sync returns 0 on success."""
        config = make_config(root=tmp_path, code_dir="/remote/project")

        mock_transfer = mocker.Mock()
        mock_ssh = mocker.Mock()

        from rex.commands.transfer import sync

        result = sync(mock_transfer, mock_ssh, config, local_path=tmp_path)

        assert result == 0

    def test_sync_returns_one_on_transfer_error(self, tmp_path, mocker):
        """Sync returns 1 when transfer fails."""
        from rex.exceptions import TransferError

        config = make_config(root=tmp_path, code_dir="/remote/project")

        mock_transfer = mocker.Mock()
        mock_transfer.sync.side_effect = TransferError("Connection failed")
        mock_ssh = mocker.Mock()

        from rex.commands.transfer import sync

        result = sync(mock_transfer, mock_ssh, config, local_path=tmp_path)

        assert result == 1
