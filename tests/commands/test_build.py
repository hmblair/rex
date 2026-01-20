"""Tests for build command with ResolvedConfig."""

import pytest

from rex.cli import resolve_config
from rex.config.global_config import HostConfig
from rex.config.resolved import ResolvedConfig
from rex.execution.base import ExecutionContext
from rex.execution.slurm import SlurmOptions
from rex.exceptions import ConfigError


def make_config(
    code_dir: str | None = None,
    modules: list[str] | None = None,
    partition: str | None = None,
) -> ResolvedConfig:
    """Create a ResolvedConfig for testing."""
    return ResolvedConfig(
        name="test-project",
        root=None,
        execution=ExecutionContext(
            python="python3",
            modules=modules or [],
            code_dir=code_dir,
            run_dir=None,
            env={},
        ),
        slurm=SlurmOptions(partition=partition),
    )


class TestBuildWithResolvedConfig:
    """Tests for build command using ResolvedConfig."""

    def test_build_requires_code_dir(self, mocker):
        """Build raises ConfigError when code_dir is None."""
        config = make_config(code_dir=None)
        mock_ssh = mocker.Mock()

        from rex.commands.build import build

        with pytest.raises(ConfigError, match="code_dir not configured"):
            build(mock_ssh, config)

    def test_build_uses_execution_code_dir(self, mocker):
        """Build uses code_dir from config.execution."""
        config = make_config(code_dir="/remote/project")

        mock_ssh = mocker.Mock()
        mock_ssh.exec.return_value = (0, "12345", "")
        mock_ssh._opts = []
        mock_ssh.target = "host"

        mock_run = mocker.patch("subprocess.run")

        from rex.commands.build import build

        build(mock_ssh, config)

        # Verify script is written to code_dir
        call_args = mock_run.call_args
        ssh_cmd = call_args[0][0]
        assert "/remote/project/.rex-build.sh" in ssh_cmd[-1]

    def test_build_uses_execution_modules(self, mocker):
        """Build includes module load from config.execution.modules."""
        config = make_config(
            code_dir="/remote/project",
            modules=["cuda/12.0", "python/3.11"],
        )

        mock_ssh = mocker.Mock()
        mock_ssh.exec.return_value = (0, "12345", "")
        mock_ssh._opts = []
        mock_ssh.target = "host"

        mock_run = mocker.patch("subprocess.run")

        from rex.commands.build import build

        build(mock_ssh, config)

        # Verify script content includes module load
        script_content = mock_run.call_args.kwargs["input"].decode()
        assert "module load cuda/12.0 python/3.11" in script_content

    def test_build_no_modules_when_empty(self, mocker):
        """Build omits module load when modules is empty."""
        config = make_config(code_dir="/remote/project", modules=[])

        mock_ssh = mocker.Mock()
        mock_ssh.exec.return_value = (0, "12345", "")
        mock_ssh._opts = []
        mock_ssh.target = "host"

        mock_run = mocker.patch("subprocess.run")

        from rex.commands.build import build

        build(mock_ssh, config)

        script_content = mock_run.call_args.kwargs["input"].decode()
        assert "module load" not in script_content

    def test_build_uses_slurm_partition(self, mocker):
        """Build passes partition to sbatch."""
        config = make_config(code_dir="/remote/project", partition="gpu-h100")

        mock_ssh = mocker.Mock()
        mock_ssh.exec.return_value = (0, "12345", "")
        mock_ssh._opts = []
        mock_ssh.target = "host"

        mocker.patch("subprocess.run")

        from rex.commands.build import build

        build(mock_ssh, config)

        # Check sbatch command includes partition
        sbatch_call = mock_ssh.exec.call_args[0][0]
        assert "--partition=gpu-h100" in sbatch_call

    def test_build_no_partition_when_none(self, mocker):
        """Build omits partition flag when None."""
        config = make_config(code_dir="/remote/project", partition=None)

        mock_ssh = mocker.Mock()
        mock_ssh.exec.return_value = (0, "12345", "")
        mock_ssh._opts = []
        mock_ssh.target = "host"

        mocker.patch("subprocess.run")

        from rex.commands.build import build

        build(mock_ssh, config)

        sbatch_call = mock_ssh.exec.call_args[0][0]
        assert "--partition" not in sbatch_call

    def test_build_with_clean_flag(self, mocker):
        """Build includes rm -rf .venv when clean=True."""
        config = make_config(code_dir="/remote/project")

        mock_ssh = mocker.Mock()
        mock_ssh.exec.return_value = (0, "12345", "")
        mock_ssh._opts = []
        mock_ssh.target = "host"

        mock_run = mocker.patch("subprocess.run")

        from rex.commands.build import build

        build(mock_ssh, config, clean=True)

        script_content = mock_run.call_args.kwargs["input"].decode()
        assert "rm -rf .venv" in script_content

    def test_build_without_clean_flag(self, mocker):
        """Build omits rm when clean=False."""
        config = make_config(code_dir="/remote/project")

        mock_ssh = mocker.Mock()
        mock_ssh.exec.return_value = (0, "12345", "")
        mock_ssh._opts = []
        mock_ssh.target = "host"

        mock_run = mocker.patch("subprocess.run")

        from rex.commands.build import build

        build(mock_ssh, config, clean=False)

        script_content = mock_run.call_args.kwargs["input"].decode()
        assert "rm -rf .venv" not in script_content

    def test_build_returns_zero_on_success(self, mocker):
        """Build returns 0 when sbatch succeeds."""
        config = make_config(code_dir="/remote/project")

        mock_ssh = mocker.Mock()
        mock_ssh.exec.return_value = (0, "12345", "")
        mock_ssh._opts = []
        mock_ssh.target = "host"

        mocker.patch("subprocess.run")

        from rex.commands.build import build

        result = build(mock_ssh, config)

        assert result == 0
