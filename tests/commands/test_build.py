"""Tests for build command."""

import pytest

from rex.execution.base import ExecutionContext, JobInfo
from rex.exceptions import ConfigError


def make_ctx(
    code_dir: str | None = None,
    modules: list[str] | None = None,
) -> ExecutionContext:
    """Create an ExecutionContext for testing."""
    return ExecutionContext(
        python="python3",
        modules=modules or [],
        code_dir=code_dir,
        run_dir=None,
        env={},
    )


class TestBuild:
    """Tests for build command."""

    def test_build_requires_code_dir(self, mocker):
        """Build raises ConfigError when code_dir is None."""
        ctx = make_ctx(code_dir=None)
        mock_executor = mocker.Mock()

        from rex.commands.build import build

        with pytest.raises(ConfigError, match="code_dir not configured"):
            build(mock_executor, ctx)

    def test_build_calls_exec_detached(self, mocker):
        """Build calls executor.exec_detached with script."""
        ctx = make_ctx(code_dir="/remote/project")
        mock_executor = mocker.Mock()
        mock_executor.exec_detached.return_value = JobInfo(
            job_id="build-abc123",
            log_path="/home/user/.rex/rex-build-abc123.log",
            is_slurm=True,
            slurm_id=12345,
        )

        from rex.commands.build import build

        result = build(mock_executor, ctx)

        mock_executor.exec_detached.assert_called_once()
        call_args = mock_executor.exec_detached.call_args
        assert call_args[0][0] == ctx
        assert "build-" in call_args[0][2]  # job_name

    def test_build_returns_job_info(self, mocker):
        """Build returns JobInfo from exec_detached."""
        ctx = make_ctx(code_dir="/remote/project")
        mock_executor = mocker.Mock()
        expected = JobInfo(
            job_id="build-abc123",
            log_path="/home/user/.rex/rex-build-abc123.log",
            is_slurm=True,
            slurm_id=12345,
        )
        mock_executor.exec_detached.return_value = expected

        from rex.commands.build import build

        result = build(mock_executor, ctx)

        assert result == expected

    def test_build_script_includes_modules(self, mocker):
        """Build script includes module load commands."""
        ctx = make_ctx(
            code_dir="/remote/project",
            modules=["cuda/12.0", "python/3.11"],
        )
        mock_executor = mocker.Mock()
        mock_executor.exec_detached.return_value = JobInfo(
            job_id="build-abc123", log_path="", is_slurm=True, slurm_id=12345
        )

        from rex.commands.build import build

        build(mock_executor, ctx)

        script = mock_executor.exec_detached.call_args[0][1]
        assert "module load cuda/12.0 python/3.11" in script

    def test_build_script_no_modules_when_empty(self, mocker):
        """Build script omits module load when modules is empty."""
        ctx = make_ctx(code_dir="/remote/project", modules=[])
        mock_executor = mocker.Mock()
        mock_executor.exec_detached.return_value = JobInfo(
            job_id="build-abc123", log_path="", is_slurm=True, slurm_id=12345
        )

        from rex.commands.build import build

        build(mock_executor, ctx)

        script = mock_executor.exec_detached.call_args[0][1]
        assert "module load" not in script

    def test_build_script_includes_clean(self, mocker):
        """Build script includes rm -rf .venv when clean=True."""
        ctx = make_ctx(code_dir="/remote/project")
        mock_executor = mocker.Mock()
        mock_executor.exec_detached.return_value = JobInfo(
            job_id="build-abc123", log_path="", is_slurm=True, slurm_id=12345
        )

        from rex.commands.build import build

        build(mock_executor, ctx, clean=True)

        script = mock_executor.exec_detached.call_args[0][1]
        assert "rm -rf .venv" in script

    def test_build_script_no_clean_by_default(self, mocker):
        """Build script omits rm when clean=False."""
        ctx = make_ctx(code_dir="/remote/project")
        mock_executor = mocker.Mock()
        mock_executor.exec_detached.return_value = JobInfo(
            job_id="build-abc123", log_path="", is_slurm=True, slurm_id=12345
        )

        from rex.commands.build import build

        build(mock_executor, ctx, clean=False)

        script = mock_executor.exec_detached.call_args[0][1]
        assert "rm -rf .venv" not in script

    def test_build_script_uses_code_dir(self, mocker):
        """Build script cds to code_dir."""
        ctx = make_ctx(code_dir="/remote/my-project")
        mock_executor = mocker.Mock()
        mock_executor.exec_detached.return_value = JobInfo(
            job_id="build-abc123", log_path="", is_slurm=True, slurm_id=12345
        )

        from rex.commands.build import build

        build(mock_executor, ctx)

        script = mock_executor.exec_detached.call_args[0][1]
        assert "cd /remote/my-project" in script
