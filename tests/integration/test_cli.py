"""Integration tests for CLI."""

import pytest
from unittest.mock import MagicMock, patch

from rex.cli import build_parser, main
from rex.exceptions import ValidationError, ConfigError


class TestBuildParser:
    """Tests for build_parser function."""

    def test_parser_creation(self):
        """Parser is created successfully."""
        parser = build_parser()
        assert parser is not None
        assert parser.prog == "rex"

    def test_version_flag(self, capsys):
        """--version flag works."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_help_flag(self, capsys):
        """--help flag works."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_target_parsing(self):
        """Target is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["user@host"])
        assert args.target == "user@host"

    def test_slurm_flag(self):
        """-s/--slurm flag is parsed."""
        parser = build_parser()
        args = parser.parse_args(["user@host", "-s"])
        assert args.slurm is True

    def test_detach_flag(self):
        """-d/--detach flag is parsed."""
        parser = build_parser()
        args = parser.parse_args(["user@host", "-d"])
        assert args.detach is True

    def test_name_option(self):
        """-n/--name option is parsed."""
        parser = build_parser()
        args = parser.parse_args(["user@host", "-n", "myexp"])
        assert args.name == "myexp"

    def test_python_option(self):
        """-p/--python option is parsed."""
        parser = build_parser()
        args = parser.parse_args(["user@host", "-p", "/opt/python3"])
        assert args.python == "/opt/python3"

    def test_modules_option(self):
        """-m/--module option accumulates."""
        parser = build_parser()
        args = parser.parse_args(["user@host", "-m", "python/3.11", "-m", "cuda/12"])
        assert args.modules == ["python/3.11", "cuda/12"]

    def test_slurm_options(self):
        """SLURM options are parsed."""
        parser = build_parser()
        args = parser.parse_args([
            "user@host",
            "--partition", "gpu",
            "--gres", "gpu:1",
            "--time", "01:00:00",
            "--cpus", "4",
            "--mem", "16G",
            "--constraint", "a100",
            "--prefer", "fast",
        ])
        assert args.partition == "gpu"
        assert args.gres == "gpu:1"
        assert args.time == "01:00:00"
        assert args.cpus == 4
        assert args.mem == "16G"
        assert args.constraint == "a100"
        assert args.prefer == "fast"

    def test_command_flags(self):
        """Command flags are parsed."""
        parser = build_parser()

        args = parser.parse_args(["user@host", "--jobs"])
        assert args.jobs is True

        args = parser.parse_args(["user@host", "--connect"])
        assert args.connect is True

        args = parser.parse_args(["user@host", "--gpu-info"])
        assert args.gpu_info is True

    def test_script_and_args(self):
        """Script and positional arguments are parsed."""
        parser = build_parser()
        # Note: Script args with -- prefixes need special handling
        # Test with positional args only
        args = parser.parse_intermixed_args(["user@host", "train.py", "arg1", "arg2"])
        assert args.target == "user@host"
        assert args.script == "train.py"
        assert args.script_args == ["arg1", "arg2"]


class TestMainExceptionHandling:
    """Tests for main() exception handling."""

    def test_handles_validation_error(self, mocker, capsys):
        """main() catches ValidationError and returns exit code."""
        mocker.patch("rex.cli._main", side_effect=ValidationError("test error"))

        result = main(["user@host"])

        assert result == 1
        captured = capsys.readouterr()
        assert "error: test error" in captured.err

    def test_handles_config_error(self, mocker, capsys):
        """main() catches ConfigError and returns exit code."""
        mocker.patch("rex.cli._main", side_effect=ConfigError("config issue"))

        result = main(["user@host"])

        assert result == 1
        captured = capsys.readouterr()
        assert "error: config issue" in captured.err

    def test_handles_keyboard_interrupt(self, mocker):
        """main() catches KeyboardInterrupt and returns 130."""
        mocker.patch("rex.cli._main", side_effect=KeyboardInterrupt)

        result = main(["user@host"])

        assert result == 130

    def test_custom_exit_code(self, mocker, capsys):
        """main() uses exception's exit_code."""
        mocker.patch("rex.cli._main", side_effect=ValidationError("error", exit_code=2))

        result = main(["user@host"])

        assert result == 2


class TestMainNoTarget:
    """Tests for main() without target."""

    def test_no_target_shows_help(self, mocker, capsys):
        """main() shows help when no target provided."""
        from rex.config.global_config import GlobalConfig
        mocker.patch.object(GlobalConfig, "load", return_value=GlobalConfig(aliases={}, hosts={}))
        mocker.patch("rex.config.project.ProjectConfig.find_and_load", return_value=None)

        result = main([])

        assert result == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "rex" in captured.out

    def test_connection_without_target_lists_all(self, mocker):
        """--connection without target lists all connections."""
        mock_status = mocker.patch("rex.commands.connection.connection_status", return_value=0)

        result = main(["--connection"])

        mock_status.assert_called_once_with(None)
        assert result == 0


class TestMainJobNameValidation:
    """Tests for job name validation in main()."""

    def test_invalid_job_name(self, mocker, capsys):
        """Invalid job name raises ValidationError."""
        from rex.config.global_config import GlobalConfig
        mocker.patch.object(GlobalConfig, "load", return_value=GlobalConfig(aliases={}, hosts={}))
        mocker.patch("rex.config.project.ProjectConfig.find_and_load", return_value=None)

        result = main(["user@host", "-n", "invalid name!", "script.py"])

        assert result == 1
        captured = capsys.readouterr()
        assert "error:" in captured.err

    def test_valid_job_name(self, mocker):
        """Valid job name is accepted."""
        from rex.config.global_config import GlobalConfig
        mocker.patch.object(GlobalConfig, "load", return_value=GlobalConfig(aliases={}, hosts={}))
        mocker.patch("rex.config.project.ProjectConfig.find_and_load", return_value=None)
        mock_ssh = mocker.MagicMock()
        mocker.patch("rex.cli.SSHExecutor", return_value=mock_ssh)
        mock_run = mocker.patch("rex.commands.run.run_python", return_value=0)

        result = main(["user@host", "-n", "valid-name_123", "script.py"])

        # Should get past validation (check_connection called before running)
        mock_ssh.check_connection.assert_called_once()
        mock_run.assert_called_once()
