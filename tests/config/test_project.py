"""Tests for project configuration."""

import pytest
from pathlib import Path

from rex.config.project import ProjectConfig, KNOWN_FIELDS


class TestProjectConfigLoad:
    """Tests for ProjectConfig._load method."""

    def test_load_minimal_config(self, tmp_path):
        """Loads minimal config with just host."""
        config = tmp_path / ".rex.toml"
        config.write_text('host = "user@cluster"')

        result = ProjectConfig._load(config)

        assert result.root == tmp_path
        assert result.host == "user@cluster"
        assert result.code_dir is None

    def test_load_full_config(self, tmp_path):
        """Loads config with all fields."""
        config = tmp_path / ".rex.toml"
        config.write_text(
            """
host = "user@cluster"
code_dir = "~/project"
run_dir = "~/runs"
modules = ["python/3.11", "cuda/12.0"]
cpu_partition = "cpu"
gpu_partition = "gpu"
gres = "gpu:1"
time = "01:00:00"
cpus = 4
mem = "16G"
constraint = "a100"
prefer = "fast"
default_gpu = true
"""
        )

        result = ProjectConfig._load(config)

        assert result.host == "user@cluster"
        assert result.code_dir == "~/project"
        assert result.run_dir == "~/runs"
        assert result.modules == ["python/3.11", "cuda/12.0"]
        assert result.cpu_partition == "cpu"
        assert result.gpu_partition == "gpu"
        assert result.gres == "gpu:1"
        assert result.time == "01:00:00"
        assert result.cpus == 4
        assert result.mem == "16G"
        assert result.constraint == "a100"
        assert result.prefer == "fast"
        assert result.default_gpu is True

    def test_load_empty_modules(self, tmp_path):
        """Loads config with empty modules list."""
        config = tmp_path / ".rex.toml"
        config.write_text('host = "user@cluster"\nmodules = []')

        result = ProjectConfig._load(config)

        assert result.modules == []

    def test_load_defaults(self, tmp_path):
        """Default values are set correctly."""
        config = tmp_path / ".rex.toml"
        config.write_text('host = "user@cluster"')

        result = ProjectConfig._load(config)

        assert result.modules == []
        assert result.default_gpu is False
        assert result.cpus is None
        assert result.env == {}

    def test_load_env_section(self, tmp_path):
        """Loads [env] section with environment variables."""
        config = tmp_path / ".rex.toml"
        config.write_text(
            """
host = "user@cluster"

[env]
MY_VAR = "value"
PYTHONPATH = "/custom/path"
"""
        )

        result = ProjectConfig._load(config)

        assert result.env == {"MY_VAR": "value", "PYTHONPATH": "/custom/path"}

    def test_warns_on_unknown_fields(self, tmp_path, capsys):
        """Warns about unknown fields in config."""
        config = tmp_path / ".rex.toml"
        config.write_text(
            """
host = "user@cluster"
unknown_field = "value"
another_unknown = 42
"""
        )

        ProjectConfig._load(config)

        captured = capsys.readouterr()
        assert "unknown fields" in captured.err
        assert "another_unknown" in captured.err or "unknown_field" in captured.err


class TestProjectConfigFindAndLoad:
    """Tests for ProjectConfig.find_and_load method."""

    def test_find_in_current_dir(self, tmp_path, mocker):
        """Finds config in current directory."""
        config = tmp_path / ".rex.toml"
        config.write_text('host = "user@cluster"')

        result = ProjectConfig.find_and_load(tmp_path)

        assert result is not None
        assert result.host == "user@cluster"

    def test_find_in_parent_dir(self, tmp_path):
        """Finds config in parent directory."""
        config = tmp_path / ".rex.toml"
        config.write_text('host = "user@cluster"')

        subdir = tmp_path / "src" / "module"
        subdir.mkdir(parents=True)

        result = ProjectConfig.find_and_load(subdir)

        assert result is not None
        assert result.host == "user@cluster"
        assert result.root == tmp_path

    def test_returns_none_when_not_found(self, tmp_path):
        """Returns None when no config found."""
        subdir = tmp_path / "empty" / "project"
        subdir.mkdir(parents=True)

        result = ProjectConfig.find_and_load(subdir)

        assert result is None

    def test_uses_cwd_by_default(self, tmp_path, mocker):
        """Uses cwd when start_dir is None."""
        config = tmp_path / ".rex.toml"
        config.write_text('host = "user@cluster"')

        mocker.patch("pathlib.Path.cwd", return_value=tmp_path)

        result = ProjectConfig.find_and_load()

        assert result is not None


class TestKnownFields:
    """Tests for KNOWN_FIELDS constant."""

    def test_contains_essential_fields(self):
        """KNOWN_FIELDS contains essential configuration fields."""
        essential = {"host", "code_dir", "modules", "cpu_partition", "gpu_partition"}
        assert essential.issubset(KNOWN_FIELDS)

    def test_contains_slurm_fields(self):
        """KNOWN_FIELDS contains SLURM-related fields."""
        slurm_fields = {"gres", "time", "cpus", "constraint", "prefer"}
        assert slurm_fields.issubset(KNOWN_FIELDS)
