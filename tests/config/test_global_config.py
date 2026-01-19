"""Tests for global configuration."""

import pytest
from pathlib import Path

from rex.config.global_config import GlobalConfig, HostConfig, KNOWN_HOST_FIELDS
from rex.exceptions import ConfigError


class TestGlobalConfigLoad:
    """Tests for GlobalConfig.load method."""

    def test_load_from_nonexistent_file(self, tmp_path):
        """Returns empty config if file doesn't exist."""
        result = GlobalConfig.load(tmp_path / "nonexistent.toml")
        assert result.aliases == {}
        assert result.hosts == {}

    def test_load_empty_file(self, tmp_path):
        """Returns empty config for empty file."""
        config = tmp_path / "config.toml"
        config.write_text("")
        result = GlobalConfig.load(config)
        assert result.aliases == {}
        assert result.hosts == {}

    def test_load_aliases_only(self, tmp_path):
        """Loads aliases without host configs."""
        config = tmp_path / "config.toml"
        config.write_text(
            """
[aliases]
sherlock = "hmblair@login.sherlock.stanford.edu"
imp = "hmblair@imp"
"""
        )
        result = GlobalConfig.load(config)

        assert len(result.aliases) == 2
        assert result.aliases["sherlock"] == "hmblair@login.sherlock.stanford.edu"
        assert result.aliases["imp"] == "hmblair@imp"
        assert result.hosts == {}

    def test_load_host_config(self, tmp_path):
        """Loads host configuration."""
        config = tmp_path / "config.toml"
        config.write_text(
            """
[aliases]
sherlock = "hmblair@login.sherlock.stanford.edu"

[hosts.sherlock]
code_dir = "/home/groups/rhiju/hmblair"
run_dir = "/scratch/users/hmblair"
modules = ["python/3.12", "cuda/12.4.0"]
cpu_partition = "biochem"
gpu_partition = "rhiju"
gres = "gpu:1"
time = "8:00:00"
prefer = "GPU_SKU:H100_SXM5"
"""
        )
        result = GlobalConfig.load(config)

        assert "sherlock" in result.hosts
        hc = result.hosts["sherlock"]
        assert hc.code_dir == "/home/groups/rhiju/hmblair"
        assert hc.run_dir == "/scratch/users/hmblair"
        assert hc.modules == ["python/3.12", "cuda/12.4.0"]
        assert hc.cpu_partition == "biochem"
        assert hc.gpu_partition == "rhiju"
        assert hc.gres == "gpu:1"
        assert hc.time == "8:00:00"
        assert hc.prefer == "GPU_SKU:H100_SXM5"

    def test_load_host_env(self, tmp_path):
        """Loads host environment variables."""
        config = tmp_path / "config.toml"
        config.write_text(
            """
[hosts.sherlock]
code_dir = "/home/user"

[hosts.sherlock.env]
MY_VAR = "value"
OTHER_VAR = "other"
"""
        )
        result = GlobalConfig.load(config)

        hc = result.hosts["sherlock"]
        assert hc.env == {"MY_VAR": "value", "OTHER_VAR": "other"}

    def test_load_multiple_hosts(self, tmp_path):
        """Loads multiple host configurations."""
        config = tmp_path / "config.toml"
        config.write_text(
            """
[aliases]
sherlock = "user@sherlock"
imp = "user@imp"

[hosts.sherlock]
code_dir = "/home/sherlock"
gpu_partition = "gpu"

[hosts.imp]
code_dir = "/home/imp"
"""
        )
        result = GlobalConfig.load(config)

        assert len(result.hosts) == 2
        assert result.hosts["sherlock"].code_dir == "/home/sherlock"
        assert result.hosts["sherlock"].gpu_partition == "gpu"
        assert result.hosts["imp"].code_dir == "/home/imp"
        assert result.hosts["imp"].gpu_partition is None

    def test_load_default_gpu(self, tmp_path):
        """Loads default_gpu setting."""
        config = tmp_path / "config.toml"
        config.write_text(
            """
[hosts.sherlock]
default_gpu = true
gpu_partition = "gpu"
"""
        )
        result = GlobalConfig.load(config)

        assert result.hosts["sherlock"].default_gpu is True

    def test_validates_time_format(self, tmp_path):
        """Validates SLURM time format."""
        config = tmp_path / "config.toml"
        config.write_text(
            """
[hosts.sherlock]
time = "invalid"
"""
        )

        with pytest.raises(ConfigError) as exc:
            GlobalConfig.load(config)
        assert "time" in str(exc.value).lower()

    def test_validates_memory_format(self, tmp_path):
        """Validates SLURM memory format."""
        config = tmp_path / "config.toml"
        config.write_text(
            """
[hosts.sherlock]
mem = "invalid"
"""
        )

        with pytest.raises(ConfigError) as exc:
            GlobalConfig.load(config)
        assert "memory" in str(exc.value).lower()

    def test_validates_gres_format(self, tmp_path):
        """Validates SLURM GRES format."""
        config = tmp_path / "config.toml"
        config.write_text(
            """
[hosts.sherlock]
gres = "invalid format!"
"""
        )

        with pytest.raises(ConfigError) as exc:
            GlobalConfig.load(config)
        assert "gres" in str(exc.value).lower()

    def test_validates_cpus(self, tmp_path):
        """Validates SLURM CPU count."""
        config = tmp_path / "config.toml"
        config.write_text(
            """
[hosts.sherlock]
cpus = 0
"""
        )

        with pytest.raises(ConfigError) as exc:
            GlobalConfig.load(config)
        assert "cpu" in str(exc.value).lower()

    def test_warns_on_unknown_fields(self, tmp_path, capsys):
        """Warns about unknown fields in host config."""
        config = tmp_path / "config.toml"
        config.write_text(
            """
[hosts.sherlock]
code_dir = "/home/user"
unknown_field = "value"
"""
        )

        GlobalConfig.load(config)

        captured = capsys.readouterr()
        assert "unknown fields" in captured.err
        assert "unknown_field" in captured.err


class TestExpandAlias:
    """Tests for GlobalConfig.expand_alias method."""

    def test_expand_existing_alias(self, tmp_path):
        """Expands known alias to target."""
        config = tmp_path / "config.toml"
        config.write_text(
            """
[aliases]
sherlock = "hmblair@login.sherlock.stanford.edu"
"""
        )
        gc = GlobalConfig.load(config)

        result = gc.expand_alias("sherlock")
        assert result == "hmblair@login.sherlock.stanford.edu"

    def test_expand_nonexistent_alias(self, tmp_path):
        """Returns None for unknown alias."""
        config = tmp_path / "config.toml"
        config.write_text("")
        gc = GlobalConfig.load(config)

        result = gc.expand_alias("unknown")
        assert result is None

    def test_expand_with_at_symbol(self, tmp_path):
        """Returns None if name contains @ (already a host)."""
        config = tmp_path / "config.toml"
        config.write_text(
            """
[aliases]
sherlock = "hmblair@sherlock"
"""
        )
        gc = GlobalConfig.load(config)

        result = gc.expand_alias("user@host")
        assert result is None


class TestGetHostConfig:
    """Tests for GlobalConfig.get_host_config method."""

    def test_get_host_config_by_alias(self, tmp_path):
        """Gets host config by alias name."""
        config = tmp_path / "config.toml"
        config.write_text(
            """
[aliases]
sherlock = "hmblair@sherlock"

[hosts.sherlock]
code_dir = "/home/user"
"""
        )
        gc = GlobalConfig.load(config)

        result = gc.get_host_config("sherlock")
        assert result is not None
        assert result.code_dir == "/home/user"

    def test_get_host_config_not_found(self, tmp_path):
        """Returns None if no config for alias."""
        config = tmp_path / "config.toml"
        config.write_text(
            """
[aliases]
sherlock = "hmblair@sherlock"
"""
        )
        gc = GlobalConfig.load(config)

        result = gc.get_host_config("sherlock")
        assert result is None

    def test_get_host_config_unknown_alias(self, tmp_path):
        """Returns None for unknown alias."""
        config = tmp_path / "config.toml"
        config.write_text("")
        gc = GlobalConfig.load(config)

        result = gc.get_host_config("unknown")
        assert result is None


class TestHostConfigDefaults:
    """Tests for HostConfig default values."""

    def test_all_defaults(self):
        """All fields have sensible defaults."""
        hc = HostConfig()

        assert hc.code_dir is None
        assert hc.run_dir is None
        assert hc.modules == []
        assert hc.cpu_partition is None
        assert hc.gpu_partition is None
        assert hc.gres is None
        assert hc.time is None
        assert hc.cpus is None
        assert hc.mem is None
        assert hc.constraint is None
        assert hc.prefer is None
        assert hc.default_gpu is False
        assert hc.default_slurm is False
        assert hc.env == {}


class TestKnownHostFields:
    """Tests for KNOWN_HOST_FIELDS constant."""

    def test_contains_path_fields(self):
        """Contains directory path fields."""
        assert "code_dir" in KNOWN_HOST_FIELDS
        assert "run_dir" in KNOWN_HOST_FIELDS

    def test_contains_slurm_fields(self):
        """Contains SLURM-related fields."""
        slurm_fields = {
            "modules", "cpu_partition", "gpu_partition",
            "gres", "time", "cpus", "mem", "constraint", "prefer"
        }
        assert slurm_fields.issubset(KNOWN_HOST_FIELDS)

    def test_contains_env(self):
        """Contains env field."""
        assert "env" in KNOWN_HOST_FIELDS
