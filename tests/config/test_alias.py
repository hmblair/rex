"""Tests for alias configuration."""

import pytest
from pathlib import Path

from rex.config.alias import Alias, load_aliases, expand_alias


class TestLoadAliases:
    """Tests for load_aliases function."""

    def test_load_from_nonexistent_file(self, tmp_path):
        """Returns empty dict if file doesn't exist."""
        result = load_aliases(tmp_path / "nonexistent")
        assert result == {}

    def test_load_empty_file(self, tmp_path):
        """Returns empty dict for empty file."""
        config = tmp_path / "rex"
        config.write_text("")
        result = load_aliases(config)
        assert result == {}

    def test_load_simple_alias(self, tmp_path):
        """Loads simple alias without extra args."""
        config = tmp_path / "rex"
        config.write_text("gpu = user@gpu.example.com")
        result = load_aliases(config)

        assert "gpu" in result
        assert result["gpu"].target == "user@gpu.example.com"
        assert result["gpu"].extra_args == []

    def test_load_alias_with_args(self, tmp_path):
        """Loads alias with extra arguments."""
        config = tmp_path / "rex"
        config.write_text("gpu = user@host -p python3.11 -s --partition gpu")
        result = load_aliases(config)

        assert "gpu" in result
        alias = result["gpu"]
        assert alias.target == "user@host"
        assert alias.extra_args == ["-p", "python3.11", "-s", "--partition", "gpu"]

    def test_load_multiple_aliases(self, tmp_path):
        """Loads multiple aliases from file."""
        config = tmp_path / "rex"
        config.write_text(
            """
gpu = user@gpu.cluster
cpu = user@cpu.cluster
dev = user@dev.example.com -p /opt/python
"""
        )
        result = load_aliases(config)

        assert len(result) == 3
        assert "gpu" in result
        assert "cpu" in result
        assert "dev" in result

    def test_ignores_comments(self, tmp_path):
        """Ignores comment lines."""
        config = tmp_path / "rex"
        config.write_text(
            """
# This is a comment
gpu = user@gpu.cluster
# Another comment
"""
        )
        result = load_aliases(config)

        assert len(result) == 1
        assert "gpu" in result

    def test_ignores_blank_lines(self, tmp_path):
        """Ignores blank lines."""
        config = tmp_path / "rex"
        config.write_text(
            """
gpu = user@gpu.cluster

cpu = user@cpu.cluster

"""
        )
        result = load_aliases(config)

        assert len(result) == 2

    def test_handles_quoted_args(self, tmp_path):
        """Handles quoted arguments properly."""
        config = tmp_path / "rex"
        config.write_text('gpu = user@host -p "/path with spaces/python"')
        result = load_aliases(config)

        assert result["gpu"].extra_args == ["-p", "/path with spaces/python"]


class TestExpandAlias:
    """Tests for expand_alias function."""

    def test_expand_existing_alias(self):
        """Expands known alias to target and args."""
        aliases = {
            "gpu": Alias(name="gpu", target="user@gpu.cluster", extra_args=["-s"])
        }
        result = expand_alias("gpu", aliases)

        assert result is not None
        target, args = result
        assert target == "user@gpu.cluster"
        assert args == ["-s"]

    def test_expand_nonexistent_alias(self):
        """Returns None for unknown alias."""
        aliases = {}
        result = expand_alias("unknown", aliases)
        assert result is None

    def test_expand_with_at_symbol(self):
        """Returns None if name contains @ (already a host)."""
        aliases = {
            "gpu": Alias(name="gpu", target="user@gpu.cluster", extra_args=[])
        }
        result = expand_alias("user@host", aliases)
        assert result is None

    def test_expand_empty_args(self):
        """Returns empty args list when alias has no extra args."""
        aliases = {
            "simple": Alias(name="simple", target="host", extra_args=[])
        }
        result = expand_alias("simple", aliases)

        assert result is not None
        target, args = result
        assert args == []
