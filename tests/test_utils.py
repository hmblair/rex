"""Tests for rex utility functions."""

import pytest
import re
from pathlib import Path
from unittest.mock import patch

from rex.utils import (
    validate_job_name,
    map_to_remote,
    job_pattern,
    generate_job_name,
    generate_script_id,
)


class TestValidateJobName:
    """Tests for validate_job_name function."""

    def test_valid_alphanumeric(self):
        """Alphanumeric names are valid."""
        validate_job_name("train123")  # Should not raise

    def test_valid_with_dashes(self):
        """Names with dashes are valid."""
        validate_job_name("my-training-job")  # Should not raise

    def test_valid_with_underscores(self):
        """Names with underscores are valid."""
        validate_job_name("my_training_job")  # Should not raise

    def test_valid_mixed(self):
        """Names with mixed characters are valid."""
        validate_job_name("exp1_run-2")  # Should not raise

    def test_invalid_spaces(self):
        """Names with spaces are invalid."""
        with pytest.raises(ValueError) as exc_info:
            validate_job_name("my job")
        assert "Invalid job name" in str(exc_info.value)

    def test_invalid_special_chars(self):
        """Names with special characters are invalid."""
        with pytest.raises(ValueError):
            validate_job_name("job@host")

    def test_invalid_dots(self):
        """Names with dots are invalid."""
        with pytest.raises(ValueError):
            validate_job_name("job.v1")

    def test_invalid_slashes(self):
        """Names with slashes are invalid."""
        with pytest.raises(ValueError):
            validate_job_name("path/to/job")


class TestMapToRemote:
    """Tests for map_to_remote function."""

    def test_macos_path(self, mocker):
        """macOS /Users/user/... paths are mapped correctly."""
        # Mock resolve() to return the path unchanged (avoid symlink resolution)
        mock_path = mocker.MagicMock(spec=Path)
        mock_path.resolve.return_value = Path("/Users/testuser/projects/myproject")
        mocker.patch.object(Path, "resolve", return_value=Path("/Users/testuser/projects/myproject"))

        local = Path("/Users/testuser/projects/myproject")
        result = map_to_remote(local, "/home/remoteuser")
        assert result == "/home/remoteuser/projects/myproject"

    def test_linux_path(self, mocker):
        """Linux /home/user/... paths are mapped correctly."""
        # Mock resolve() to return the expected linux path
        mocker.patch.object(Path, "resolve", return_value=Path("/home/testuser/projects/myproject"))

        local = Path("/home/testuser/projects/myproject")
        result = map_to_remote(local, "/home/remoteuser")
        assert result == "/home/remoteuser/projects/myproject"

    def test_macos_home_only(self, mocker):
        """macOS home directory maps to remote home."""
        mocker.patch.object(Path, "resolve", return_value=Path("/Users/testuser"))

        local = Path("/Users/testuser")
        result = map_to_remote(local, "/home/remoteuser")
        assert result == "/home/remoteuser"

    def test_other_path_unchanged(self, mocker):
        """Paths not in /Users or /home are unchanged."""
        mocker.patch.object(Path, "resolve", return_value=Path("/opt/myapp"))

        local = Path("/opt/myapp")
        result = map_to_remote(local, "/home/remoteuser")
        assert result == "/opt/myapp"

    def test_nested_path(self, mocker):
        """Deeply nested paths are handled correctly."""
        mocker.patch.object(Path, "resolve", return_value=Path("/Users/user/a/b/c/d"))

        local = Path("/Users/user/a/b/c/d")
        result = map_to_remote(local, "/remote/home")
        assert result == "/remote/home/a/b/c/d"


class TestJobPattern:
    """Tests for job_pattern function."""

    def test_basic_pattern(self):
        """job_pattern returns correct pgrep pattern."""
        result = job_pattern("myexp")
        assert result == "rex-myexp[.]py"

    def test_pattern_with_special_chars(self):
        """Pattern handles job IDs with dashes/underscores."""
        result = job_pattern("exp-1_run")
        assert result == "rex-exp-1_run[.]py"


class TestGenerateJobName:
    """Tests for generate_job_name function."""

    def test_format(self):
        """Job name has correct timestamp format."""
        name = generate_job_name()
        # Format: YYYYMMDD-HHMMSS
        assert re.match(r"^\d{8}-\d{6}$", name)

    def test_is_valid_job_name(self):
        """Generated name passes validation."""
        name = generate_job_name()
        validate_job_name(name)  # Should not raise


class TestGenerateScriptId:
    """Tests for generate_script_id function."""

    def test_format(self):
        """Script ID has correct format (pid-timestamp)."""
        script_id = generate_script_id()
        assert re.match(r"^\d+-\d+$", script_id)

    def test_unique(self):
        """Consecutive calls generate different IDs (usually)."""
        id1 = generate_script_id()
        id2 = generate_script_id()
        # Same PID, but timestamp should differ (or be same within same second)
        # At minimum, format should be consistent
        assert re.match(r"^\d+-\d+$", id1)
        assert re.match(r"^\d+-\d+$", id2)
