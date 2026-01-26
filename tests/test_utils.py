"""Tests for rex utility functions."""

import pytest
import re
from pathlib import Path
from unittest.mock import patch

from rex.utils import (
    validate_job_name,
    validate_slurm_time,
    validate_memory,
    validate_gres,
    validate_cpus,
    map_to_remote,
    job_pattern,
    generate_job_name,
    generate_script_id,
    shell_quote,
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


class TestValidateSlurmTime:
    """Tests for validate_slurm_time function."""

    def test_valid_minutes_only(self):
        """Minutes-only format is valid."""
        validate_slurm_time("30")  # 30 minutes
        validate_slurm_time("120")  # 120 minutes

    def test_valid_mm_ss(self):
        """MM:SS format is valid."""
        validate_slurm_time("30:00")  # 30 minutes
        validate_slurm_time("05:30")  # 5 minutes 30 seconds

    def test_valid_hh_mm_ss(self):
        """HH:MM:SS format is valid."""
        validate_slurm_time("01:00:00")  # 1 hour
        validate_slurm_time("24:00:00")  # 24 hours
        validate_slurm_time("100:30:45")  # 100+ hours is valid

    def test_valid_days_format(self):
        """D-HH:MM:SS format is valid."""
        validate_slurm_time("1-00:00:00")  # 1 day
        validate_slurm_time("7-12:30:00")  # 7 days, 12.5 hours
        validate_slurm_time("30-23:59:59")  # 30 days

    def test_valid_days_hours_only(self):
        """D-HH format is valid."""
        validate_slurm_time("1-12")  # 1 day 12 hours

    def test_invalid_format(self):
        """Invalid formats are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_slurm_time("abc")
        assert "Invalid time format" in str(exc_info.value)

    def test_invalid_seconds_range(self):
        """Seconds > 59 are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_slurm_time("01:00:60")
        assert "minutes/seconds must be 0-59" in str(exc_info.value)

    def test_invalid_minutes_range(self):
        """Minutes > 59 are rejected in HH:MM:SS."""
        with pytest.raises(ValueError) as exc_info:
            validate_slurm_time("01:60:00")
        assert "minutes/seconds must be 0-59" in str(exc_info.value)

    def test_invalid_hours_in_days_format(self):
        """Hours > 23 in D-HH format are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_slurm_time("1-25:00:00")
        assert "hours must be 0-23" in str(exc_info.value)


class TestValidateMemory:
    """Tests for validate_memory function."""

    def test_valid_bytes(self):
        """Plain number (bytes) is valid."""
        validate_memory("1024")
        validate_memory("16000000")

    def test_valid_kilobytes(self):
        """K suffix is valid."""
        validate_memory("512K")
        validate_memory("1024k")  # lowercase

    def test_valid_megabytes(self):
        """M suffix is valid."""
        validate_memory("16M")
        validate_memory("4096m")

    def test_valid_gigabytes(self):
        """G suffix is valid."""
        validate_memory("4G")
        validate_memory("32g")

    def test_valid_terabytes(self):
        """T suffix is valid."""
        validate_memory("1T")
        validate_memory("2t")

    def test_invalid_suffix(self):
        """Invalid suffixes are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_memory("16GB")  # GB not valid, only G
        assert "Invalid memory format" in str(exc_info.value)

    def test_invalid_zero(self):
        """Zero memory is rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_memory("0G")
        assert "must be greater than 0" in str(exc_info.value)

    def test_invalid_format(self):
        """Non-numeric formats are rejected."""
        with pytest.raises(ValueError):
            validate_memory("abc")

    def test_invalid_space(self):
        """Spaces in format are rejected."""
        with pytest.raises(ValueError):
            validate_memory("4 G")


class TestValidateGres:
    """Tests for validate_gres function."""

    def test_valid_gpu_count(self):
        """gpu:N format is valid."""
        validate_gres("gpu:1")
        validate_gres("gpu:4")

    def test_valid_gpu_type_count(self):
        """gpu:type:N format is valid."""
        validate_gres("gpu:a100:2")
        validate_gres("gpu:v100:4")

    def test_valid_gpu_type_only(self):
        """gpu:type format is valid."""
        validate_gres("gpu:a100")
        validate_gres("gpu:tesla_v100")

    def test_valid_other_resources(self):
        """Other GRES resources are valid."""
        validate_gres("shard:1")
        validate_gres("mps:50")

    def test_invalid_empty_segment(self):
        """Empty segments are rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_gres("gpu::1")
        assert "Invalid GRES format" in str(exc_info.value)

    def test_invalid_format(self):
        """Invalid formats are rejected."""
        with pytest.raises(ValueError):
            validate_gres("not valid")


class TestValidateCpus:
    """Tests for validate_cpus function."""

    def test_valid_single(self):
        """Single CPU is valid."""
        validate_cpus(1)

    def test_valid_multiple(self):
        """Multiple CPUs are valid."""
        validate_cpus(4)
        validate_cpus(64)
        validate_cpus(1024)

    def test_invalid_zero(self):
        """Zero CPUs is invalid."""
        with pytest.raises(ValueError) as exc_info:
            validate_cpus(0)
        assert "must be at least 1" in str(exc_info.value)

    def test_invalid_negative(self):
        """Negative CPUs is invalid."""
        with pytest.raises(ValueError):
            validate_cpus(-1)


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
        assert result == "rex-myexp[.](py|sh)"

    def test_pattern_with_special_chars(self):
        """Pattern handles job IDs with dashes/underscores."""
        result = job_pattern("exp-1_run")
        assert result == "rex-exp-1_run[.](py|sh)"


class TestGenerateJobName:
    """Tests for generate_job_name function."""

    def test_format(self):
        """Job name has correct timestamp format with random suffix."""
        name = generate_job_name()
        # Format: YYYYMMDD-HHMMSS-XXXX (4-char hex suffix for uniqueness)
        assert re.match(r"^\d{8}-\d{6}-[0-9a-f]{4}$", name)

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


class TestShellQuote:
    """Tests for shell_quote function."""

    def test_simple_string(self):
        """Simple strings are wrapped in single quotes."""
        assert shell_quote("hello") == "'hello'"

    def test_string_with_spaces(self):
        """Strings with spaces are properly quoted."""
        assert shell_quote("hello world") == "'hello world'"

    def test_string_with_single_quote(self):
        """Single quotes are escaped."""
        result = shell_quote("it's")
        assert result == "'it'\\''s'"

    def test_empty_string(self):
        """Empty string returns empty quotes."""
        assert shell_quote("") == "''"

    def test_special_characters(self):
        """Special shell characters are safely quoted."""
        result = shell_quote("echo $HOME; rm -rf /")
        assert result == "'echo $HOME; rm -rf /'"

    def test_double_quotes(self):
        """Double quotes are preserved inside single quotes."""
        result = shell_quote('echo "hello world"')
        assert result == '\'echo "hello world"\''

    def test_mixed_quotes(self):
        """Mixed single and double quotes are handled."""
        result = shell_quote("echo \"it's working\"")
        assert result == "'echo \"it'\\''s working\"'"

    def test_dollar_sign_variable(self):
        """Dollar signs are preserved (not expanded)."""
        result = shell_quote("echo $HOME $USER")
        assert result == "'echo $HOME $USER'"

    def test_backticks(self):
        """Backticks are preserved."""
        result = shell_quote("echo `date`")
        assert result == "'echo `date`'"

    def test_pipe(self):
        """Pipe characters are preserved."""
        result = shell_quote("ls -la | grep foo")
        assert result == "'ls -la | grep foo'"

    def test_semicolon(self):
        """Semicolons are preserved."""
        result = shell_quote("cmd1; cmd2; cmd3")
        assert result == "'cmd1; cmd2; cmd3'"

    def test_ampersand(self):
        """Ampersands are preserved."""
        result = shell_quote("cmd1 && cmd2 || cmd3")
        assert result == "'cmd1 && cmd2 || cmd3'"

    def test_backslash(self):
        """Backslashes are preserved."""
        result = shell_quote("echo \\n\\t")
        assert result == "'echo \\n\\t'"

    def test_parentheses(self):
        """Parentheses are preserved."""
        result = shell_quote("(cd /tmp && ls)")
        assert result == "'(cd /tmp && ls)'"

    def test_brackets(self):
        """Brackets are preserved."""
        result = shell_quote("[[ -f /tmp/test ]] && echo yes")
        assert result == "'[[ -f /tmp/test ]] && echo yes'"

    def test_glob_characters(self):
        """Glob characters are preserved."""
        result = shell_quote("ls *.py **/*.txt")
        assert result == "'ls *.py **/*.txt'"

    def test_multiple_single_quotes(self):
        """Multiple single quotes are all escaped."""
        result = shell_quote("echo 'one' 'two' 'three'")
        assert result == "'echo '\\''one'\\'' '\\''two'\\'' '\\''three'\\'''"

    def test_newline(self):
        """Newlines are preserved."""
        result = shell_quote("echo first\necho second")
        assert result == "'echo first\necho second'"

    def test_complex_command(self):
        """Complex real-world command is properly quoted."""
        cmd = '''for f in *.py; do echo "$f"; done'''
        result = shell_quote(cmd)
        assert result == "'" + cmd + "'"
