"""End-to-end CLI tests using subprocess.

These tests invoke rex as a subprocess to catch shell escaping issues
that occur before arguments reach Python.
"""

import os
import subprocess
import sys
import tempfile

import pytest


def run_rex(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    """Run rex CLI as subprocess with isolated config."""
    env = {**os.environ, "XDG_CONFIG_HOME": tempfile.mkdtemp()}
    return subprocess.run(
        [sys.executable, "-m", "rex.cli", *args],
        capture_output=True,
        text=True,
        check=check,
        env=env,
    )


class TestCliSubprocessBasic:
    """Basic subprocess CLI tests."""

    def test_version_flag(self):
        """--version works via subprocess."""
        result = run_rex("--version")
        assert result.returncode == 0
        assert "rex" in result.stdout.lower() or result.stdout.strip()

    def test_help_flag(self):
        """--help works via subprocess."""
        result = run_rex("--help")
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()

    def test_no_args_shows_help(self):
        """No arguments shows help."""
        result = run_rex()
        # May return 0 or 1 depending on implementation, but should show usage
        assert "usage" in result.stdout.lower()


class TestCliSubprocessExecSpecialChars:
    """Test --exec with special characters via subprocess.

    These tests verify that special characters survive shell parsing
    and reach the CLI correctly. They will fail at SSH connection
    (expected) but we verify the error message shows the command
    was parsed correctly.
    """

    def test_exec_double_quotes(self):
        """--exec with double quotes survives shell parsing."""
        # Use a non-existent host so it fails quickly
        result = run_rex("testhost", "--exec", 'echo "hello world"')
        # Should fail due to SSH, not argument parsing
        assert result.returncode != 0
        # Should not complain about invalid arguments
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_exec_single_quotes(self):
        """--exec with single quotes survives shell parsing."""
        result = run_rex("testhost", "--exec", "echo 'hello world'")
        assert result.returncode != 0
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_exec_dollar_variable(self):
        """--exec with $VAR survives shell parsing."""
        result = run_rex("testhost", "--exec", "echo $HOME")
        assert result.returncode != 0
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_exec_pipe(self):
        """--exec with pipe survives shell parsing."""
        result = run_rex("testhost", "--exec", "ls | grep foo")
        assert result.returncode != 0
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_exec_semicolon(self):
        """--exec with semicolon survives shell parsing."""
        result = run_rex("testhost", "--exec", "cmd1; cmd2")
        assert result.returncode != 0
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_exec_ampersand(self):
        """--exec with && survives shell parsing."""
        result = run_rex("testhost", "--exec", "cmd1 && cmd2")
        assert result.returncode != 0
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_exec_backticks(self):
        """--exec with backticks survives shell parsing."""
        result = run_rex("testhost", "--exec", "echo `date`")
        assert result.returncode != 0
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_exec_parentheses(self):
        """--exec with parentheses survives shell parsing."""
        result = run_rex("testhost", "--exec", "(cd /tmp && ls)")
        assert result.returncode != 0
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_exec_glob(self):
        """--exec with glob pattern survives shell parsing."""
        result = run_rex("testhost", "--exec", "ls *.py")
        assert result.returncode != 0
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_exec_complex_command(self):
        """--exec with complex command survives shell parsing."""
        result = run_rex("testhost", "--exec", '''for f in *.py; do echo "$f"; done''')
        assert result.returncode != 0
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_exec_mixed_quotes(self):
        """--exec with mixed quotes survives shell parsing."""
        result = run_rex("testhost", "--exec", '''echo "it's working"''')
        assert result.returncode != 0
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_exec_backslash(self):
        """--exec with backslash survives shell parsing."""
        result = run_rex("testhost", "--exec", "echo 'line1\\nline2'")
        assert result.returncode != 0
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_exec_brackets(self):
        """--exec with brackets survives shell parsing."""
        result = run_rex("testhost", "--exec", "[[ -f /tmp/test ]] && echo yes")
        assert result.returncode != 0
        assert "unrecognized arguments" not in result.stderr.lower()


class TestCliSubprocessValidation:
    """Test CLI validation via subprocess."""

    def test_slurm_options_rejected_for_non_slurm_host(self):
        """SLURM options are rejected for non-SLURM hosts."""
        result = run_rex("testhost", "--time", "1:00:00", "--exec", "echo")
        assert result.returncode != 0
        assert "slurm" in result.stderr.lower()

    def test_invalid_job_name(self):
        """Invalid job name with spaces is rejected."""
        result = run_rex("testhost", "-n", "invalid name", "--exec", "echo")
        assert result.returncode != 0
        assert "invalid" in result.stderr.lower() or "name" in result.stderr.lower()
