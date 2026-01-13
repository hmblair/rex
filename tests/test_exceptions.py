"""Tests for rex exception hierarchy."""

import pytest

from rex.exceptions import (
    RexError,
    ConfigError,
    ValidationError,
    SSHError,
    TransferError,
    ExecutionError,
    SlurmError,
)


class TestRexError:
    """Tests for base RexError class."""

    def test_message_attribute(self):
        """RexError stores message as attribute."""
        err = RexError("test message")
        assert err.message == "test message"

    def test_default_exit_code(self):
        """RexError has default exit code of 1."""
        err = RexError("test")
        assert err.exit_code == 1

    def test_custom_exit_code(self):
        """RexError accepts custom exit code."""
        err = RexError("test", exit_code=2)
        assert err.exit_code == 2

    def test_str_representation(self):
        """RexError string representation is the message."""
        err = RexError("test message")
        assert str(err) == "test message"

    def test_can_be_raised_and_caught(self):
        """RexError can be raised and caught."""
        with pytest.raises(RexError) as exc_info:
            raise RexError("test error")
        assert exc_info.value.message == "test error"


class TestConfigError:
    """Tests for ConfigError class."""

    def test_inherits_from_rex_error(self):
        """ConfigError inherits from RexError."""
        err = ConfigError("config issue")
        assert isinstance(err, RexError)

    def test_can_catch_as_rex_error(self):
        """ConfigError can be caught as RexError."""
        with pytest.raises(RexError):
            raise ConfigError("config issue")


class TestValidationError:
    """Tests for ValidationError class."""

    def test_inherits_from_rex_error(self):
        """ValidationError inherits from RexError."""
        err = ValidationError("invalid input")
        assert isinstance(err, RexError)


class TestSSHError:
    """Tests for SSHError class."""

    def test_inherits_from_rex_error(self):
        """SSHError inherits from RexError."""
        err = SSHError("connection failed")
        assert isinstance(err, RexError)


class TestTransferError:
    """Tests for TransferError class."""

    def test_inherits_from_rex_error(self):
        """TransferError inherits from RexError."""
        err = TransferError("rsync failed")
        assert isinstance(err, RexError)


class TestExecutionError:
    """Tests for ExecutionError class."""

    def test_inherits_from_rex_error(self):
        """ExecutionError inherits from RexError."""
        err = ExecutionError("command failed")
        assert isinstance(err, RexError)


class TestSlurmError:
    """Tests for SlurmError class."""

    def test_inherits_from_execution_error(self):
        """SlurmError inherits from ExecutionError."""
        err = SlurmError("sbatch failed")
        assert isinstance(err, ExecutionError)

    def test_inherits_from_rex_error(self):
        """SlurmError also inherits from RexError (via ExecutionError)."""
        err = SlurmError("sbatch failed")
        assert isinstance(err, RexError)

    def test_can_catch_as_execution_error(self):
        """SlurmError can be caught as ExecutionError."""
        with pytest.raises(ExecutionError):
            raise SlurmError("sbatch failed")
