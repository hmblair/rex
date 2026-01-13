"""Exception hierarchy for rex."""

from __future__ import annotations


class RexError(Exception):
    """Base exception for all rex errors.

    Attributes:
        message: Human-readable error message
        exit_code: Suggested exit code for CLI (default 1)
    """

    def __init__(self, message: str, exit_code: int = 1):
        self.message = message
        self.exit_code = exit_code
        super().__init__(message)


class ConfigError(RexError):
    """Configuration-related errors (.rex.toml, aliases)."""

    pass


class ValidationError(RexError):
    """Input validation errors (job names, paths, arguments)."""

    pass


class SSHError(RexError):
    """SSH connection and execution errors."""

    pass


class TransferError(RexError):
    """File transfer (rsync/scp) errors."""

    pass


class ExecutionError(RexError):
    """Remote command execution errors."""

    pass


class SlurmError(ExecutionError):
    """SLURM-specific errors (sbatch, srun, squeue)."""

    pass
