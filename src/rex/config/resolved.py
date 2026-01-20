"""Resolved configuration after CLI > project > host merging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rex.execution.base import ExecutionContext
from rex.execution.slurm import SlurmOptions


@dataclass
class ResolvedConfig:
    """Fully resolved config after CLI > project > host merging."""

    # Identity
    name: str | None = None
    root: Path | None = None  # Local project root (for sync)

    # Composed configs
    execution: ExecutionContext | None = None
    slurm: SlurmOptions | None = None  # None when not using SLURM

    def __post_init__(self):
        if self.execution is None:
            self.execution = ExecutionContext()
