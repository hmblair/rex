"""Project configuration (.rex.toml parsing)."""

from __future__ import annotations

import tomli
from dataclasses import dataclass, field
from pathlib import Path

from rex.exceptions import ConfigError
from rex.output import warn
from rex.utils import validate_slurm_time, validate_memory, validate_gres, validate_cpus

KNOWN_FIELDS = {
    "name",
    "code_dir",
    "run_dir",
    "modules",
    "cpu_partition",
    "gpu_partition",
    "gres",
    "time",
    "cpus",
    "mem",
    "constraint",
    "prefer",
    "default_gpu",
    "env",
}


@dataclass
class ProjectConfig:
    """Project configuration from .rex.toml."""

    root: Path
    name: str
    code_dir: str | None = None
    run_dir: str | None = None
    modules: list[str] | None = None
    cpu_partition: str | None = None
    gpu_partition: str | None = None
    gres: str | None = None
    time: str | None = None
    cpus: int | None = None
    mem: str | None = None
    constraint: str | None = None
    prefer: str | None = None
    default_gpu: bool | None = None
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def find_and_load(cls, start_dir: Path | None = None) -> "ProjectConfig | None":
        """Walk up from start_dir to find .rex.toml and load it.

        Returns None if no config file is found.
        """
        if start_dir is None:
            start_dir = Path.cwd()

        current = start_dir.resolve()
        while current != current.parent:
            config_path = current / ".rex.toml"
            if config_path.exists():
                return cls._load(config_path)
            current = current.parent

        return None

    @classmethod
    def _load(cls, path: Path) -> "ProjectConfig":
        """Load config from a specific path."""
        with open(path, "rb") as f:
            data = tomli.load(f)

        # Warn about unknown fields
        unknown = set(data.keys()) - KNOWN_FIELDS
        if unknown:
            warn(f".rex.toml: unknown fields: {', '.join(sorted(unknown))}")

        # Require name field
        if "name" not in data:
            raise ConfigError(".rex.toml: missing required field 'name'")

        # Validate SLURM options
        try:
            if data.get("time"):
                validate_slurm_time(data["time"])
            if data.get("mem"):
                validate_memory(data["mem"])
            if data.get("gres"):
                validate_gres(data["gres"])
            if data.get("cpus") is not None:
                validate_cpus(data["cpus"])
        except ValueError as e:
            raise ConfigError(f".rex.toml: {e}")

        return cls(
            root=path.parent,
            name=data["name"],
            code_dir=data.get("code_dir"),
            run_dir=data.get("run_dir"),
            modules=data.get("modules"),
            cpu_partition=data.get("cpu_partition"),
            gpu_partition=data.get("gpu_partition"),
            gres=data.get("gres"),
            time=data.get("time"),
            cpus=data.get("cpus"),
            mem=data.get("mem"),
            constraint=data.get("constraint"),
            prefer=data.get("prefer"),
            default_gpu=data.get("default_gpu"),
            env=data.get("env", {}),
        )
