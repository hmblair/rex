"""Project configuration (.rex.toml parsing)."""

from __future__ import annotations

import tomli
from dataclasses import dataclass, field
from pathlib import Path

from rex.output import warn

KNOWN_FIELDS = {
    "host",
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
}


@dataclass
class ProjectConfig:
    """Project configuration from .rex.toml."""

    root: Path
    host: str | None = None
    code_dir: str | None = None
    run_dir: str | None = None
    modules: list[str] = field(default_factory=list)
    cpu_partition: str | None = None
    gpu_partition: str | None = None
    gres: str | None = None
    time: str | None = None
    cpus: int | None = None
    mem: str | None = None
    constraint: str | None = None
    prefer: str | None = None
    default_gpu: bool = False

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

        return cls(
            root=path.parent,
            host=data.get("host"),
            code_dir=data.get("code_dir"),
            run_dir=data.get("run_dir"),
            modules=data.get("modules", []),
            cpu_partition=data.get("cpu_partition"),
            gpu_partition=data.get("gpu_partition"),
            gres=data.get("gres"),
            time=data.get("time"),
            cpus=data.get("cpus"),
            mem=data.get("mem"),
            constraint=data.get("constraint"),
            prefer=data.get("prefer"),
            default_gpu=data.get("default_gpu", False),
        )
