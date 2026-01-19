"""Global configuration (~/.config/rex/config.toml)."""

from __future__ import annotations

import tomli
from dataclasses import dataclass, field
from pathlib import Path

from rex.exceptions import ConfigError
from rex.output import warn
from rex.utils import validate_slurm_time, validate_memory, validate_gres, validate_cpus

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "rex" / "config.toml"

KNOWN_HOST_FIELDS = {
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
class HostConfig:
    """Per-host configuration defaults."""

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
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class GlobalConfig:
    """Global configuration from ~/.config/rex/config.toml."""

    aliases: dict[str, str]  # name -> user@host
    hosts: dict[str, HostConfig]  # alias_name -> config

    @classmethod
    def load(cls, path: Path | None = None) -> "GlobalConfig":
        """Load global config from file.

        Returns empty config if file doesn't exist.
        """
        if path is None:
            path = DEFAULT_CONFIG_PATH

        if not path.exists():
            return cls(aliases={}, hosts={})

        with open(path, "rb") as f:
            data = tomli.load(f)

        aliases = data.get("aliases", {})
        hosts: dict[str, HostConfig] = {}

        hosts_data = data.get("hosts", {})
        for host_name, host_data in hosts_data.items():
            # Warn about unknown fields
            unknown = set(host_data.keys()) - KNOWN_HOST_FIELDS
            if unknown:
                warn(f"config.toml: [hosts.{host_name}] unknown fields: {', '.join(sorted(unknown))}")

            # Validate SLURM options
            try:
                if host_data.get("time"):
                    validate_slurm_time(host_data["time"])
                if host_data.get("mem"):
                    validate_memory(host_data["mem"])
                if host_data.get("gres"):
                    validate_gres(host_data["gres"])
                if host_data.get("cpus") is not None:
                    validate_cpus(host_data["cpus"])
            except ValueError as e:
                raise ConfigError(f"config.toml: [hosts.{host_name}] {e}")

            hosts[host_name] = HostConfig(
                code_dir=host_data.get("code_dir"),
                run_dir=host_data.get("run_dir"),
                modules=host_data.get("modules", []),
                cpu_partition=host_data.get("cpu_partition"),
                gpu_partition=host_data.get("gpu_partition"),
                gres=host_data.get("gres"),
                time=host_data.get("time"),
                cpus=host_data.get("cpus"),
                mem=host_data.get("mem"),
                constraint=host_data.get("constraint"),
                prefer=host_data.get("prefer"),
                default_gpu=host_data.get("default_gpu", False),
                env=host_data.get("env", {}),
            )

        return cls(aliases=aliases, hosts=hosts)

    def get_host_config(self, alias_or_host: str) -> HostConfig | None:
        """Get host config for an alias or host string.

        Returns None if no config exists for this host.
        """
        # Try direct lookup by alias name
        if alias_or_host in self.hosts:
            return self.hosts[alias_or_host]

        # Try to match by resolved target
        if alias_or_host in self.aliases:
            # alias_or_host is an alias name, look up its config
            return self.hosts.get(alias_or_host)

        return None

    def expand_alias(self, name: str) -> str | None:
        """Expand alias name to target (user@host).

        Returns None if not found or if name already contains @.
        """
        if "@" in name:
            return None
        return self.aliases.get(name)
