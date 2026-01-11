"""Alias file parsing (~/.config/rex)."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "rex"


@dataclass
class Alias:
    """A rex alias definition."""

    name: str
    target: str  # user@host or just host
    extra_args: list[str]  # Additional flags like -p /path/python


def load_aliases(path: Path | None = None) -> dict[str, Alias]:
    """Load aliases from config file.

    Format:
        name = user@host [options...]
        gpu = user@gpu.example.com -p /opt/python3.12
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH

    if not path.exists():
        return {}

    aliases = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            match = re.match(r"^(\w+)\s*=\s*(.+)$", line)
            if match:
                name = match.group(1)
                value = match.group(2).strip()

                # Split the value - first part is target, rest are args
                parts = shlex.split(value)
                if parts:
                    target = parts[0]
                    extra_args = parts[1:] if len(parts) > 1 else []
                    aliases[name] = Alias(name=name, target=target, extra_args=extra_args)

    return aliases


def expand_alias(
    name: str, aliases: dict[str, Alias]
) -> tuple[str, list[str]] | None:
    """Expand alias name to (target, extra_args).

    Returns None if not found or if name looks like a host (contains @).
    """
    if "@" in name:
        return None

    alias = aliases.get(name)
    if alias:
        return (alias.target, alias.extra_args)
    return None
