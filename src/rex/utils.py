"""Utility functions."""

from __future__ import annotations

import os
import re
from pathlib import Path


def validate_job_name(name: str) -> None:
    """Validate job name (alphanumeric, dash, underscore only).

    Raises ValueError if invalid.
    """
    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        raise ValueError(
            f"Invalid job name: '{name}' (use only alphanumeric, dash, underscore)"
        )


def validate_slurm_time(time_str: str) -> None:
    """Validate SLURM time format.

    Valid formats:
        - MM (minutes)
        - MM:SS (minutes:seconds)
        - HH:MM:SS (hours:minutes:seconds)
        - D-HH:MM:SS (days-hours:minutes:seconds)
        - D-HH:MM (days-hours:minutes)
        - D-HH (days-hours)

    Raises ValueError if invalid.
    """
    # Pattern for D-HH:MM:SS or D-HH:MM or D-HH
    days_pattern = r"^\d+-\d{1,2}(:\d{2}(:\d{2})?)?$"
    # Pattern for HH:MM:SS or MM:SS or MM
    time_pattern = r"^\d+(:\d{2}(:\d{2})?)?$"

    if re.match(days_pattern, time_str) or re.match(time_pattern, time_str):
        # Validate numeric ranges
        if "-" in time_str:
            days_part, time_part = time_str.split("-", 1)
            parts = time_part.split(":")
        else:
            parts = time_str.split(":")

        # Check each component is in valid range
        if len(parts) >= 2:
            # Minutes and seconds should be 0-59
            for part in parts[1:]:
                val = int(part)
                if val < 0 or val > 59:
                    raise ValueError(
                        f"Invalid time format: '{time_str}' (minutes/seconds must be 0-59)"
                    )
        if len(parts) >= 1 and len(parts) <= 3:
            # Hours can be 0-23 in D-HH format, but unlimited in HH:MM:SS
            if "-" in time_str and len(parts) >= 1:
                hours = int(parts[0])
                if hours > 23:
                    raise ValueError(
                        f"Invalid time format: '{time_str}' (hours must be 0-23 in D-HH format)"
                    )
        return

    raise ValueError(
        f"Invalid time format: '{time_str}' (use MM, HH:MM:SS, or D-HH:MM:SS)"
    )


def validate_memory(mem_str: str) -> None:
    """Validate SLURM memory format.

    Valid formats:
        - <number> (bytes)
        - <number>K (kilobytes)
        - <number>M (megabytes)
        - <number>G (gigabytes)
        - <number>T (terabytes)

    Raises ValueError if invalid.
    """
    if not re.match(r"^\d+[KMGT]?$", mem_str, re.IGNORECASE):
        raise ValueError(
            f"Invalid memory format: '{mem_str}' (use e.g., 4G, 16000M, 512K)"
        )

    # Extract numeric part and check it's positive
    num_match = re.match(r"^(\d+)", mem_str)
    if num_match and int(num_match.group(1)) == 0:
        raise ValueError(f"Invalid memory format: '{mem_str}' (must be greater than 0)")


def validate_gres(gres_str: str) -> None:
    """Validate SLURM GRES format.

    Valid formats:
        - gpu:N (N GPUs of any type)
        - gpu:type:N (N GPUs of specific type)
        - gpu:type (GPUs of specific type, count determined by SLURM)

    Raises ValueError if invalid.
    """
    # Basic pattern: resource:count or resource:type:count
    # Common: gpu:1, gpu:a100:2, gpu:v100
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*(:[a-zA-Z0-9_]+)*(:\d+)?$", gres_str):
        raise ValueError(
            f"Invalid GRES format: '{gres_str}' (use e.g., gpu:1, gpu:a100:2)"
        )


def validate_cpus(cpus: int) -> None:
    """Validate SLURM CPU count.

    Raises ValueError if invalid.
    """
    if cpus < 1:
        raise ValueError(f"Invalid CPU count: {cpus} (must be at least 1)")


def map_to_remote(local_path: Path, remote_home: str) -> str:
    """Map local path to remote path under remote $HOME.

    Strips /Users/<user> or /home/<user> and prepends remote home.
    """
    path_str = str(local_path.resolve())

    if path_str.startswith("/Users/"):
        # macOS: /Users/<user>/... -> $HOME/...
        parts = path_str.split("/")
        if len(parts) >= 3:
            # Skip /Users/<user>
            remainder = "/".join(parts[3:])
            return f"{remote_home}/{remainder}" if remainder else remote_home
    elif path_str.startswith("/home/"):
        # Linux: /home/<user>/... -> $HOME/...
        parts = path_str.split("/")
        if len(parts) >= 3:
            remainder = "/".join(parts[3:])
            return f"{remote_home}/{remainder}" if remainder else remote_home

    # No transformation
    return path_str


def job_pattern(job_id: str) -> str:
    """Build pgrep pattern that won't match itself (character class trick)."""
    return f"rex-{job_id}[.]py"


def generate_job_name() -> str:
    """Generate unique job name with timestamp and random suffix."""
    import secrets
    from datetime import datetime

    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def generate_script_id() -> str:
    """Generate unique script ID for temp files."""
    import time

    return f"{os.getpid()}-{int(time.time())}"
