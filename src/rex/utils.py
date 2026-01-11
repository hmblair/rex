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
    """Generate timestamp-based job name."""
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d-%H%M%S")


def generate_script_id() -> str:
    """Generate unique script ID for temp files."""
    import time

    return f"{os.getpid()}-{int(time.time())}"
