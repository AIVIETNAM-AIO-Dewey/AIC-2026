"""Shared portable relative-path validation for artifact contracts."""

from __future__ import annotations

import re
from pathlib import PurePosixPath


def require_safe_relative_path(value: str, *, field_name: str = "path") -> None:
    """Reject absolute, platform-specific, escaping, or non-canonical paths."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty safe relative path")
    canonical = PurePosixPath(value).as_posix()
    if (
        "\\" in value
        or value.startswith("/")
        or re.match(r"^[A-Za-z]:", value)
        or canonical != value
        or canonical in {"", "."}
        or ".." in PurePosixPath(value).parts
    ):
        raise ValueError(f"{field_name} must be a canonical safe relative path")
