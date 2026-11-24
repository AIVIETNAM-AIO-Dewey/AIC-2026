"""Utilities shared by repository CLI entry points."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="YAML configuration file.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--video-id")


def read_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is required to read --config. Install the platform requirements first."
        ) from error
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return payload


def runtime_roots(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    required: tuple[str, ...] = ("data_root", "output_root", "cache_root"),
) -> dict[str, Path]:
    paths = config.get("paths", {})
    if paths is None:
        paths = {}
    if not isinstance(paths, dict):
        raise ValueError("config.paths must be a mapping")

    values = {
        "data_root": args.data_root or os.environ.get("AIC_DATA_ROOT") or paths.get("data_root"),
        "output_root": args.output_root
        or os.environ.get("AIC_ARTIFACT_ROOT")
        or paths.get("output_root"),
        "cache_root": args.cache_root
        or os.environ.get("AIC_CACHE_ROOT")
        or paths.get("cache_root"),
    }
    missing = [key for key in required if not values[key]]
    if missing:
        switches = ", ".join("--" + key.replace("_", "-") for key in missing)
        raise ValueError(f"Missing runtime roots: {switches}")
    return {
        key: Path(value).expanduser().resolve()
        for key, value in values.items()
        if value is not None
    }


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_device(explicit: str | None, config: dict[str, Any]) -> str:
    requested = explicit or str(config.get("device", "cuda"))
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_seed(explicit: int | None, config: dict[str, Any]) -> int:
    return int(explicit if explicit is not None else config.get("seed", 2026))


def seed_everything(seed: int) -> None:
    """Apply the recorded seed without forcing unsupported deterministic CUDA kernels."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
