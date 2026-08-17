#!/usr/bin/env python3
"""Verify a cloud runtime before downloading or loading GPU models."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _path_from(value: str | None, env_name: str) -> Path | None:
    raw = value or os.environ.get(env_name)
    return Path(raw).expanduser().resolve() if raw else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_report() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False, "cuda_available": False}

    report: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        report["device_index"] = index
        report["device_name"] = properties.name
        report["total_vram_gb"] = round(properties.total_memory / 1024**3, 3)
        report["capability"] = list(torch.cuda.get_device_capability(index))
    return report


def _disk_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    return {
        "path": str(path),
        "probe_path": str(existing),
        "free_gb": round(usage.free / 1024**3, 3),
        "total_gb": round(usage.total / 1024**3, 3),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _data_root_check(path: Path | None) -> dict[str, Any]:
    ok = bool(path and path.is_dir())
    return {
        "name": "data_root",
        "ok": ok,
        "value": str(path) if path else None,
        "message": "Data root must already exist as a readable mounted directory.",
    }


def _writable_root_check(path: Path | None, name: str) -> dict[str, Any]:
    ok = False
    error: str | None = None
    if path is not None:
        try:
            path.mkdir(parents=True, exist_ok=True)
            if not path.is_dir():
                raise NotADirectoryError(path)
            with tempfile.NamedTemporaryFile(prefix=".aic-write-probe-", dir=path) as probe:
                probe.write(b"ok")
                probe.flush()
            ok = True
        except OSError as caught:
            error = f"{caught.__class__.__name__}: {caught}"
    return {
        "name": name,
        "ok": ok,
        "value": str(path) if path else None,
        "error": error,
        "message": f"{name} must be a writable directory.",
    }


def _offline_cache_check(path: Path | None) -> dict[str, Any]:
    ok = bool(path and path.is_dir())
    return {
        "name": "cache_root",
        "ok": ok,
        "value": str(path) if path else None,
        "message": "Offline cache root must be an existing attached snapshot directory.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report Python, CUDA, GPU, package, path, and disk information."
    )
    parser.add_argument("--config", help="Accepted for a uniform runner interface.")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--video-id")
    parser.add_argument("--minimum-vram-gb", type=float, default=14.0)
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write environment.json below --output-root in addition to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = _path_from(args.data_root, "AIC_DATA_ROOT")
    output_root = _path_from(args.output_root, "AIC_ARTIFACT_ROOT")
    cache_root = _path_from(args.cache_root, "AIC_CACHE_ROOT")
    torch_report = _torch_report()

    checks: list[dict[str, Any]] = []
    if args.device.startswith("cuda"):
        checks.append(
            {
                "name": "cuda_available",
                "ok": bool(torch_report.get("cuda_available")),
                "message": "PyTorch must see a CUDA device for SAM/DAM.",
            }
        )
        total_vram = float(torch_report.get("total_vram_gb", 0.0))
        checks.append(
            {
                "name": "minimum_vram",
                "ok": total_vram >= args.minimum_vram_gb,
                "actual_gb": total_vram,
                "minimum_gb": args.minimum_vram_gb,
            }
        )

    checks.append(_data_root_check(data_root))
    checks.append(_writable_root_check(output_root, "output_root"))
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        checks.append(_offline_cache_check(cache_root))
    else:
        checks.append(_writable_root_check(cache_root, "cache_root"))

    report = {
        "schema_version": "aic26.environment.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "accelerate",
                "huggingface-hub",
                "dam",
                "pydantic",
                "numpy",
                "pillow",
                "pyyaml",
                "jsonschema",
                "tqdm",
            )
        },
        "torch": torch_report,
        "paths": {
            "data_root": str(data_root) if data_root else None,
            "output_root": str(output_root) if output_root else None,
            "cache_root": str(cache_root) if cache_root else None,
        },
        "cache_disk": _disk_report(cache_root),
        "checks": checks,
        "ok": all(check["ok"] for check in checks),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.write_report:
        if output_root is None:
            print("--write-report requires --output-root or AIC_ARTIFACT_ROOT", file=sys.stderr)
            return 2
        output_check = next(check for check in checks if check["name"] == "output_root")
        if not output_check["ok"]:
            print("Cannot write report because output root is not writable", file=sys.stderr)
            return 2
        _atomic_json(output_root / "environment.json", report)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
