"""Process RSS accounting used by the CPU readiness gates."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_PRODUCTION_RSS_BYTES = int(float(os.environ.get("AIC_MAX_MODEL_RSS_GIB", "6.5")) * 1024**3)


def current_process_rss_bytes() -> int:
    """Return the current process RSS on Linux and macOS."""

    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError, AttributeError):
        pass

    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            # BSD ps reports RSS in KiB.
            return int(result.stdout.strip()) * 1024
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return 0


def peak_process_rss_bytes() -> int | None:
    """Return this process' peak RSS in bytes (Linux and macOS aware)."""
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB while macOS reports bytes.  Detect the platform
        # explicitly: a size-based heuristic turns every macOS process whose
        # peak is below 1 GiB into a fictitious terabyte-scale process.
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


RESOURCE_QUALIFICATION_SCHEMA = "aic.resource-qualification.v2"
RUNTIME_FINGERPRINT_SCHEMA = "aic.runtime-fingerprint.v1"


def _is_valid_timestamp(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except (TypeError, ValueError):
        return False


def _runtime_fingerprint(state_root: Path) -> dict[str, Any]:
    """Read the fingerprint produced by the host-side measurement setup.

    The API image intentionally cannot invent a compose/image identity at
    runtime.  Requiring a host-produced, immutable record prevents an old RAM
    measurement from being reused after an image, compose, or model change.
    """
    path = state_root / "runtime_fingerprint.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("runtime fingerprint must be an object")
        fingerprint = str(payload.get("fingerprint") or "").strip()
        material = payload.get("material")
        canonical = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        valid = (
            payload.get("schema_version") == RUNTIME_FINGERPRINT_SCHEMA
            and isinstance(material, dict)
            and len(fingerprint) == 64
            and all(character in "0123456789abcdef" for character in fingerprint)
            and hashlib.sha256(canonical.encode("utf-8")).hexdigest() == fingerprint
        )
        return {
            "ready": valid,
            "path": str(path),
            "fingerprint": fingerprint if valid else "",
            "payload": payload if valid else None,
            "error": None if valid else "invalid runtime fingerprint schema or digest",
        }
    except (OSError, ValueError, TypeError) as error:
        return {
            "ready": False,
            "path": str(path),
            "fingerprint": "",
            "payload": None,
            "error": str(error),
        }


def resource_qualification(state_root: Path) -> dict[str, Any]:
    """Read the host/container stack memory qualification, fail closed."""
    path = state_root / "resource_qualification.json"
    fingerprint_matches = False
    runtime_identity = _runtime_fingerprint(state_root)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("resource qualification must be an object")
    except (OSError, ValueError, TypeError) as error:
        return {
            "ready": False,
            "production_ready": False,
            "schema_version": RESOURCE_QUALIFICATION_SCHEMA,
            "path": str(path),
            "fingerprint_matches": False,
            "error": str(error),
        }
    component_peaks = {
        name: -1
        for name in (
            "api_peak_rss_bytes",
            "worker_peak_rss_bytes",
            "qdrant_peak_rss_bytes",
        )
    }
    try:
        stack_peak = int(report.get("stack_peak_rss_bytes", -1))
        component_peaks = {name: int(report.get(name, -1)) for name in component_peaks}
        coverage = report.get("coverage") or {}
        coverage_ready = all(
            coverage.get(name) is True
            for name in ("branch1", "branch2", "siglip2", "metaclip2", "bge_m3", "beit3")
        )
        measurements_ready = (
            all(value >= 0 for value in component_peaks.values())
            and stack_peak >= max(component_peaks.values(), default=0)
            and bool(str(report.get("measured_at") or "").strip())
        )
        expected_fingerprint = str(os.environ.get("AIC_COMPOSE_FINGERPRINT") or "").strip()
        measured_at = str(report.get("measured_at") or "").strip()
        timestamp_valid = _is_valid_timestamp(measured_at)
        fingerprint = str(report.get("compose_fingerprint") or "").strip()
        fingerprint_matches = bool(
            runtime_identity["ready"]
            and expected_fingerprint
            and fingerprint
            and fingerprint == expected_fingerprint == runtime_identity["fingerprint"]
        )
        valid = (
            report.get("schema_version") == RESOURCE_QUALIFICATION_SCHEMA
            and report.get("passed") is True
            and report.get("production_ready") is True
            and stack_peak >= 0
            and stack_peak <= MAX_PRODUCTION_RSS_BYTES
            and coverage_ready
            and measurements_ready
            and timestamp_valid
            and fingerprint_matches
        )
    except (TypeError, ValueError, AttributeError):
        valid = False
        stack_peak = -1
    return {
        "ready": valid,
        "production_ready": valid,
        "schema_version": report.get("schema_version"),
        "path": str(path),
        "stack_peak_rss_bytes": stack_peak,
        "api_peak_rss_bytes": component_peaks.get("api_peak_rss_bytes", -1),
        "worker_peak_rss_bytes": component_peaks.get("worker_peak_rss_bytes", -1),
        "qdrant_peak_rss_bytes": component_peaks.get("qdrant_peak_rss_bytes", -1),
        "limit_bytes": MAX_PRODUCTION_RSS_BYTES,
        "coverage": report.get("coverage") or {},
        "fingerprint_matches": fingerprint_matches,
        "measurements_valid": bool(
            all(value >= 0 for value in component_peaks.values())
            and stack_peak >= max(component_peaks.values(), default=0)
        ),
        "timestamp_valid": bool(
            str(report.get("measured_at") or "").strip()
            and _is_valid_timestamp(str(report.get("measured_at") or "").strip())
        ),
        "measured_at": report.get("measured_at"),
        "compose_fingerprint": report.get("compose_fingerprint"),
        "runtime_fingerprint": runtime_identity,
        "error": None if valid else "missing, stale, incomplete, or over-limit qualification",
    }


__all__ = [
    "MAX_PRODUCTION_RSS_BYTES",
    "RESOURCE_QUALIFICATION_SCHEMA",
    "RUNTIME_FINGERPRINT_SCHEMA",
    "current_process_rss_bytes",
    "peak_process_rss_bytes",
    "resource_qualification",
]
