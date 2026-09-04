#!/usr/bin/env python3
"""Publish an explicit host/container memory qualification for production readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

MAX_BYTES = int(float(os.environ.get("AIC_MAX_MODEL_RSS_GIB", "6.5")) * 1024**3)
SCHEMA = "aic.resource-qualification.v2"
RUNTIME_FINGERPRINT_SCHEMA = "aic.runtime-fingerprint.v1"


def _load_runtime_fingerprint(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = str(payload.get("fingerprint") or "").strip()
    if (
        payload.get("schema_version") != RUNTIME_FINGERPRINT_SCHEMA
        or not isinstance(payload.get("material"), dict)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("runtime fingerprint file is missing a valid SHA-256 digest")
    # Reject a hand-edited payload whose declared canonical material does not
    # hash to its own fingerprint.
    material = payload.get("material")
    canonical = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != fingerprint:
        raise ValueError("runtime fingerprint digest does not match its canonical material")
    return fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-root", type=Path, default=Path(os.environ.get("AIC_STATE_ROOT", "/state"))
    )
    parser.add_argument("--stack-peak-bytes", type=int, required=True)
    parser.add_argument("--api-peak-bytes", type=int, required=True)
    parser.add_argument("--worker-peak-bytes", type=int, required=True)
    parser.add_argument("--qdrant-peak-bytes", type=int, required=True)
    parser.add_argument(
        "--runtime-fingerprint-file",
        type=Path,
        default=None,
        help="Host-produced runtime_fingerprint.json; arbitrary fingerprints are rejected.",
    )
    parser.add_argument("--branch1-tested", action="store_true")
    parser.add_argument("--branch2-tested", action="store_true")
    parser.add_argument("--siglip2-tested", action="store_true")
    parser.add_argument("--metaclip2-tested", action="store_true")
    parser.add_argument("--bge-m3-tested", action="store_true")
    parser.add_argument("--beit3-tested", action="store_true")
    args = parser.parse_args()
    fingerprint_path = args.runtime_fingerprint_file or (
        args.state_root / "runtime_fingerprint.json"
    )
    runtime_fingerprint = _load_runtime_fingerprint(fingerprint_path)
    values = (
        args.stack_peak_bytes,
        args.api_peak_bytes,
        args.worker_peak_bytes,
        args.qdrant_peak_bytes,
    )
    if any(value < 0 for value in values):
        raise ValueError("memory measurements must be non-negative")
    if args.stack_peak_bytes < max(
        args.api_peak_bytes, args.worker_peak_bytes, args.qdrant_peak_bytes
    ):
        raise ValueError("stack peak must be at least each component peak")
    if args.stack_peak_bytes > MAX_BYTES:
        raise ValueError(
            f"stack peak {args.stack_peak_bytes} exceeds the {MAX_BYTES} byte production limit"
        )
    if not (
        args.branch1_tested
        and args.branch2_tested
        and args.siglip2_tested
        and args.metaclip2_tested
        and args.bge_m3_tested
        and args.beit3_tested
    ):
        raise ValueError(
            "resource qualification requires completed Branch-1/2 and all four model measurements"
        )
    args.state_root.mkdir(parents=True, exist_ok=True)
    destination = args.state_root / "resource_qualification.json"
    staging = destination.with_suffix(".staging.json")
    report = {
        "schema_version": SCHEMA,
        "passed": True,
        "production_ready": True,
        "stack_peak_rss_bytes": args.stack_peak_bytes,
        "api_peak_rss_bytes": args.api_peak_bytes,
        "worker_peak_rss_bytes": args.worker_peak_bytes,
        "qdrant_peak_rss_bytes": args.qdrant_peak_bytes,
        "limit_bytes": MAX_BYTES,
        "coverage": {
            "branch1": args.branch1_tested,
            "branch2": args.branch2_tested,
            "siglip2": args.siglip2_tested,
            "metaclip2": args.metaclip2_tested,
            "bge_m3": args.bge_m3_tested,
            "beit3": args.beit3_tested,
        },
        "compose_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_path": str(fingerprint_path),
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }
    staging.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(staging, destination)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
