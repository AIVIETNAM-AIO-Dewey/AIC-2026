#!/usr/bin/env python3
"""Create the deterministic identity required before qualifying Docker RAM.

Run on the Docker host after images have been built.  The caller supplies
immutable image IDs from ``docker image inspect`` and the script records the
compose, model manifests, and retrieval/RAM configuration as canonical JSON.
The resulting file is mounted in ``/state`` and must match the API environment
variable ``AIC_COMPOSE_FINGERPRINT``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA = "aic.runtime-fingerprint.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_fingerprint(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--compose-file", type=Path, required=True)
    parser.add_argument("--search-api-image-id", required=True)
    parser.add_argument("--qdrant-image-id", required=True)
    parser.add_argument("--query-model-manifest", type=Path, required=True)
    parser.add_argument("--branch1-model-manifest", type=Path, required=True)
    parser.add_argument("--cpu-threads", default=os.environ.get("AIC_CPU_THREADS", "8"))
    parser.add_argument("--max-rss-gib", default=os.environ.get("AIC_MAX_MODEL_RSS_GIB", "6.5"))
    args = parser.parse_args()
    material: dict[str, Any] = {
        "compose": _manifest_fingerprint(args.compose_file),
        "images": {
            "search_api": str(args.search_api_image_id),
            "qdrant": str(args.qdrant_image_id),
        },
        "model_manifests": {
            "query": _manifest_fingerprint(args.query_model_manifest),
            "branch1": _manifest_fingerprint(args.branch1_model_manifest),
        },
        "runtime": {
            "cpu_threads": str(args.cpu_threads),
            "max_rss_gib": str(args.max_rss_gib),
        },
    }
    canonical = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload = {
        "schema_version": SCHEMA,
        "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "material": material,
    }
    args.state_root.mkdir(parents=True, exist_ok=True)
    destination = args.state_root / "runtime_fingerprint.json"
    staging = destination.with_suffix(".staging.json")
    staging.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(staging, destination)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
