#!/usr/bin/env python3
"""Plan or verify deterministic whole-video OCR Phase 1 shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _common import read_config  # noqa: E402
from aic2026.ocr import (  # noqa: E402
    CropConfig,
    OcrShardArtifactBundle,
    Phase1Identity,
    TrackingConfig,
    canonical_config_sha256,
    plan_frame_shards,
    runtime_identity_from_config,
    verify_global_shards,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "offline" / "ocr_phase1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "verify"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--global-manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        metavar="SHARD_ID=DETECTIONS,TRAJECTORIES,REPRESENTATIVES",
        help="Required complete per-shard artifact bundle for production verification.",
    )
    return parser


def _bundle_mapping(values: list[str]) -> dict[str, OcrShardArtifactBundle]:
    if not values:
        raise ValueError("verify requires at least one --bundle")
    output: dict[str, OcrShardArtifactBundle] = {}
    for value in values:
        shard_id, separator, raw_paths = value.partition("=")
        if not separator or not shard_id or shard_id in output:
            raise ValueError("--bundle must use unique SHARD_ID=... values")
        paths = raw_paths.split(",")
        if len(paths) != 3 or any(not item for item in paths):
            raise ValueError("--bundle requires detection,trajectory,representative paths")
        output[shard_id] = OcrShardArtifactBundle(
            detections=Path(paths[0]).expanduser().resolve(),
            trajectories=Path(paths[1]).expanduser().resolve(),
            representatives=Path(paths[2]).expanduser().resolve(),
        )
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = read_config(args.config)
    if config.get("schema_version") != "aic26.ocr_phase1.config.v1":
        raise ValueError("unsupported OCR Phase 1 config schema")
    if config.get("execution_profile") != "cpu_pinned":
        raise ValueError("runnable OCR Phase 1 config requires execution_profile=cpu_pinned")
    config_hash = canonical_config_sha256(config)
    tracking = TrackingConfig(**config.get("tracking", {}))
    crop = CropConfig(**config.get("crop", {}))
    runtime = runtime_identity_from_config(config)
    identity = Phase1Identity(
        detector_id=runtime["detector_id"],
        detector_revision=runtime["detector_revision"],
        detector_tree_sha256=runtime["detector_tree_sha256"],
        runtime_identity_sha256=runtime["runtime_identity_sha256"],
    )
    source = args.source_manifest.expanduser().resolve()
    if args.command == "plan":
        if args.output_dir is None:
            raise SystemExit("--output-dir is required for plan")
        path, manifest = plan_frame_shards(
            source_manifest=source,
            output_dir=args.output_dir.expanduser().resolve(),
            config_sha256=config_hash,
            tracking_config=tracking,
        )
        print(
            json.dumps(
                {
                    "status": "planned",
                    "global_manifest": str(path),
                    "shards": len(manifest.shards),
                    "frames": sum(item.frame_count for item in manifest.shards),
                }
            )
        )
        return 0
    if args.global_manifest is None:
        raise SystemExit("--global-manifest is required for verify")
    if args.data_root is None:
        raise SystemExit("--data-root is required for verify")
    counts = verify_global_shards(
        source_manifest=source,
        global_manifest=args.global_manifest.expanduser().resolve(),
        expected_config_sha256=config_hash,
        tracking_config=tracking,
        expected_run_id=config["run"]["run_id"],
        expected_identity=identity,
        shard_bundles=_bundle_mapping(args.bundle),
        data_root=args.data_root.expanduser().resolve(),
        crop_config=crop,
    )
    print(json.dumps({"status": "verified", **counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
