#!/usr/bin/env python3
"""Run or verify deterministic detector-only OCR Phase 1 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _common import read_config  # noqa: E402
from aic2026.contracts import OcrPhase1Receipt  # noqa: E402
from aic2026.ocr import (  # noqa: E402
    SUPPORTED_EXECUTION_PROFILES,
    CheckpointArtifactPaths,
    CropConfig,
    PaddleOcrV6Detector,
    Phase1Identity,
    TrackingConfig,
    canonical_config_sha256,
    publish_checkpoint,
    restore_checkpoint,
    run_detect_crop,
    run_representative_selection,
    run_tracking,
    runtime_identity_from_config,
    verify_detection_artifact,
    verify_detector_only,
    verify_linked_artifacts,
)
from aic2026.ocr.phase1 import (  # noqa: E402
    load_frame_manifest,
    receipt_path_for,
    validate_frame_sources,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "offline" / "ocr_phase1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "detect", "track", "select", "run", "verify", "checkpoint"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--runtime-cache-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--frame-manifest", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--global-manifest", type=Path)
    parser.add_argument("--detections", type=Path)
    parser.add_argument("--trajectories", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Verified read-only checkpoint bundle/root to restore into --output-root.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Durable export root, normally /kaggle/working/checkpoints/RUN/SHARD.",
    )
    parser.add_argument("--git-commit-sha")
    parser.add_argument("--shard-id", default="shard-standalone")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run model/frame preflight only; do not construct Paddle or write artifacts.",
    )
    return parser


def _required(path: Path | None, switch: str) -> Path:
    if path is None:
        raise SystemExit(f"{switch} is required")
    return path.expanduser().resolve()


def _settings(
    args: argparse.Namespace,
) -> tuple[dict, CropConfig, TrackingConfig, str, str, Phase1Identity]:
    config = read_config(args.config)
    if config.get("schema_version") != "aic26.ocr_phase1.config.v1":
        raise ValueError("unsupported OCR Phase 1 config schema")
    if config.get("execution_profile") not in SUPPORTED_EXECUTION_PROFILES:
        raise ValueError("runnable OCR Phase 1 config requires a supported pinned profile")
    crop = CropConfig(**config.get("crop", {}))
    tracking = TrackingConfig(**config.get("tracking", {}))
    run = config.get("run")
    if not isinstance(run, Mapping):
        raise ValueError("Phase 1 config requires run as a mapping")
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Phase 1 config requires run.run_id as a non-empty string")
    runtime = runtime_identity_from_config(config)
    identity = Phase1Identity(
        detector_id=runtime["detector_id"],
        detector_revision=runtime["detector_revision"],
        detector_tree_sha256=runtime["detector_tree_sha256"],
        runtime_identity_sha256=runtime["runtime_identity_sha256"],
    )
    return config, crop, tracking, run_id, canonical_config_sha256(config), identity


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    required_by_command = {
        "detect": ("detections",),
        "track": ("detections", "trajectories"),
        "select": ("trajectories", "representatives"),
        "run": ("detections", "trajectories", "representatives"),
        "verify": ("detections", "trajectories", "representatives"),
        "checkpoint": ("detections", "trajectories", "representatives"),
    }
    required = required_by_command.get(args.command, ())
    output_root = args.output_root.expanduser().resolve() if args.output_root else None
    paths: dict[str, Path] = {}
    for name in required:
        explicit = getattr(args, name)
        if explicit is not None:
            paths[name] = explicit.expanduser().resolve()
        elif output_root is not None:
            paths[name] = output_root / f"{name}.jsonl"
        else:
            raise SystemExit(f"--{name} or --output-root is required for {args.command}")
    return paths


def _resume_existing_stage(requested: bool, output: Path) -> bool:
    """Resume only a started stage, including an output published before its receipt."""

    candidates = (
        output,
        receipt_path_for(output),
        output.with_suffix(output.suffix + ".partial"),
        output.with_suffix(output.suffix + ".tmp"),
    )
    return requested and any(path.exists() for path in candidates)


def _detector_required(receipt: OcrPhase1Receipt | None, frame_count: int) -> bool:
    """A fully committed prefix can be verified/promoted without constructing Paddle."""

    return receipt is None or receipt.committed_records < frame_count


def _preflight(
    args: argparse.Namespace, config: dict, crop: CropConfig, tracking: TrackingConfig
) -> dict:
    del crop, tracking
    cache_root = _required(args.cache_root, "--cache-root")
    evidence = verify_detector_only(config, cache_root)
    frames = None
    if args.frame_manifest is not None:
        data_root = _required(args.data_root, "--data-root")
        source_validation = validate_frame_sources(
            args.frame_manifest.expanduser().resolve(), data_root
        )
        frames = source_validation["frames"]
    return {
        "status": "preflight_pass",
        "source_validation": "full_canonical" if frames is not None else "not_requested",
        "frames": frames,
        **evidence,
    }


def _checkpoint_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, str]:
    if args.output_root is None:
        raise SystemExit("--output-root is required with checkpoint export/restore")
    source_manifest = _required(args.source_manifest, "--source-manifest")
    global_manifest = _required(args.global_manifest, "--global-manifest")
    frame_manifest = _required(args.frame_manifest, "--frame-manifest")
    data_root = _required(args.data_root, "--data-root")
    if not isinstance(args.git_commit_sha, str) or not args.git_commit_sha:
        raise SystemExit("--git-commit-sha is required with checkpoint export/restore")
    output_root = args.output_root.expanduser().resolve()
    for name in ("detections", "trajectories", "representatives"):
        explicit = getattr(args, name)
        expected = output_root / f"{name}.jsonl"
        if explicit is not None and explicit.expanduser().resolve() != expected:
            raise SystemExit(
                f"--{name} must equal {expected} when checkpoint export/restore is enabled"
            )
    return source_manifest, global_manifest, frame_manifest, data_root, args.git_commit_sha


def _checkpoint_artifacts(output_root: Path) -> CheckpointArtifactPaths:
    return CheckpointArtifactPaths(
        detections=output_root / "detections.jsonl",
        trajectories=output_root / "trajectories.jsonl",
        representatives=output_root / "representatives.jsonl",
    )


def _publish_requested_checkpoint(
    *,
    args: argparse.Namespace,
    run_id: str,
    config_hash: str,
    identity: Phase1Identity,
    crop: CropConfig,
    tracking: TrackingConfig,
) -> Path | None:
    if args.checkpoint_root is None:
        return None
    source, global_manifest, frame_manifest, data_root, git_commit = _checkpoint_inputs(args)
    output_root = _required(args.output_root, "--output-root")
    return publish_checkpoint(
        checkpoint_root=args.checkpoint_root.expanduser().resolve(),
        artifact_root=output_root,
        artifacts=_checkpoint_artifacts(output_root),
        source_manifest=source,
        global_manifest=global_manifest,
        frame_manifest=frame_manifest,
        data_root=data_root,
        run_id=run_id,
        config_sha256=config_hash,
        git_commit_sha=git_commit,
        identity=identity,
        crop_config=crop,
        tracking_config=tracking,
        shard_id=args.shard_id,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, crop, tracking, run_id, config_hash, identity = _settings(args)
    if args.command == "preflight" or args.dry_run:
        print(json.dumps(_preflight(args, config, crop, tracking), ensure_ascii=False))
        return 0

    paths = _paths(args)
    if args.resume_from is not None:
        if args.checkpoint_root is None:
            raise SystemExit(
                "--checkpoint-root under writable storage is required with --resume-from"
            )
        source, global_manifest, frame_manifest, data_root, git_commit = _checkpoint_inputs(args)
        output_root = _required(args.output_root, "--output-root")
        restored = restore_checkpoint(
            checkpoint_root=args.resume_from.expanduser().resolve(),
            artifact_root=output_root,
            source_manifest=source,
            global_manifest=global_manifest,
            frame_manifest=frame_manifest,
            data_root=data_root,
            run_id=run_id,
            config_sha256=config_hash,
            git_commit_sha=git_commit,
            identity=identity,
            crop_config=crop,
            tracking_config=tracking,
            shard_id=args.shard_id,
            checkpoint_destination_root=args.checkpoint_root.expanduser().resolve(),
        )
        args.resume = True
        if args.command == "checkpoint":
            print(
                json.dumps(
                    {
                        "status": "restored",
                        "stage": restored.stage,
                        "checkpoint_sequence": restored.checkpoint_sequence,
                    }
                )
            )
            return 0

    if args.command == "checkpoint":
        published = _publish_requested_checkpoint(
            args=args,
            run_id=run_id,
            config_hash=config_hash,
            identity=identity,
            crop=crop,
            tracking=tracking,
        )
        if published is None:
            raise SystemExit("--checkpoint-root is required for checkpoint")
        print(json.dumps({"status": "checkpoint_published", "checkpoint": str(published)}))
        return 0
    if args.command in {"detect", "run"}:
        frame_manifest = _required(args.frame_manifest, "--frame-manifest")
        data_root = _required(args.data_root, "--data-root")
        # Validate manifest structure now; each source is decoded and bound when used.
        frames = load_frame_manifest(frame_manifest, data_root)
        detections = paths["detections"]
        detect_resume = _resume_existing_stage(args.resume, detections)
        detector = None
        receipt_path = receipt_path_for(detections)
        receipt = None
        if receipt_path.exists():
            receipt = OcrPhase1Receipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
        if _detector_required(receipt, len(frames)):
            cache_root = _required(args.cache_root, "--cache-root")
            runtime_cache_root = _required(args.runtime_cache_root, "--runtime-cache-root")
            detector = PaddleOcrV6Detector.create(
                config=config,
                cache_root=cache_root,
                runtime_cache_root=runtime_cache_root,
            )
        detect_counts = run_detect_crop(
            frame_manifest=frame_manifest,
            data_root=data_root,
            output=detections,
            run_id=run_id,
            config_sha256=config_hash,
            detector=detector,
            crop_config=crop,
            identity=identity,
            resume=detect_resume,
            tracking_config=tracking,
            shard_id=args.shard_id,
        )
        if args.command == "detect":
            checkpoint = _publish_requested_checkpoint(
                args=args,
                run_id=run_id,
                config_hash=config_hash,
                identity=identity,
                crop=crop,
                tracking=tracking,
            )
            print(
                json.dumps(
                    {
                        "status": "completed",
                        **detect_counts,
                        "checkpoint": str(checkpoint) if checkpoint else None,
                    }
                )
            )
            return 0

    if args.command in {"track", "run"}:
        detections = paths["detections"]
        trajectories = paths["trajectories"]
        track_resume = _resume_existing_stage(args.resume, trajectories)
        track_counts = run_tracking(
            detections=detections,
            output=trajectories,
            run_id=run_id,
            config_sha256=config_hash,
            tracking_config=tracking,
            identity=identity,
            resume=track_resume,
        )
        if args.command == "track":
            checkpoint = _publish_requested_checkpoint(
                args=args,
                run_id=run_id,
                config_hash=config_hash,
                identity=identity,
                crop=crop,
                tracking=tracking,
            )
            print(
                json.dumps(
                    {
                        "status": "completed",
                        **track_counts,
                        "checkpoint": str(checkpoint) if checkpoint else None,
                    }
                )
            )
            return 0

    if args.command in {"select", "run"}:
        trajectories = paths["trajectories"]
        representatives = paths["representatives"]
        select_resume = _resume_existing_stage(args.resume, representatives)
        selection_counts = run_representative_selection(
            trajectories=trajectories,
            output=representatives,
            run_id=run_id,
            config_sha256=config_hash,
            tracking_config=tracking,
            identity=identity,
            resume=select_resume,
        )
        checkpoint = _publish_requested_checkpoint(
            args=args,
            run_id=run_id,
            config_hash=config_hash,
            identity=identity,
            crop=crop,
            tracking=tracking,
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    **selection_counts,
                    "checkpoint": str(checkpoint) if checkpoint else None,
                }
            )
        )
        return 0

    if args.command == "verify":
        detections = paths["detections"]
        trajectories = paths["trajectories"]
        representatives = paths["representatives"]
        frame_manifest = _required(args.frame_manifest, "--frame-manifest")
        data_root = _required(args.data_root, "--data-root")
        detection_counts = verify_detection_artifact(
            output=detections,
            frame_manifest=frame_manifest,
            data_root=data_root,
            crop_config=crop,
            expected_run_id=run_id,
            expected_config_sha256=config_hash,
            expected_identity=identity,
            expected_shard_id=args.shard_id,
            tracking_config=tracking,
        )
        linked_counts = verify_linked_artifacts(
            detections=detections,
            trajectories=trajectories,
            representatives=representatives,
            expected_run_id=run_id,
            expected_config_sha256=config_hash,
            expected_identity=identity,
            tracking_config=tracking,
        )
        print(json.dumps({"status": "verified", **detection_counts, **linked_counts}))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
