#!/usr/bin/env python3
"""Prepare and run the fail-closed local-recognizer A/B evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _common import read_config  # noqa: E402
from aic2026.common import atomic_write_json  # noqa: E402
from aic2026.ocr.detector_only import runtime_identity_from_config  # noqa: E402
from aic2026.ocr.local_recognition import (  # noqa: E402
    VietOcrRecognizer,
    evaluate_local_recognition,
    export_l23_verified_evaluation,
    run_local_recognition_evaluation,
)
from aic2026.ocr.phase1 import Phase1Identity, canonical_config_sha256  # noqa: E402
from aic2026.ocr.representative_recognition import (  # noqa: E402
    DEFAULT_FRAME_CACHE_CAPACITY,
    DEFAULT_FRAME_CACHE_MAX_BYTES,
    DEFAULT_RECOGNITION_BATCH_SIZE,
    local_runtime_identity,
    merge_representative_recognition_partitions,
    run_representative_recognition,
)
from aic2026.ocr.tracking import TrackingConfig  # noqa: E402
from aic2026.ocr.trajectory_consensus import (  # noqa: E402
    build_final_ocr_artifact,
    run_trajectory_consensus,
)

DEFAULT_PHASE1_CONFIG = REPO_ROOT / "configs" / "offline" / "ocr_phase1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-l23", help="Export verified L23 labels read-only")
    prepare.add_argument("--annotation-root", type=Path, required=True)
    prepare.add_argument("--state-db", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--expected-samples", type=int, default=165)

    run = commands.add_parser("run-vietocr", help="Run pinned VietOCR on an eval manifest")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--crop-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "configs" / "offline" / "vietocr_vgg_seq2seq.yaml",
    )
    run.add_argument("--weights", type=Path, required=True)
    run.add_argument("--weights-sha256", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--resume", action="store_true")

    representatives = commands.add_parser(
        "run-representatives",
        help="Recognize one verified Phase 1 shard with crash-safe resume",
    )
    representatives.add_argument("--phase1-config", type=Path, default=DEFAULT_PHASE1_CONFIG)
    representatives.add_argument("--frame-manifest", type=Path, required=True)
    representatives.add_argument("--data-root", type=Path, required=True)
    representatives.add_argument("--detections", type=Path, required=True)
    representatives.add_argument("--trajectories", type=Path, required=True)
    representatives.add_argument("--representatives", type=Path, required=True)
    representatives.add_argument("--output", type=Path, required=True)
    representatives.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "configs" / "offline" / "vietocr_vgg_seq2seq.yaml",
    )
    representatives.add_argument("--weights", type=Path, required=True)
    representatives.add_argument("--weights-sha256", required=True)
    representatives.add_argument("--device", default="cuda:0")
    representatives.add_argument("--source-commit-sha", required=True)
    representatives.add_argument("--commit-interval-records", type=int, default=32)
    representatives.add_argument("--batch-size", type=int, default=DEFAULT_RECOGNITION_BATCH_SIZE)
    representatives.add_argument(
        "--frame-cache-capacity", type=int, default=DEFAULT_FRAME_CACHE_CAPACITY
    )
    representatives.add_argument(
        "--frame-cache-max-bytes", type=int, default=DEFAULT_FRAME_CACHE_MAX_BYTES
    )
    representatives.add_argument("--resume", action="store_true")
    representatives.add_argument("--representative-start", type=int, default=0)
    representatives.add_argument("--representative-end", type=int)

    merge = commands.add_parser(
        "merge-representatives",
        help="Merge completed recognition partitions into canonical representative order",
    )
    merge.add_argument("--representatives", type=Path, required=True)
    merge.add_argument("--partition-output", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)

    consensus = commands.add_parser(
        "consensus", help="Build one deterministic result per Phase 1 trajectory"
    )
    consensus.add_argument("--trajectories", type=Path, required=True)
    consensus.add_argument("--representatives", type=Path, required=True)
    consensus.add_argument("--recognition-output", type=Path, required=True)
    consensus.add_argument("--output", type=Path, required=True)
    consensus.add_argument("--run-id", required=True)

    final = commands.add_parser("build-final", help="Build backend-ingestible OCR JSONL")
    final.add_argument("--trajectories", type=Path, required=True)
    final.add_argument("--consensus", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--run-id", required=True)

    score = commands.add_parser("score", help="Score completed local OCR results")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--results", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument(
        "--minimum-exact-match",
        type=float,
        default=0.0,
        help="Optional quality threshold; zero checks only that inference works",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-l23":
        annotation_root = args.annotation_root.expanduser().resolve()
        state_db = (
            args.state_db.expanduser().resolve()
            if args.state_db
            else annotation_root / "annotation_state.sqlite3"
        )
        summary = export_l23_verified_evaluation(
            state_db=state_db,
            annotation_root=annotation_root,
            output=args.output.expanduser().resolve(),
        )
        if summary["samples"] != args.expected_samples:
            raise SystemExit(
                f"verified sample count {summary['samples']} differs from expected "
                f"{args.expected_samples}"
            )
        print(json.dumps({"status": "prepared", **summary}, ensure_ascii=False))
        return 0

    if args.command == "run-vietocr":
        recognizer = VietOcrRecognizer.create(
            config_path=args.model_config.expanduser().resolve(),
            weights_path=args.weights.expanduser().resolve(),
            device=args.device,
            expected_weights_sha256=args.weights_sha256,
        )
        counts = run_local_recognition_evaluation(
            manifest=args.manifest.expanduser().resolve(),
            crop_root=args.crop_root.expanduser().resolve(),
            output=args.output.expanduser().resolve(),
            recognizer=recognizer,
            resume=args.resume,
        )
        print(json.dumps({"status": "completed", **counts}))
        return 0

    if args.command == "run-representatives":
        phase1_config_path = args.phase1_config.expanduser().resolve()
        phase1_config = read_config(phase1_config_path)
        if phase1_config.get("schema_version") != "aic26.ocr_phase1.config.v1":
            raise SystemExit("unsupported OCR Phase 1 config schema")
        run_config = phase1_config.get("run")
        if not isinstance(run_config, Mapping) or not isinstance(run_config.get("run_id"), str):
            raise SystemExit("Phase 1 config requires run.run_id")
        tracking_config = phase1_config.get("tracking")
        if not isinstance(tracking_config, Mapping):
            raise SystemExit("Phase 1 config requires tracking settings")
        runtime = runtime_identity_from_config(phase1_config)
        identity = Phase1Identity(
            detector_id=runtime["detector_id"],
            detector_revision=runtime["detector_revision"],
            detector_tree_sha256=runtime["detector_tree_sha256"],
            runtime_identity_sha256=runtime["runtime_identity_sha256"],
        )
        model_config = args.model_config.expanduser().resolve()
        weights = args.weights.expanduser().resolve()
        recognizer = VietOcrRecognizer.create(
            config_path=model_config,
            weights_path=weights,
            device=args.device,
            expected_weights_sha256=args.weights_sha256,
        )
        packages, recognition_runtime, recognition_runtime_hash = local_runtime_identity(
            args.device
        )
        counts = run_representative_recognition(
            frame_manifest=args.frame_manifest.expanduser().resolve(),
            data_root=args.data_root.expanduser().resolve(),
            detections=args.detections.expanduser().resolve(),
            trajectories=args.trajectories.expanduser().resolve(),
            representatives=args.representatives.expanduser().resolve(),
            output=args.output.expanduser().resolve(),
            run_id=run_config["run_id"],
            phase1_config_sha256=canonical_config_sha256(phase1_config),
            phase1_identity=identity,
            tracking_config=TrackingConfig(**dict(tracking_config)),
            recognizer=recognizer,
            model_config=model_config,
            model_weights=weights,
            expected_model_weights_sha256=args.weights_sha256,
            source_commit_sha=args.source_commit_sha,
            package_versions=packages,
            runtime=recognition_runtime,
            runtime_identity_sha256=recognition_runtime_hash,
            commit_interval_records=args.commit_interval_records,
            batch_size=args.batch_size,
            frame_cache_capacity=args.frame_cache_capacity,
            frame_cache_max_bytes=args.frame_cache_max_bytes,
            representative_start=args.representative_start,
            representative_end=args.representative_end,
            resume=args.resume,
        )
        print(json.dumps({"status": "completed", **counts}, ensure_ascii=False))
        return 0

    if args.command == "merge-representatives":
        counts = merge_representative_recognition_partitions(
            representatives=args.representatives.expanduser().resolve(),
            partition_outputs=[item.expanduser().resolve() for item in args.partition_output],
            output=args.output.expanduser().resolve(),
        )
        print(json.dumps({"status": "completed", **counts}, ensure_ascii=False))
        return 0

    if args.command == "consensus":
        counts = run_trajectory_consensus(
            trajectories=args.trajectories.expanduser().resolve(),
            representatives=args.representatives.expanduser().resolve(),
            recognition_output=args.recognition_output.expanduser().resolve(),
            output=args.output.expanduser().resolve(),
            run_id=args.run_id,
        )
        print(json.dumps({"status": "completed", **counts}, ensure_ascii=False))
        return 0

    if args.command == "build-final":
        counts = build_final_ocr_artifact(
            trajectories=args.trajectories.expanduser().resolve(),
            consensus=args.consensus.expanduser().resolve(),
            output=args.output.expanduser().resolve(),
            run_id=args.run_id,
        )
        print(json.dumps({"status": "completed", **counts}, ensure_ascii=False))
        return 0

    if not 0 <= args.minimum_exact_match <= 1:
        raise SystemExit("--minimum-exact-match must be inside [0, 1]")
    report = evaluate_local_recognition(
        manifest=args.manifest.expanduser().resolve(),
        results=args.results.expanduser().resolve(),
        minimum_exact_match=args.minimum_exact_match,
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"fresh score output is required: {output}")
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
