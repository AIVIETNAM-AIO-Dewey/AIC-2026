#!/usr/bin/env python3
"""Non-skippable real-model quality and performance gate for OCR Phase 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _common import read_config  # noqa: E402
from aic2026.common import iter_jsonl, sha256_file  # noqa: E402
from aic2026.contracts import FrameRef, OcrDetectionFrameRecord  # noqa: E402
from aic2026.ocr import (  # noqa: E402
    CropConfig,
    DetectionQualityConfig,
    PaddleOcrV6Detector,
    Phase1Identity,
    TrackingConfig,
    build_trajectories,
    canonical_config_sha256,
    enforce_quality_thresholds,
    evaluate_detection_quality,
    load_and_verify_execution_attestation,
    load_ground_truth_manifest,
    run_detect_crop,
    run_representative_selection,
    run_tracking,
    runtime_identity_from_config,
    validate_frame_sources,
    verify_detection_artifact,
    verify_file_unchanged,
    verify_linked_artifacts,
    verify_negative_fixture_suite,
    verify_negative_suite_receipt,
)
from aic2026.ocr.phase1 import receipt_path_for  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "offline" / "ocr_phase1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--runtime-cache-root", type=Path, required=True)
    parser.add_argument("--performance-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--execution-attestation", type=Path, required=True)
    parser.add_argument("--expected-source-commit-sha", required=True)
    parser.add_argument("--negative-manifest", type=Path, required=True)
    parser.add_argument("--negative-data-root", type=Path, required=True)
    parser.add_argument("--negative-suite-receipt", type=Path, required=True)
    return parser


def _peak_rss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if platform.system() == "Darwin" else raw * 1024)


def _verified_detection_snapshot(
    detections_path: Path,
    *,
    verify: Callable[[], object],
) -> tuple[list[OcrDetectionFrameRecord], tuple[tuple[Path, str], ...]]:
    """Verify and parse the same byte snapshot, rejecting any intervening mutation."""

    receipt_path = receipt_path_for(detections_path)
    artifact_payload = detections_path.read_bytes()
    receipt_payload = receipt_path.read_bytes()
    baseline = (
        (detections_path, hashlib.sha256(artifact_payload).hexdigest()),
        (receipt_path, hashlib.sha256(receipt_payload).hexdigest()),
    )
    for path, expected_hash in baseline:
        verify_file_unchanged(path, expected_hash, label=str(path))
    verify()
    for path, expected_hash in baseline:
        verify_file_unchanged(path, expected_hash, label=str(path))
    if sha256_file(detections_path) != baseline[0][1]:
        raise ValueError("detection artifact changed before quality metric snapshot")
    records = [
        OcrDetectionFrameRecord.model_validate_json(line)
        for line in artifact_payload.splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("verified detection artifact is empty")
    # Receipt bytes are captured with the artifact so verify cannot authenticate
    # one receipt and let the report bind a later replacement.
    if sha256_file(receipt_path) != baseline[1][1] or not receipt_payload:
        raise ValueError("detection receipt changed before quality metric snapshot")
    return records, baseline


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model_root = args.model_root.expanduser().resolve()
        runtime_root = args.runtime_cache_root.expanduser().resolve()
        performance_manifest = args.performance_manifest.expanduser().resolve()
        evaluation_manifest = args.evaluation_manifest.expanduser().resolve()
        data_root = args.data_root.expanduser().resolve()
        attestation_path = args.execution_attestation.expanduser().resolve()
        negative_manifest = args.negative_manifest.expanduser().resolve()
        negative_data_root = args.negative_data_root.expanduser().resolve()
        negative_receipt_path = args.negative_suite_receipt.expanduser().resolve()
        if not model_root.is_dir():
            raise ValueError("--model-root is missing")
        if not runtime_root.is_dir():
            raise ValueError("--runtime-cache-root must already exist and be writable")
        if not performance_manifest.is_file():
            raise ValueError("--performance-manifest is missing")
        probe = runtime_root / ".aic26-ocr-gate-write-probe"
        probe.write_bytes(b"")
        probe.unlink()

        config = read_config(args.config)
        if config.get("execution_profile") != "cpu_pinned":
            raise ValueError("real-model gate requires execution_profile=cpu_pinned")
        crop = CropConfig(**config["crop"])
        tracking = TrackingConfig(**config["tracking"])
        quality = DetectionQualityConfig(**config["quality"])
        config_hash = canonical_config_sha256(config)
        runtime = runtime_identity_from_config(config)
        identity = Phase1Identity(
            detector_id=runtime["detector_id"],
            detector_revision=runtime["detector_revision"],
            detector_tree_sha256=runtime["detector_tree_sha256"],
            runtime_identity_sha256=runtime["runtime_identity_sha256"],
        )

        performance_manifest_sha256 = sha256_file(performance_manifest)
        source_counts = validate_frame_sources(performance_manifest, data_root)
        frame_count = source_counts["frames"]
        if not 500 <= frame_count <= 1_000:
            raise ValueError("real-model performance pilot requires 500..1000 real frames")
        performance_frames = [
            FrameRef.model_validate(value) for value in iter_jsonl(performance_manifest)
        ]
        frames_by_uid = {frame.frame_uid: frame for frame in performance_frames}
        if len(frames_by_uid) != len(performance_frames):
            raise ValueError("duplicate frame_uid in performance manifest")
        evaluation_manifest_sha256, ground_truth = load_ground_truth_manifest(
            evaluation_manifest,
            frames_by_uid,
            config=quality,
        )
        execution_attestation_sha256, _attestation = load_and_verify_execution_attestation(
            attestation_path,
            expected_config_sha256=config_hash,
            expected_detector_revision=identity.detector_revision,
            expected_detector_tree_sha256=identity.detector_tree_sha256,
            expected_runtime_identity_sha256=identity.runtime_identity_sha256,
            expected_source_commit_sha=args.expected_source_commit_sha,
        )
        expected_negative_receipt, negative_baseline = verify_negative_fixture_suite(
            negative_manifest,
            negative_data_root,
            config_sha256=config_hash,
        )
        negative_suite_receipt_sha256 = verify_negative_suite_receipt(
            negative_receipt_path,
            expected_negative_receipt,
        )

        detector = PaddleOcrV6Detector.create(
            config=config,
            cache_root=model_root,
            runtime_cache_root=runtime_root,
        )
        artifact_root = runtime_root / "ocr" / "real-model-gates" / performance_manifest_sha256
        if artifact_root.exists():
            raise FileExistsError(
                "pilot artifact directory already exists; use a fresh runtime root"
            )
        detections_path = artifact_root / "detections.jsonl"
        trajectories_path = artifact_root / "trajectories.jsonl"
        representatives_path = artifact_root / "representatives.jsonl"
        run_id = config["run"]["run_id"]

        started = time.perf_counter()
        detection_started = time.perf_counter()
        run_detect_crop(
            frame_manifest=performance_manifest,
            data_root=data_root,
            output=detections_path,
            run_id=run_id,
            config_sha256=config_hash,
            detector=detector,
            crop_config=crop,
            identity=identity,
            tracking_config=tracking,
            shard_id="real-model-pilot",
        )
        detection_elapsed = time.perf_counter() - detection_started
        run_tracking(
            detections=detections_path,
            output=trajectories_path,
            run_id=run_id,
            config_sha256=config_hash,
            tracking_config=tracking,
            identity=identity,
        )
        run_representative_selection(
            trajectories=trajectories_path,
            output=representatives_path,
            run_id=run_id,
            config_sha256=config_hash,
            tracking_config=tracking,
            identity=identity,
        )
        records, detection_baseline = _verified_detection_snapshot(
            detections_path,
            verify=lambda: verify_detection_artifact(
                output=detections_path,
                frame_manifest=performance_manifest,
                data_root=data_root,
                crop_config=crop,
                expected_run_id=run_id,
                expected_config_sha256=config_hash,
                expected_identity=identity,
                expected_shard_id="real-model-pilot",
                tracking_config=tracking,
            ),
        )
        verify_linked_artifacts(
            detections=detections_path,
            trajectories=trajectories_path,
            representatives=representatives_path,
            expected_run_id=run_id,
            expected_config_sha256=config_hash,
            expected_identity=identity,
            tracking_config=tracking,
        )
        elapsed = time.perf_counter() - started

        quality_metrics = evaluate_detection_quality(records, ground_truth, config=quality)
        enforce_quality_thresholds(quality_metrics, quality)
        total = sum(len(record.detections) for record in records)
        multi_box = sum(len(record.detections) > 1 for record in records)
        detection_count_histogram: dict[str, int] = {}
        detections_by_video: dict[str, int] = {}
        for record in records:
            key = str(len(record.detections))
            detection_count_histogram[key] = detection_count_histogram.get(key, 0) + 1
            detections_by_video[record.video_id] = detections_by_video.get(
                record.video_id, 0
            ) + len(record.detections)
        tracking_metrics: dict[str, int] = {}
        build_trajectories(records, config=tracking, metrics=tracking_metrics)

        immutable_inputs = (
            (performance_manifest, performance_manifest_sha256),
            (evaluation_manifest, evaluation_manifest_sha256),
            (attestation_path, execution_attestation_sha256),
            (negative_receipt_path, negative_suite_receipt_sha256),
            *detection_baseline,
            *negative_baseline,
        )
        for path, expected_hash in immutable_inputs:
            verify_file_unchanged(path, expected_hash, label=str(path))

        report = {
            "status": "real_model_gate_pass",
            "elapsed_seconds": elapsed,
            "detection_elapsed_seconds": detection_elapsed,
            "frames_per_second": frame_count / detection_elapsed,
            "peak_rss_bytes": _peak_rss_bytes(),
            "frames": frame_count,
            "detections": total,
            "detections_per_frame": total / frame_count,
            "detection_count_histogram": dict(
                sorted(detection_count_histogram.items(), key=lambda item: int(item[0]))
            ),
            "detections_by_video": dict(sorted(detections_by_video.items())),
            "detections_by_shard": {"real-model-pilot": total},
            "multi_box_frames": multi_box,
            "performance_manifest_sha256": performance_manifest_sha256,
            "evaluation_manifest_sha256": evaluation_manifest_sha256,
            "config_sha256": config_hash,
            "source_commit_sha": args.expected_source_commit_sha,
            "threshold_policy": asdict(quality),
            "threshold_policy_sha256": quality.sha256,
            "execution_attestation_sha256": execution_attestation_sha256,
            "negative_manifest_sha256": expected_negative_receipt.negative_manifest_sha256,
            "negative_suite_receipt_sha256": negative_suite_receipt_sha256,
            **quality_metrics,
            **tracking_metrics,
            "artifact_root": str(artifact_root),
        }
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"real-model gate failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
