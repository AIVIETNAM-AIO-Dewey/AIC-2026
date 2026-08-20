from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import aic2026.ocr.phase1 as phase1_module
import numpy as np
import pytest
import yaml
from aic2026.common import atomic_write_json, sha256_file, write_jsonl_atomic
from aic2026.contracts import (
    FrameRef,
    OcrDetection,
    OcrDetectionFrameRecord,
    OcrPhase1Receipt,
    QuadGeometry,
)
from aic2026.ocr.detector_only import (
    DetectorPolygon,
    PaddleOcrV6Detector,
    runtime_identity_from_config,
)
from aic2026.ocr.frame_snapshot import decode_canonical_frame
from aic2026.ocr.geometry import CropConfig, canonical_quad
from aic2026.ocr.phase1 import (
    Phase1Identity,
    canonical_config_sha256,
    load_frame_manifest,
    receipt_path_for,
    run_representative_selection,
    run_tracking,
    validate_frame_sources,
    verify_detection_artifact,
    verify_linked_artifacts,
)
from aic2026.ocr.phase1 import (
    _run_detect_crop_for_test as run_detect_crop,
)
from aic2026.ocr.quality_gate import (
    DetectionQualityConfig,
    GroundTruthFrame,
    GroundTruthInstance,
    enforce_quality_thresholds,
    evaluate_detection_quality,
    load_and_verify_execution_attestation,
    load_ground_truth_manifest,
    verify_file_unchanged,
    verify_negative_fixture_suite,
    verify_negative_suite_receipt,
)
from aic2026.ocr.sharding import (
    OcrShardArtifactBundle,
    plan_frame_shards,
    verify_global_shard_structure,
    verify_global_shards,
)
from aic2026.ocr.tracking import TrackingConfig
from PIL import Image

CONFIG_HASH = "a" * 64
IDENTITY = Phase1Identity()


class FakeDetector:
    def __init__(self, *, interrupt_call: int | None = None) -> None:
        self.interrupt_call = interrupt_call
        self.calls: list[str] = []

    def detect(self, image_bgr: np.ndarray, *, width: int, height: int) -> list[DetectorPolygon]:
        assert image_bgr.shape == (height, width, 3)
        self.calls.append(str(int(image_bgr[0, 0, 0])))
        if self.interrupt_call == len(self.calls):
            raise KeyboardInterrupt
        raw_points = ((3.0, 3.0), (24.0, 2.0), (25.0, 12.0), (2.0, 13.0))
        return [
            DetectorPolygon(
                source_order=0,
                raw_points=raw_points,
                points=canonical_quad(raw_points),
                score=0.9,
                clamped=False,
            )
        ]


def _manifest(tmp_path: Path, frame_indices: tuple[int, ...] = (10, 2)) -> Path:
    frame_dir = tmp_path / "frames" / "video2"
    frame_dir.mkdir(parents=True)
    refs: list[FrameRef] = []
    for index in frame_indices:
        path = frame_dir / f"{index}.png"
        Image.new("RGB", (32, 18), color=(index, index, index)).save(path)
        refs.append(
            FrameRef(
                video_id="video2",
                frame_uid=f"video2:{index}",
                keyframe_n=index + 1,
                frame_idx=index,
                pts_time_s=index / 25,
                fps=25.0,
                frame_relpath=f"frames/video2/{index}.png",
                source_image_sha256=sha256_file(path),
                width=32,
                height=18,
            )
        )
    manifest = tmp_path / "frames.jsonl"
    write_jsonl_atomic(manifest, refs)
    return manifest


def _run_all(
    tmp_path: Path,
    artifact_dir: str,
    *,
    run_id: str = "phase1-test",
    config_hash: str = CONFIG_HASH,
    identity: Phase1Identity = IDENTITY,
    tracking_config: TrackingConfig | None = None,
) -> tuple[Path, Path, Path]:
    tracking_config = tracking_config or TrackingConfig()
    manifest = tmp_path / "frames.jsonl"
    detections = tmp_path / artifact_dir / "detections.jsonl"
    trajectories = tmp_path / artifact_dir / "trajectories.jsonl"
    representatives = tmp_path / artifact_dir / "representatives.jsonl"
    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=detections,
        run_id=run_id,
        config_sha256=config_hash,
        detector=FakeDetector(),
        crop_config=CropConfig(),
        identity=identity,
        tracking_config=tracking_config,
    )
    run_tracking(
        detections=detections,
        output=trajectories,
        run_id=run_id,
        config_sha256=config_hash,
        tracking_config=tracking_config,
        identity=identity,
    )
    run_representative_selection(
        trajectories=trajectories,
        output=representatives,
        run_id=run_id,
        config_sha256=config_hash,
        tracking_config=tracking_config,
        identity=identity,
    )
    return detections, trajectories, representatives


def test_same_input_and_config_produce_byte_identical_jsonl_and_receipts(tmp_path: Path) -> None:
    _manifest(tmp_path)
    first = _run_all(tmp_path, "first")
    second = _run_all(tmp_path, "second")

    for first_path, second_path in zip(first, second, strict=True):
        assert first_path.read_bytes() == second_path.read_bytes()
        assert (
            receipt_path_for(first_path).read_bytes() == receipt_path_for(second_path).read_bytes()
        )
    assert verify_detection_artifact(
        output=first[0],
        frame_manifest=tmp_path / "frames.jsonl",
        data_root=tmp_path,
        crop_config=CropConfig(),
        expected_run_id="phase1-test",
        expected_config_sha256=CONFIG_HASH,
        expected_identity=IDENTITY,
    ) == {"frames": 2, "detections": 2}
    assert verify_linked_artifacts(
        detections=first[0],
        trajectories=first[1],
        representatives=first[2],
        expected_run_id="phase1-test",
        expected_config_sha256=CONFIG_HASH,
        expected_identity=IDENTITY,
        tracking_config=TrackingConfig(),
    ) == {"frames": 2, "detections": 2, "trajectories": 1, "representatives": 2}


def test_interrupted_write_resumes_only_receipt_authenticated_prefix(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "artifacts" / "detections.jsonl"
    first = FakeDetector(interrupt_call=2)
    with pytest.raises(KeyboardInterrupt):
        run_detect_crop(
            frame_manifest=manifest,
            data_root=tmp_path,
            output=output,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=first,
            crop_config=CropConfig(),
        )

    partial = output.with_suffix(".jsonl.partial")
    receipt = OcrPhase1Receipt.model_validate_json(
        receipt_path_for(output).read_text(encoding="utf-8")
    )
    assert receipt.status == "running"
    assert receipt.record_counts == {"frames": 1, "detections": 1}
    assert receipt.output_sha256 == sha256_file(partial)

    resumed = FakeDetector()
    counts = run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=output,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=resumed,
        crop_config=CropConfig(),
        resume=True,
    )
    assert counts == {"frames": 2, "detections": 2}
    assert resumed.calls == ["10"]
    assert not partial.exists()


def test_batched_detection_receipt_discards_mid_batch_tail_and_resumes_identically(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, tuple(range(7)))
    tracking = TrackingConfig(detection_receipt_commit_interval_frames=3)
    clean = tmp_path / "clean" / "detections.jsonl"
    interrupted = tmp_path / "interrupted" / "detections.jsonl"
    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=clean,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=FakeDetector(),
        crop_config=CropConfig(),
        tracking_config=tracking,
    )

    # Three records form the first durable batch. The next two records are only
    # an uncommitted tail when inference crashes while starting record six.
    with pytest.raises(KeyboardInterrupt):
        run_detect_crop(
            frame_manifest=manifest,
            data_root=tmp_path,
            output=interrupted,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=FakeDetector(interrupt_call=6),
            crop_config=CropConfig(),
            tracking_config=tracking,
        )
    partial = interrupted.with_suffix(".jsonl.partial")
    receipt = OcrPhase1Receipt.model_validate_json(
        receipt_path_for(interrupted).read_text(encoding="utf-8")
    )
    assert receipt.committed_records == 3
    assert receipt.record_counts == {"frames": 3, "detections": 3}
    assert partial.stat().st_size > receipt.committed_bytes

    resumed = FakeDetector()
    counts = run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=interrupted,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=resumed,
        crop_config=CropConfig(),
        tracking_config=tracking,
        resume=True,
    )
    assert counts == {"frames": 7, "detections": 7}
    assert resumed.calls == ["3", "4", "5", "6"]
    assert interrupted.read_bytes() == clean.read_bytes()
    assert receipt_path_for(interrupted).read_bytes() == receipt_path_for(clean).read_bytes()


def test_batched_detection_receipt_commits_final_short_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path, tuple(range(7)))
    running_commits: list[int] = []
    original = phase1_module._write_receipt

    def capture_running_commit(path: Path, receipt: OcrPhase1Receipt, fault=None) -> None:
        if receipt.status == "running":
            running_commits.append(receipt.committed_records)
        original(path, receipt, fault)

    monkeypatch.setattr(phase1_module, "_write_receipt", capture_running_commit)
    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=tmp_path / "detections.jsonl",
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=FakeDetector(),
        crop_config=CropConfig(),
        tracking_config=TrackingConfig(detection_receipt_commit_interval_frames=3),
    )

    # Initial empty receipt, two complete batches, then the one-frame final batch.
    assert running_commits == [0, 3, 6, 7]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_detection_receipt_commit_interval_requires_positive_true_integer(value: object) -> None:
    with pytest.raises(ValueError, match="tracking counts"):
        TrackingConfig(detection_receipt_commit_interval_frames=value)  # type: ignore[arg-type]


def test_resume_rejects_input_config_model_and_output_tamper(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, (2,))

    def completed(name: str) -> Path:
        output = tmp_path / name / "detections.jsonl"
        run_detect_crop(
            frame_manifest=manifest,
            data_root=tmp_path,
            output=output,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=FakeDetector(),
            crop_config=CropConfig(),
        )
        return output

    config_output = completed("config")
    with pytest.raises(ValueError, match="identity drift"):
        run_detect_crop(
            frame_manifest=manifest,
            data_root=tmp_path,
            output=config_output,
            run_id="phase1-test",
            config_sha256="b" * 64,
            detector=None,
            crop_config=CropConfig(),
            resume=True,
        )

    model_output = completed("model")
    with pytest.raises(ValueError, match="identity drift"):
        run_detect_crop(
            frame_manifest=manifest,
            data_root=tmp_path,
            output=model_output,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=None,
            crop_config=CropConfig(),
            identity=Phase1Identity(
                detector_revision="c" * 64,
                detector_tree_sha256="d" * 64,
                runtime_identity_sha256="e" * 64,
            ),
            resume=True,
        )

    output_tamper = completed("output")
    with output_tamper.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="byte size|checksum differs"):
        run_detect_crop(
            frame_manifest=manifest,
            data_root=tmp_path,
            output=output_tamper,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=None,
            crop_config=CropConfig(),
            resume=True,
        )

    input_output = completed("input")
    with manifest.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="identity drift"):
        run_detect_crop(
            frame_manifest=manifest,
            data_root=tmp_path,
            output=input_output,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=None,
            crop_config=CropConfig(),
            resume=True,
        )


def test_verify_rejects_missing_or_duplicate_detection_records_even_with_rehashed_receipt(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "artifacts" / "detections.jsonl"
    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=output,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=FakeDetector(),
        crop_config=CropConfig(),
    )
    lines = output.read_text(encoding="utf-8").splitlines()
    output.write_text(lines[0] + "\n", encoding="utf-8")
    raw_receipt = json.loads(receipt_path_for(output).read_text(encoding="utf-8"))
    raw_receipt["output_sha256"] = sha256_file(output)
    raw_receipt["committed_sha256"] = sha256_file(output)
    raw_receipt["committed_bytes"] = output.stat().st_size
    raw_receipt["committed_records"] = 1
    raw_receipt["record_counts"] = {"frames": 1, "detections": 1}
    atomic_write_json(receipt_path_for(output), raw_receipt)
    with pytest.raises(ValueError, match="missing/duplicate"):
        verify_detection_artifact(
            output=output,
            frame_manifest=manifest,
            data_root=tmp_path,
            crop_config=CropConfig(),
            expected_run_id="phase1-test",
            expected_config_sha256=CONFIG_HASH,
            expected_identity=IDENTITY,
        )

    output.write_text(lines[0] + "\n" + lines[0] + "\n", encoding="utf-8")
    raw_receipt["output_sha256"] = sha256_file(output)
    raw_receipt["committed_sha256"] = sha256_file(output)
    raw_receipt["committed_bytes"] = output.stat().st_size
    raw_receipt["committed_records"] = 2
    raw_receipt["record_counts"] = {"frames": 2, "detections": 2}
    atomic_write_json(receipt_path_for(output), raw_receipt)
    with pytest.raises(ValueError, match="valid frame prefix"):
        verify_detection_artifact(
            output=output,
            frame_manifest=manifest,
            data_root=tmp_path,
            crop_config=CropConfig(),
            expected_run_id="phase1-test",
            expected_config_sha256=CONFIG_HASH,
            expected_identity=IDENTITY,
        )


def test_uncommitted_partial_tail_is_truncated_and_resumed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "artifacts" / "detections.jsonl"
    with pytest.raises(KeyboardInterrupt):
        run_detect_crop(
            frame_manifest=manifest,
            data_root=tmp_path,
            output=output,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=FakeDetector(interrupt_call=2),
            crop_config=CropConfig(),
        )
    partial = output.with_suffix(".jsonl.partial")
    with partial.open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    counts = run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=output,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=FakeDetector(),
        crop_config=CropConfig(),
        resume=True,
    )
    assert counts == {"frames": 2, "detections": 2}
    assert "{}" not in output.read_text(encoding="utf-8")


def test_frame_preflight_rejects_duplicate_checksum_dimension_and_corrupt_input(
    tmp_path: Path,
) -> None:
    duplicate_root = tmp_path / "duplicate"
    duplicate_manifest = _manifest(duplicate_root, (2,))
    line = duplicate_manifest.read_text(encoding="utf-8")
    duplicate_manifest.write_text(line + line, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate frame_uid"):
        load_frame_manifest(duplicate_manifest, duplicate_root)

    checksum_root = tmp_path / "checksum"
    checksum_manifest = _manifest(checksum_root, (2,))
    Image.new("RGB", (32, 18), "red").save(checksum_root / "frames" / "video2" / "2.png")
    with pytest.raises(ValueError, match="checksum drift"):
        ref = FrameRef.model_validate_json(checksum_manifest.read_text(encoding="utf-8"))
        decode_canonical_frame(ref, checksum_root / ref.frame_relpath)

    dimension_root = tmp_path / "dimension"
    dimension_manifest = _manifest(dimension_root, (2,))
    dimension_path = dimension_root / "frames" / "video2" / "2.png"
    Image.new("RGB", (31, 18), "white").save(dimension_path)
    raw = json.loads(dimension_manifest.read_text(encoding="utf-8"))
    raw["source_image_sha256"] = sha256_file(dimension_path)
    write_jsonl_atomic(dimension_manifest, [raw])
    with pytest.raises(ValueError, match="dimensions"):
        ref = FrameRef.model_validate_json(dimension_manifest.read_text(encoding="utf-8"))
        decode_canonical_frame(ref, dimension_root / ref.frame_relpath)

    corrupt_root = tmp_path / "corrupt"
    corrupt_manifest = _manifest(corrupt_root, (2,))
    corrupt_path = corrupt_root / "frames" / "video2" / "2.png"
    corrupt_path.write_bytes(b"not an image")
    raw = json.loads(corrupt_manifest.read_text(encoding="utf-8"))
    raw["source_image_sha256"] = sha256_file(corrupt_path)
    write_jsonl_atomic(corrupt_manifest, [raw])
    with pytest.raises(ValueError, match="corrupt"):
        ref = FrameRef.model_validate_json(corrupt_manifest.read_text(encoding="utf-8"))
        decode_canonical_frame(ref, corrupt_root / ref.frame_relpath)
    with pytest.raises(ValueError, match="corrupt"):
        validate_frame_sources(corrupt_manifest, corrupt_root)


class SnapshotDetector:
    def __init__(self, mutate_path: Path | None = None) -> None:
        self.seen: np.ndarray | None = None
        self.mutate_path = mutate_path

    def detect(self, image_bgr: np.ndarray, *, width: int, height: int) -> list[DetectorPolygon]:
        self.seen = image_bgr
        assert image_bgr.dtype == np.uint8
        assert image_bgr.flags.writeable is False
        if self.mutate_path is not None:
            Image.new("RGB", (width, height), "black").save(self.mutate_path)
        bottom = float(min(height - 3, 12))
        return [
            DetectorPolygon(
                source_order=0,
                raw_points=(
                    (2.0, 2.0),
                    (width - 3.0, 2.0),
                    (width - 3.0, bottom),
                    (2.0, bottom),
                ),
                points=(
                    (2.0, 2.0),
                    (width - 3.0, 2.0),
                    (width - 3.0, bottom),
                    (2.0, bottom),
                ),
                score=0.8,
                clamped=False,
            )
        ]


def _single_ref_manifest(
    root: Path, image_path: Path, *, width: int, height: int
) -> tuple[Path, FrameRef]:
    ref = FrameRef(
        video_id="video1",
        frame_uid="video1:0",
        keyframe_n=1,
        frame_idx=0,
        pts_time_s=0,
        fps=25,
        frame_relpath=image_path.relative_to(root).as_posix(),
        source_image_sha256=sha256_file(image_path),
        width=width,
        height=height,
    )
    manifest = root / "frames.jsonl"
    write_jsonl_atomic(manifest, [ref])
    return manifest, ref


def test_exif_orientation_uses_one_canonical_pixel_space_for_detector_and_crop(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frames" / "video1" / "0.jpg"
    image_path.parent.mkdir(parents=True)
    image = Image.new("RGB", (40, 20), "red")
    image.paste("blue", (20, 0, 40, 20))
    exif = Image.Exif()
    exif[274] = 6
    image.save(image_path, exif=exif)
    manifest, ref = _single_ref_manifest(tmp_path, image_path, width=20, height=40)
    detector = SnapshotDetector()
    output = tmp_path / "artifacts" / "detections.jsonl"

    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=output,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=detector,
        crop_config=CropConfig(),
    )
    snapshot = decode_canonical_frame(ref, image_path)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert detector.seen is not None
    assert detector.seen.shape == (40, 20, 3)
    assert np.array_equal(detector.seen, snapshot.bgr)
    assert record["width"] == 20 and record["height"] == 40
    assert record["canonical_image_sha256"] == snapshot.canonical_image_sha256
    assert record["detections"][0]["crop"]["output_width"] > 0


def test_16_bit_png_is_rejected_and_source_mutation_cannot_complete_receipt(
    tmp_path: Path,
) -> None:
    sixteen_root = tmp_path / "sixteen"
    image_path = sixteen_root / "frames" / "video1" / "0.png"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(np.full((20, 30), 1024, dtype=np.uint16), mode="I;16").save(image_path)
    manifest, _ = _single_ref_manifest(sixteen_root, image_path, width=30, height=20)
    with pytest.raises(ValueError, match="unsupported source image mode"):
        run_detect_crop(
            frame_manifest=manifest,
            data_root=sixteen_root,
            output=sixteen_root / "detections.jsonl",
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=FakeDetector(),
            crop_config=CropConfig(),
        )

    changed_root = tmp_path / "changed"
    changed_manifest = _manifest(changed_root, (2,))
    load_frame_manifest(changed_manifest, changed_root)
    changed_path = changed_root / "frames" / "video2" / "2.png"
    Image.new("RGB", (32, 18), "red").save(changed_path)
    with pytest.raises(ValueError, match="checksum drift"):
        run_detect_crop(
            frame_manifest=changed_manifest,
            data_root=changed_root,
            output=changed_root / "detections.jsonl",
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=FakeDetector(),
            crop_config=CropConfig(),
        )

    during_root = tmp_path / "during"
    during_manifest = _manifest(during_root, (2,))
    during_path = during_root / "frames" / "video2" / "2.png"
    with pytest.raises(ValueError, match="changed during detector/crop use"):
        run_detect_crop(
            frame_manifest=during_manifest,
            data_root=during_root,
            output=during_root / "detections.jsonl",
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=SnapshotDetector(during_path),
            crop_config=CropConfig(),
        )


@pytest.mark.parametrize(
    "mutation_boundary",
    ("after_final_rename_before_receipt", "after_completed_receipt_before_final_verification"),
)
def test_detection_rechecks_every_source_around_completed_receipt_publication(
    tmp_path: Path, mutation_boundary: str
) -> None:
    manifest = _manifest(tmp_path, (2,))
    source_image = tmp_path / "frames" / "video2" / "2.png"

    def mutate_source(boundary: str) -> None:
        if boundary == mutation_boundary:
            Image.new("RGB", (32, 18), "magenta").save(source_image)

    with pytest.raises(ValueError, match="source image checksum drift"):
        run_detect_crop(
            frame_manifest=manifest,
            data_root=tmp_path,
            output=tmp_path / "detections.jsonl",
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=FakeDetector(),
            crop_config=CropConfig(),
            fault_injector=mutate_source,
        )


def _rehash_receipt(path: Path, *, input_hash: str | None = None) -> None:
    receipt_path = receipt_path_for(path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    receipt["output_sha256"] = digest
    receipt["committed_sha256"] = digest
    receipt["committed_bytes"] = path.stat().st_size
    if input_hash is not None:
        receipt["input_artifact_sha256"] = input_hash
    atomic_write_json(receipt_path, receipt)


@pytest.mark.parametrize("tamper", ["score", "polygon", "sharpness", "visual_hash", "path"])
def test_semantic_verifier_rejects_rehashed_detection_tamper(tmp_path: Path, tamper: str) -> None:
    root = tmp_path / tamper
    _manifest(root)
    detections, trajectories, representatives = _run_all(root, "artifacts")
    records = [json.loads(line) for line in detections.read_text(encoding="utf-8").splitlines()]
    detection = records[0]["detections"][0]
    if tamper == "score":
        detection["detector_score"] = 0.8
    elif tamper == "polygon":
        detection["polygon_xy"]["points"][0][0] += 0.25
    elif tamper == "sharpness":
        detection["crop"]["sharpness"] += 1
    elif tamper == "visual_hash":
        original = detection["crop"]["visual_hash"]
        detection["crop"]["visual_hash"] = "f" * 16 if original != "f" * 16 else "0" * 16
    else:
        records[0]["frame_relpath"] = "../escape.png"
    detections.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    _rehash_receipt(detections)

    if tamper == "score":
        _rehash_receipt(trajectories, input_hash=sha256_file(detections))
        with pytest.raises(ValueError):
            verify_linked_artifacts(
                detections=detections,
                trajectories=trajectories,
                representatives=representatives,
                expected_run_id="phase1-test",
                expected_config_sha256=CONFIG_HASH,
                expected_identity=IDENTITY,
                tracking_config=TrackingConfig(),
            )
    else:
        with pytest.raises(ValueError):
            verify_detection_artifact(
                output=detections,
                frame_manifest=root / "frames.jsonl",
                data_root=root,
                crop_config=CropConfig(),
                expected_run_id="phase1-test",
                expected_config_sha256=CONFIG_HASH,
                expected_identity=IDENTITY,
            )


@pytest.mark.parametrize("artifact", ["trajectory", "representative"])
def test_linked_verifier_rejects_rehashed_member_binding_tamper(
    tmp_path: Path, artifact: str
) -> None:
    root = tmp_path / artifact
    _manifest(root)
    detections, trajectories, representatives = _run_all(root, "artifacts")
    target = trajectories if artifact == "trajectory" else representatives
    value = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
    if artifact == "trajectory":
        value["members"][0]["detector_score"] = 0.8
    else:
        value["detector_score"] = 0.8
    lines = target.read_text(encoding="utf-8").splitlines()
    target.write_text(
        json.dumps(value, separators=(",", ":")) + "\n" + "\n".join(lines[1:]) + "\n",
        encoding="utf-8",
    )
    _rehash_receipt(target)
    if artifact == "trajectory":
        _rehash_receipt(representatives, input_hash=sha256_file(trajectories))
    with pytest.raises(ValueError):
        verify_linked_artifacts(
            detections=detections,
            trajectories=trajectories,
            representatives=representatives,
            expected_run_id="phase1-test",
            expected_config_sha256=CONFIG_HASH,
            expected_identity=IDENTITY,
            tracking_config=TrackingConfig(),
        )


@pytest.mark.parametrize(
    "event",
    [
        "after_partial_create_before_receipt",
        "after_receipt_temp_fsync_before_rename",
        "after_record_fsync_before_receipt",
        "after_final_rename_before_receipt",
    ],
)
def test_detect_fault_windows_resume_twice_to_clean_bytes(tmp_path: Path, event: str) -> None:
    manifest = _manifest(tmp_path)
    clean = tmp_path / "clean" / "detections.jsonl"
    faulted = tmp_path / event / "detections.jsonl"
    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=clean,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=FakeDetector(),
        crop_config=CropConfig(),
    )
    injected = False

    def fail_once(current: str) -> None:
        nonlocal injected
        if current == event and not injected:
            injected = True
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        run_detect_crop(
            frame_manifest=manifest,
            data_root=tmp_path,
            output=faulted,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=FakeDetector(),
            crop_config=CropConfig(),
            fault_injector=fail_once,
        )
    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=faulted,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=FakeDetector(),
        crop_config=CropConfig(),
        resume=True,
    )
    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=faulted,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=None,
        crop_config=CropConfig(),
        resume=True,
    )
    assert faulted.read_bytes() == clean.read_bytes()
    assert receipt_path_for(faulted).read_bytes() == receipt_path_for(clean).read_bytes()


def test_track_and_select_publish_faults_resume_twice_to_clean_bytes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    detections = tmp_path / "base" / "detections.jsonl"
    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=detections,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=FakeDetector(),
        crop_config=CropConfig(),
    )
    clean_trajectories = tmp_path / "clean" / "trajectories.jsonl"
    fault_trajectories = tmp_path / "fault" / "trajectories.jsonl"
    run_tracking(
        detections=detections,
        output=clean_trajectories,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        tracking_config=TrackingConfig(),
    )

    def crash(_event: str) -> None:
        raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        run_tracking(
            detections=detections,
            output=fault_trajectories,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            tracking_config=TrackingConfig(),
            fault_injector=crash,
        )
    for _ in range(2):
        run_tracking(
            detections=detections,
            output=fault_trajectories,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            tracking_config=TrackingConfig(),
            resume=True,
        )
    assert fault_trajectories.read_bytes() == clean_trajectories.read_bytes()
    assert (
        receipt_path_for(fault_trajectories).read_bytes()
        == receipt_path_for(clean_trajectories).read_bytes()
    )

    clean_representatives = tmp_path / "clean" / "representatives.jsonl"
    fault_representatives = tmp_path / "fault" / "representatives.jsonl"
    run_representative_selection(
        trajectories=clean_trajectories,
        output=clean_representatives,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        tracking_config=TrackingConfig(),
    )
    with pytest.raises(RuntimeError, match="injected crash"):
        run_representative_selection(
            trajectories=clean_trajectories,
            output=fault_representatives,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            tracking_config=TrackingConfig(),
            fault_injector=crash,
        )
    for _ in range(2):
        run_representative_selection(
            trajectories=clean_trajectories,
            output=fault_representatives,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            tracking_config=TrackingConfig(),
            resume=True,
        )
    assert fault_representatives.read_bytes() == clean_representatives.read_bytes()
    assert (
        receipt_path_for(fault_representatives).read_bytes()
        == receipt_path_for(clean_representatives).read_bytes()
    )


@pytest.mark.parametrize(
    "event",
    [
        "after_derived_temp_fsync_before_publish",
        "after_output_publish_before_receipt",
        "after_receipt_temp_fsync_before_rename",
    ],
)
@pytest.mark.parametrize("stage", ["track", "select"])
def test_derived_fault_boundaries_recover_to_clean_bytes(
    tmp_path: Path, stage: str, event: str
) -> None:
    manifest = _manifest(tmp_path)
    detections = tmp_path / "base" / "detections.jsonl"
    trajectories = tmp_path / "base" / "trajectories.jsonl"
    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=detections,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=FakeDetector(),
        crop_config=CropConfig(),
    )
    if stage == "select":
        run_tracking(
            detections=detections,
            output=trajectories,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            tracking_config=TrackingConfig(),
        )
    clean = tmp_path / "clean" / f"{stage}.jsonl"
    faulted = tmp_path / "fault" / f"{stage}.jsonl"
    injected = False

    def fail_once(current: str) -> None:
        nonlocal injected
        if current == event and not injected:
            injected = True
            raise RuntimeError("injected crash")

    def invoke(output: Path, *, resume: bool = False, fault=None) -> None:
        if stage == "track":
            run_tracking(
                detections=detections,
                output=output,
                run_id="phase1-test",
                config_sha256=CONFIG_HASH,
                tracking_config=TrackingConfig(),
                resume=resume,
                fault_injector=fault,
            )
        else:
            run_representative_selection(
                trajectories=trajectories,
                output=output,
                run_id="phase1-test",
                config_sha256=CONFIG_HASH,
                tracking_config=TrackingConfig(),
                resume=resume,
                fault_injector=fault,
            )

    invoke(clean)
    with pytest.raises(RuntimeError, match="injected crash"):
        invoke(faulted, fault=fail_once)
    invoke(faulted, resume=True)
    invoke(faulted, resume=True)
    assert faulted.read_bytes() == clean.read_bytes()
    assert receipt_path_for(faulted).read_bytes() == receipt_path_for(clean).read_bytes()


@pytest.mark.parametrize("corruption", ["truncated", "missing_newline"])
def test_derived_orphan_temp_is_quarantined_and_canonically_rewritten(
    tmp_path: Path, corruption: str
) -> None:
    manifest = _manifest(tmp_path)
    detections = tmp_path / "base" / "detections.jsonl"
    output = tmp_path / "fault" / "trajectories.jsonl"
    clean = tmp_path / "clean" / "trajectories.jsonl"
    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=detections,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=FakeDetector(),
        crop_config=CropConfig(),
    )
    run_tracking(
        detections=detections,
        output=clean,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        tracking_config=TrackingConfig(),
    )

    def crash(current: str) -> None:
        if current == "after_derived_temp_fsync_before_publish":
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError):
        run_tracking(
            detections=detections,
            output=output,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            tracking_config=TrackingConfig(),
            fault_injector=crash,
        )
    temporary = output.with_suffix(output.suffix + ".tmp")
    payload = temporary.read_bytes()
    temporary.write_bytes(
        payload[: len(payload) // 2] if corruption == "truncated" else payload.rstrip(b"\n")
    )
    run_tracking(
        detections=detections,
        output=output,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        tracking_config=TrackingConfig(),
        resume=True,
    )
    assert output.read_bytes() == clean.read_bytes()
    assert temporary.with_suffix(temporary.suffix + ".uncommitted.000001").exists()


def test_derived_final_orphan_requires_exact_canonical_bytes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    detections = tmp_path / "base" / "detections.jsonl"
    output = tmp_path / "trajectories.jsonl"
    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=detections,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=FakeDetector(),
        crop_config=CropConfig(),
    )
    run_tracking(
        detections=detections,
        output=output,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        tracking_config=TrackingConfig(),
    )
    receipt_path_for(output).unlink()
    output.write_bytes(output.read_bytes().rstrip(b"\n"))
    with pytest.raises(ValueError, match="canonical replay bytes"):
        run_tracking(
            detections=detections,
            output=output,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            tracking_config=TrackingConfig(),
            resume=True,
        )


def test_manifest_and_upstream_toctou_mutation_are_detected_before_receipt(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    detections = tmp_path / "detections.jsonl"

    def mutate_manifest(current: str) -> None:
        if current == "after_final_rename_before_receipt":
            manifest.write_bytes(manifest.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="manifest changed"):
        run_detect_crop(
            frame_manifest=manifest,
            data_root=tmp_path,
            output=detections,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=FakeDetector(),
            crop_config=CropConfig(),
            fault_injector=mutate_manifest,
        )

    stable_root = tmp_path / "stable"
    stable_manifest = _manifest(stable_root)
    stable_detections = stable_root / "detections.jsonl"
    trajectories = stable_root / "trajectories.jsonl"
    run_detect_crop(
        frame_manifest=stable_manifest,
        data_root=stable_root,
        output=stable_detections,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=FakeDetector(),
        crop_config=CropConfig(),
    )

    def mutate_upstream(current: str) -> None:
        if current == "after_output_publish_before_receipt":
            stable_detections.write_bytes(stable_detections.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="upstream artifact changed"):
        run_tracking(
            detections=stable_detections,
            output=trajectories,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            tracking_config=TrackingConfig(),
            fault_injector=mutate_upstream,
        )


def test_explicit_shard_bounds_reject_oversized_detection_work(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="maximum_frames_per_shard"):
        run_detect_crop(
            frame_manifest=manifest,
            data_root=tmp_path,
            output=tmp_path / "detections.jsonl",
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=FakeDetector(),
            crop_config=CropConfig(),
            tracking_config=TrackingConfig(maximum_frames_per_shard=1),
        )


class TwoDetectionDetector(FakeDetector):
    def detect(self, image_bgr: np.ndarray, *, width: int, height: int) -> list[DetectorPolygon]:
        first = super().detect(image_bgr, width=width, height=height)[0]
        return [
            first,
            DetectorPolygon(
                source_order=1,
                raw_points=((4.0, 4.0), (26.0, 4.0), (26.0, 14.0), (4.0, 14.0)),
                points=((4.0, 4.0), (26.0, 4.0), (26.0, 14.0), (4.0, 14.0)),
                score=0.8,
                clamped=False,
            ),
        ]


def test_detection_limit_fails_before_commit_and_never_publishes_completed_artifact(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, (2,))
    output = tmp_path / "detections.jsonl"
    for resume in (False, True):
        with pytest.raises(ValueError, match="remaining shard detection capacity before crop"):
            run_detect_crop(
                frame_manifest=manifest,
                data_root=tmp_path,
                output=output,
                run_id="phase1-test",
                config_sha256=CONFIG_HASH,
                detector=TwoDetectionDetector(),
                crop_config=CropConfig(),
                tracking_config=TrackingConfig(maximum_detections_per_shard=1),
                shard_id="shard-test",
                resume=resume,
            )
    receipt = OcrPhase1Receipt.model_validate_json(
        receipt_path_for(output).read_text(encoding="utf-8")
    )
    assert receipt.status == "running"
    assert receipt.record_counts == {"frames": 0, "detections": 0}


def test_per_frame_detection_cap_fails_before_any_crop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path, (2,))
    output = tmp_path / "detections.jsonl"
    crop_calls = 0

    def forbidden_crop(*_args, **_kwargs):
        nonlocal crop_calls
        crop_calls += 1
        raise AssertionError("crop must not run")

    monkeypatch.setattr(phase1_module, "encode_crop", forbidden_crop)
    with pytest.raises(ValueError, match="maximum_detections_per_frame before crop"):
        run_detect_crop(
            frame_manifest=manifest,
            data_root=tmp_path,
            output=output,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=TwoDetectionDetector(),
            crop_config=CropConfig(),
            tracking_config=TrackingConfig(maximum_detections_per_frame=1),
        )
    assert crop_calls == 0
    assert not output.exists()


def _multi_video_manifest(root: Path, counts: dict[str, int]) -> Path:
    refs = []
    for video_id, count in counts.items():
        for frame_idx in range(count):
            path = root / "frames" / video_id / f"{frame_idx}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 18), (frame_idx + 1,) * 3).save(path)
            refs.append(
                FrameRef(
                    video_id=video_id,
                    frame_uid=f"{video_id}:{frame_idx}",
                    keyframe_n=frame_idx + 1,
                    frame_idx=frame_idx,
                    pts_time_s=frame_idx / 25,
                    fps=25,
                    frame_relpath=path.relative_to(root).as_posix(),
                    source_image_sha256=sha256_file(path),
                    width=32,
                    height=18,
                )
            )
    manifest = root / "global-frames.jsonl"
    write_jsonl_atomic(manifest, list(reversed(refs)))
    return manifest


def test_global_whole_video_shards_have_exact_coverage_and_unique_trajectories(
    tmp_path: Path,
) -> None:
    source = _multi_video_manifest(tmp_path, {"video1": 3, "video2": 2, "video3": 1})
    shard_tracking = TrackingConfig(
        maximum_frames_per_shard=3,
        maximum_detections_per_shard=10,
    )
    global_path, global_manifest = plan_frame_shards(
        source_manifest=source,
        output_dir=tmp_path / "shards",
        config_sha256=CONFIG_HASH,
        tracking_config=shard_tracking,
    )
    assert [item.frame_count for item in global_manifest.shards] == [3, 3]
    owners = {
        video_id: shard.shard_id for shard in global_manifest.shards for video_id in shard.video_ids
    }
    assert len(owners) == 3

    bundles = {}
    for shard in global_manifest.shards:
        frame_manifest = global_path.parent / shard.manifest_relpath
        detections = tmp_path / "artifacts" / shard.shard_id / "detections.jsonl"
        trajectories = tmp_path / "artifacts" / shard.shard_id / "trajectories.jsonl"
        representatives = tmp_path / "artifacts" / shard.shard_id / "representatives.jsonl"
        run_detect_crop(
            frame_manifest=frame_manifest,
            data_root=tmp_path,
            output=detections,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=FakeDetector(),
            crop_config=CropConfig(),
            tracking_config=shard_tracking,
            shard_id=shard.shard_id,
            shard_manifest_sha256=shard.manifest_sha256,
        )
        receipt = OcrPhase1Receipt.model_validate_json(
            receipt_path_for(detections).read_text(encoding="utf-8")
        )
        assert (receipt.shard_id, receipt.shard_manifest_sha256) == (
            shard.shard_id,
            shard.manifest_sha256,
        )
        run_tracking(
            detections=detections,
            output=trajectories,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            tracking_config=shard_tracking,
        )
        run_representative_selection(
            trajectories=trajectories,
            output=representatives,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            tracking_config=shard_tracking,
        )
        bundles[shard.shard_id] = OcrShardArtifactBundle(
            detections=detections,
            trajectories=trajectories,
            representatives=representatives,
        )

    assert verify_global_shards(
        source_manifest=source,
        global_manifest=global_path,
        expected_config_sha256=CONFIG_HASH,
        tracking_config=shard_tracking,
        expected_run_id="phase1-test",
        expected_identity=IDENTITY,
        shard_bundles=bundles,
        data_root=tmp_path,
        crop_config=CropConfig(),
    ) == {"shards": 2, "videos": 3, "frames": 6, "trajectories": 3}

    global_receipt = global_path.with_suffix(global_path.suffix + ".receipt.json")
    original_global_receipt = global_receipt.read_bytes()

    def mutate_after_structural(boundary: str) -> None:
        if boundary == "after_structural_verification":
            global_receipt.write_bytes(original_global_receipt + b" ")

    with pytest.raises(ValueError, match="global verification input changed"):
        verify_global_shards(
            source_manifest=source,
            global_manifest=global_path,
            expected_config_sha256=CONFIG_HASH,
            tracking_config=shard_tracking,
            expected_run_id="phase1-test",
            expected_identity=IDENTITY,
            shard_bundles=bundles,
            data_root=tmp_path,
            crop_config=CropConfig(),
            _test_fault_injector=mutate_after_structural,
        )
    global_receipt.write_bytes(original_global_receipt)

    first_bundle = bundles[min(bundles)]
    linked_receipt = receipt_path_for(first_bundle.representatives)
    original_linked_receipt = linked_receipt.read_bytes()

    def mutate_after_linked(boundary: str) -> None:
        if boundary == "after_linked_verification":
            linked_receipt.write_bytes(original_linked_receipt + b" ")

    with pytest.raises(ValueError, match="global verification input changed"):
        verify_global_shards(
            source_manifest=source,
            global_manifest=global_path,
            expected_config_sha256=CONFIG_HASH,
            tracking_config=shard_tracking,
            expected_run_id="phase1-test",
            expected_identity=IDENTITY,
            shard_bundles=bundles,
            data_root=tmp_path,
            crop_config=CropConfig(),
            _test_fault_injector=mutate_after_linked,
        )
    linked_receipt.write_bytes(original_linked_receipt)

    first_source_record = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    source_image = tmp_path / first_source_record["frame_relpath"]
    original_source_image = source_image.read_bytes()

    def mutate_source_after_linked(boundary: str) -> None:
        if boundary == "after_linked_verification":
            source_image.write_bytes(original_source_image + b" ")

    with pytest.raises(ValueError, match="global verification input changed"):
        verify_global_shards(
            source_manifest=source,
            global_manifest=global_path,
            expected_config_sha256=CONFIG_HASH,
            tracking_config=shard_tracking,
            expected_run_id="phase1-test",
            expected_identity=IDENTITY,
            shard_bundles=bundles,
            data_root=tmp_path,
            crop_config=CropConfig(),
            _test_fault_injector=mutate_source_after_linked,
        )
    source_image.write_bytes(original_source_image)

    with pytest.raises(ValueError, match="identity"):
        verify_global_shards(
            source_manifest=source,
            global_manifest=global_path,
            expected_config_sha256=CONFIG_HASH,
            tracking_config=shard_tracking,
            expected_run_id="phase1-test",
            expected_identity=Phase1Identity(runtime_identity_sha256="f" * 64),
            shard_bundles=bundles,
            data_root=tmp_path,
            crop_config=CropConfig(),
        )

    shard_ids = sorted(bundles)
    first_line = bundles[shard_ids[0]].trajectories.read_text(encoding="utf-8").splitlines()[0]
    second_path = bundles[shard_ids[1]].trajectories
    second_records = [
        json.loads(line) for line in second_path.read_text(encoding="utf-8").splitlines()
    ]
    second_records[0]["trajectory_id"] = json.loads(first_line)["trajectory_id"]
    second_path.write_text(
        "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in second_records),
        encoding="utf-8",
    )
    _rehash_receipt(second_path)
    with pytest.raises(ValueError):
        verify_global_shards(
            source_manifest=source,
            global_manifest=global_path,
            expected_config_sha256=CONFIG_HASH,
            tracking_config=shard_tracking,
            expected_run_id="phase1-test",
            expected_identity=IDENTITY,
            shard_bundles=bundles,
            data_root=tmp_path,
            crop_config=CropConfig(),
        )


def test_shard_planner_rejects_single_video_over_frame_limit(tmp_path: Path) -> None:
    source = _multi_video_manifest(tmp_path, {"video1": 4})
    with pytest.raises(ValueError, match="stateful cross-shard tracking is required"):
        plan_frame_shards(
            source_manifest=source,
            output_dir=tmp_path / "shards",
            config_sha256=CONFIG_HASH,
            tracking_config=TrackingConfig(maximum_frames_per_shard=3),
        )


def test_shard_planner_accepts_relative_source_and_symlink_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _multi_video_manifest(tmp_path, {"video1": 2})
    real_parent = tmp_path / "real-output"
    real_parent.mkdir()
    (tmp_path / "linked-output").symlink_to(real_parent, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    global_path, manifest = plan_frame_shards(
        source_manifest=Path(source.name),
        output_dir=Path("linked-output/shards"),
        config_sha256=CONFIG_HASH,
        tracking_config=TrackingConfig(maximum_frames_per_shard=2),
    )

    assert global_path.parent == (real_parent / "shards").resolve()
    assert len(manifest.shards) == 1


def test_shard_planner_performs_contract_validation_before_any_output(
    tmp_path: Path,
) -> None:
    source = _multi_video_manifest(tmp_path, {"video1": 1})
    output = tmp_path / "invalid-plan"

    with pytest.raises(ValueError):
        plan_frame_shards(
            source_manifest=source,
            output_dir=output,
            config_sha256="not-a-sha256",
            tracking_config=TrackingConfig(),
        )

    assert not output.exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS /var alias regression")
def test_shard_planner_treats_var_and_private_var_as_same_canonical_tree(tmp_path: Path) -> None:
    source = _multi_video_manifest(tmp_path, {"video1": 1})
    private_prefix = "/private/var/"
    if private_prefix not in str(source.resolve()):
        pytest.skip("pytest temp directory is not under /private/var")
    source_alias = Path(str(source.resolve()).replace(private_prefix, "/var/", 1))
    output_private = tmp_path / "alias-shards"
    output_alias = Path(str(output_private).replace(private_prefix, "/var/", 1))

    global_path, _ = plan_frame_shards(
        source_manifest=source_alias,
        output_dir=output_alias,
        config_sha256=CONFIG_HASH,
        tracking_config=TrackingConfig(),
    )

    assert global_path.parent == output_private.resolve()


def test_global_shard_verifier_rejects_rehashed_missing_frame(tmp_path: Path) -> None:
    source = _multi_video_manifest(tmp_path, {"video1": 2, "video2": 1})
    global_path, manifest = plan_frame_shards(
        source_manifest=source,
        output_dir=tmp_path / "shards",
        config_sha256=CONFIG_HASH,
        tracking_config=TrackingConfig(maximum_frames_per_shard=2),
    )
    raw_global = json.loads(global_path.read_text(encoding="utf-8"))
    shard = raw_global["shards"][0]
    shard_path = global_path.parent / shard["manifest_relpath"]
    lines = shard_path.read_text(encoding="utf-8").splitlines()
    shard_path.write_text(lines[0] + "\n", encoding="utf-8")
    shard["manifest_sha256"] = sha256_file(shard_path)
    shard["frame_uids"] = shard["frame_uids"][:1]
    shard["frame_count"] = 1
    atomic_write_json(global_path, raw_global)
    receipt_path = global_path.with_suffix(global_path.suffix + ".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["global_manifest_sha256"] = sha256_file(global_path)
    atomic_write_json(receipt_path, receipt)

    assert manifest.shards[0].frame_count == 2
    with pytest.raises(ValueError, match="missing source frames"):
        verify_global_shard_structure(
            source_manifest=source,
            global_manifest=global_path,
            expected_config_sha256=CONFIG_HASH,
            tracking_config=TrackingConfig(maximum_frames_per_shard=2),
        )


def test_shard_planner_source_collision_is_non_destructive(tmp_path: Path) -> None:
    output = tmp_path / "shards"
    output.mkdir()
    source = _multi_video_manifest(output, {"video1": 1})
    collision = output / "shard-000001.frames.jsonl"
    source.replace(collision)
    before = collision.read_bytes()
    with pytest.raises(ValueError, match="target collides with source"):
        plan_frame_shards(
            source_manifest=collision,
            output_dir=output,
            config_sha256=CONFIG_HASH,
            tracking_config=TrackingConfig(),
        )
    assert collision.read_bytes() == before


def test_global_manifest_limit_tamper_is_rejected_even_with_rehashed_receipt(
    tmp_path: Path,
) -> None:
    source = _multi_video_manifest(tmp_path, {"video1": 2})
    tracking = TrackingConfig(maximum_frames_per_shard=2)
    global_path, _ = plan_frame_shards(
        source_manifest=source,
        output_dir=tmp_path / "shards",
        config_sha256=CONFIG_HASH,
        tracking_config=tracking,
    )
    payload = json.loads(global_path.read_text(encoding="utf-8"))
    payload["maximum_frames_per_shard"] = 3
    atomic_write_json(global_path, payload)
    receipt_path = global_path.with_suffix(global_path.suffix + ".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["global_manifest_sha256"] = sha256_file(global_path)
    atomic_write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="identity drift"):
        verify_global_shard_structure(
            source_manifest=source,
            global_manifest=global_path,
            expected_config_sha256=CONFIG_HASH,
            tracking_config=tracking,
        )


def test_five_repeated_derived_crashes_use_collision_free_quarantine(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    detections = tmp_path / "base" / "detections.jsonl"
    trajectories = tmp_path / "fault" / "trajectories.jsonl"
    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=detections,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=FakeDetector(),
        crop_config=CropConfig(),
    )

    def crash(boundary: str) -> None:
        if boundary == "after_derived_temp_fsync_before_publish":
            raise RuntimeError("repeat")

    for attempt in range(5):
        with pytest.raises(RuntimeError, match="repeat"):
            run_tracking(
                detections=detections,
                output=trajectories,
                run_id="phase1-test",
                config_sha256=CONFIG_HASH,
                tracking_config=TrackingConfig(),
                resume=attempt > 0,
                fault_injector=crash,
            )
    run_tracking(
        detections=detections,
        output=trajectories,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        tracking_config=TrackingConfig(),
        resume=True,
    )
    temporary = trajectories.with_suffix(trajectories.suffix + ".tmp")
    assert all(
        temporary.with_suffix(temporary.suffix + f".uncommitted.{index:06d}").is_file()
        for index in range(1, 6)
    )


def test_torn_receipt_temps_are_quarantined_and_recovered(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    detections = tmp_path / "detect" / "detections.jsonl"
    partial = detections.with_suffix(detections.suffix + ".partial")
    receipt_temporary = receipt_path_for(detections).with_suffix(
        receipt_path_for(detections).suffix + ".tmp"
    )
    partial.parent.mkdir(parents=True)
    partial.write_text("unauthenticated bytes", encoding="utf-8")
    receipt_temporary.write_text("{", encoding="utf-8")
    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=detections,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=FakeDetector(),
        crop_config=CropConfig(),
        resume=True,
    )
    assert receipt_temporary.with_suffix(receipt_temporary.suffix + ".uncommitted.000001").is_file()
    assert partial.with_suffix(partial.suffix + ".uncommitted.000001").is_file()

    clean = tmp_path / "clean" / "trajectories.jsonl"
    run_tracking(
        detections=detections,
        output=clean,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        tracking_config=TrackingConfig(),
    )
    recovered = tmp_path / "derived" / "trajectories.jsonl"
    recovered.parent.mkdir(parents=True)
    recovered.write_bytes(clean.read_bytes())
    derived_receipt_temp = receipt_path_for(recovered).with_suffix(
        receipt_path_for(recovered).suffix + ".tmp"
    )
    derived_receipt_temp.write_bytes(b"")
    run_tracking(
        detections=detections,
        output=recovered,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        tracking_config=TrackingConfig(),
        resume=True,
    )
    assert recovered.read_bytes() == clean.read_bytes()
    assert derived_receipt_temp.with_suffix(
        derived_receipt_temp.suffix + ".uncommitted.000001"
    ).is_file()


@pytest.mark.parametrize("stage", ["detect", "track"])
def test_final_verification_detects_fault_mutation_after_completed_receipt(
    tmp_path: Path, stage: str
) -> None:
    manifest = _manifest(tmp_path)
    detections = tmp_path / "detections.jsonl"

    def mutate_detect(boundary: str) -> None:
        if boundary == "after_completed_receipt_before_final_verification":
            detections.write_bytes(detections.read_bytes() + b"tamper")

    if stage == "detect":
        with pytest.raises(ValueError, match="byte size|checksum"):
            run_detect_crop(
                frame_manifest=manifest,
                data_root=tmp_path,
                output=detections,
                run_id="phase1-test",
                config_sha256=CONFIG_HASH,
                detector=FakeDetector(),
                crop_config=CropConfig(),
                fault_injector=mutate_detect,
            )
        return
    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=detections,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=FakeDetector(),
        crop_config=CropConfig(),
    )
    trajectories = tmp_path / "trajectories.jsonl"

    def mutate_track(boundary: str) -> None:
        if boundary == "after_completed_receipt_before_final_verification":
            trajectories.write_bytes(trajectories.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="byte size|checksum"):
        run_tracking(
            detections=detections,
            output=trajectories,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            tracking_config=TrackingConfig(),
            fault_injector=mutate_track,
        )


@pytest.mark.parametrize("pipeline_stage", ("detect", "track", "select"))
@pytest.mark.parametrize(
    ("identity_field", "replacement"),
    (
        ("run_id", "forged-run"),
        ("stage", "detect_crop"),
        ("input_artifact_sha256", "0" * 64),
        ("config_sha256", "0" * 64),
        ("detector_id", "forged-detector"),
        ("detector_revision", "0" * 64),
        ("detector_tree_sha256", "0" * 64),
        ("runtime_identity_sha256", "0" * 64),
        ("shard_id", "forged-shard"),
        ("shard_manifest_sha256", "0" * 64),
        ("resource_limits_sha256", "0" * 64),
    ),
)
def test_each_completed_receipt_identity_field_is_rechecked_after_publication(
    tmp_path: Path,
    pipeline_stage: str,
    identity_field: str,
    replacement: str,
) -> None:
    manifest = _manifest(tmp_path)
    detections = tmp_path / "detections.jsonl"
    trajectories = tmp_path / "trajectories.jsonl"
    representatives = tmp_path / "representatives.jsonl"
    tracking = TrackingConfig()

    def mutate_receipt_for(output: Path):
        def mutate(boundary: str) -> None:
            if boundary != "after_completed_receipt_before_final_verification":
                return
            receipt = receipt_path_for(output)
            value = json.loads(receipt.read_text(encoding="utf-8"))
            if identity_field == "stage" and value[identity_field] == replacement:
                value[identity_field] = "track"
            else:
                value[identity_field] = replacement
            receipt.write_text(json.dumps(value) + "\n", encoding="utf-8")

        return mutate

    if pipeline_stage == "detect":
        with pytest.raises(ValueError):
            run_detect_crop(
                frame_manifest=manifest,
                data_root=tmp_path,
                output=detections,
                run_id="phase1-test",
                config_sha256=CONFIG_HASH,
                detector=FakeDetector(),
                crop_config=CropConfig(),
                tracking_config=tracking,
                fault_injector=mutate_receipt_for(detections),
            )
        return

    run_detect_crop(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=detections,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        detector=FakeDetector(),
        crop_config=CropConfig(),
        tracking_config=tracking,
    )
    if pipeline_stage == "track":
        with pytest.raises(ValueError):
            run_tracking(
                detections=detections,
                output=trajectories,
                run_id="phase1-test",
                config_sha256=CONFIG_HASH,
                tracking_config=tracking,
                fault_injector=mutate_receipt_for(trajectories),
            )
        return

    run_tracking(
        detections=detections,
        output=trajectories,
        run_id="phase1-test",
        config_sha256=CONFIG_HASH,
        tracking_config=tracking,
    )
    with pytest.raises(ValueError):
        run_representative_selection(
            trajectories=trajectories,
            output=representatives,
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            tracking_config=tracking,
            fault_injector=mutate_receipt_for(representatives),
        )


def test_real_model_gate_is_nonzero_when_model_root_is_missing(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    script = Path(__file__).resolve().parents[2] / "scripts" / "gate_ocr_phase1_real_model.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model-root",
            str(tmp_path / "missing-model"),
            "--runtime-cache-root",
            str(runtime),
            "--performance-manifest",
            str(tmp_path / "missing.jsonl"),
            "--evaluation-manifest",
            str(tmp_path / "missing-evaluation.jsonl"),
            "--data-root",
            str(tmp_path),
            "--execution-attestation",
            str(tmp_path / "missing-attestation.json"),
            "--expected-source-commit-sha",
            "1" * 40,
            "--negative-manifest",
            str(tmp_path / "missing-negative.jsonl"),
            "--negative-data-root",
            str(tmp_path),
            "--negative-suite-receipt",
            str(tmp_path / "missing-negative-receipt.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--model-root is missing" in result.stderr


def test_real_model_gate_is_nonzero_when_pilot_dataset_is_missing(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    runtime = tmp_path / "runtime"
    model_root.mkdir()
    runtime.mkdir()
    script = Path(__file__).resolve().parents[2] / "scripts" / "gate_ocr_phase1_real_model.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model-root",
            str(model_root),
            "--runtime-cache-root",
            str(runtime),
            "--performance-manifest",
            str(tmp_path / "missing.jsonl"),
            "--evaluation-manifest",
            str(tmp_path / "missing-evaluation.jsonl"),
            "--data-root",
            str(tmp_path),
            "--execution-attestation",
            str(tmp_path / "missing-attestation.json"),
            "--expected-source-commit-sha",
            "1" * 40,
            "--negative-manifest",
            str(tmp_path / "missing-negative.jsonl"),
            "--negative-data-root",
            str(tmp_path),
            "--negative-suite-receipt",
            str(tmp_path / "missing-negative-receipt.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--performance-manifest is missing" in result.stderr


QUALITY_STRATA = [
    "positive_text",
    "multi_box",
    "horizontal",
    "perspective",
    "clipped_edge",
    "near_vertical",
]


def _quality_frame(index: int) -> FrameRef:
    return FrameRef(
        video_id="pilot",
        frame_uid=f"pilot:{index}",
        keyframe_n=index + 1,
        frame_idx=index,
        pts_time_s=float(index),
        fps=25.0,
        frame_relpath=f"frames/pilot/{index}.png",
        source_image_sha256="1" * 64,
        width=100,
        height=80,
    )


def _ground_truth_payload(
    index: int,
    *,
    frame_strata: list[str] | None = None,
    instance_strata: list[str] | None = None,
) -> dict:
    instance_strata = instance_strata or [
        "positive_text",
        "horizontal",
        "perspective",
        "clipped_edge",
        "near_vertical",
    ]
    return {
        "frame_uid": f"pilot:{index}",
        "strata": ["multi_box"] if frame_strata is None else frame_strata,
        "instances": [
            {
                "instance_id": f"text-{index}-a",
                "polygon_xy": {"points": [[10, 10], [30, 10], [30, 20], [10, 20]]},
                "ignore": False,
                "strata": instance_strata,
            },
            {
                "instance_id": f"text-{index}-b",
                "polygon_xy": {"points": [[40, 40], [60, 40], [60, 50], [40, 50]]},
                "ignore": False,
                "strata": instance_strata,
            },
        ],
    }


def test_old_one_record_for_1000_frame_coverage_contract_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "evaluation.jsonl"
    write_jsonl_atomic(manifest, [_ground_truth_payload(0)])
    frames = {frame.frame_uid: frame for frame in (_quality_frame(i) for i in range(1_000))}

    with pytest.raises(ValueError, match="fewer than 100"):
        load_ground_truth_manifest(manifest, frames, config=DetectionQualityConfig())


@pytest.mark.parametrize(
    "override",
    (
        {"minimum_labeled_frames": 99},
        {"minimum_non_ignored_instances": 199},
        {"minimum_frames_per_stratum": 14},
        {"matching_iou_threshold": 0.49},
        {"minimum_overall_recall": 0.94},
        {"minimum_stratum_recall": 0.89},
        {"minimum_overall_precision": 0.49},
    ),
)
def test_quality_policy_thresholds_cannot_be_lowered(override: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="locked v1"):
        DetectionQualityConfig(**override)


def test_ground_truth_minimum_coverage_contract_and_manifest_mutation(tmp_path: Path) -> None:
    manifest = tmp_path / "evaluation.jsonl"
    frames = {frame.frame_uid: frame for frame in (_quality_frame(i) for i in range(100))}
    write_jsonl_atomic(manifest, [_ground_truth_payload(i) for i in range(100)])

    digest, records = load_ground_truth_manifest(manifest, frames, config=DetectionQualityConfig())
    assert len(records) == 100
    assert sum(not item.ignore for record in records for item in record.instances) == 200
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed during real-model gate"):
        verify_file_unchanged(manifest, digest, label="ground-truth manifest")


def test_ground_truth_missing_strata_coverage_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "evaluation.jsonl"
    frames = {frame.frame_uid: frame for frame in (_quality_frame(i) for i in range(100))}
    write_jsonl_atomic(
        manifest,
        [
            _ground_truth_payload(
                i,
                instance_strata=["positive_text", "horizontal"],
            )
            for i in range(100)
        ],
    )
    with pytest.raises(ValueError, match="strata coverage"):
        load_ground_truth_manifest(manifest, frames, config=DetectionQualityConfig())


@pytest.mark.parametrize(
    "tamper",
    (
        "nan",
        "out_of_bounds",
        "self_intersecting",
        "extra",
        "missing",
        "unknown_stratum",
        "unknown_frame",
    ),
)
def test_ground_truth_polygon_and_schema_are_strict(tmp_path: Path, tamper: str) -> None:
    manifest = tmp_path / f"{tamper}.jsonl"
    frames = {frame.frame_uid: frame for frame in (_quality_frame(i) for i in range(100))}
    records = [_ground_truth_payload(i) for i in range(100)]
    instance = records[0]["instances"][0]
    if tamper == "nan":
        instance["polygon_xy"]["points"][0][0] = float("nan")
    elif tamper == "out_of_bounds":
        instance["polygon_xy"]["points"][0][0] = 101
    elif tamper == "self_intersecting":
        instance["polygon_xy"]["points"] = [[10, 10], [30, 20], [30, 10], [10, 20]]
    elif tamper == "extra":
        instance["unexpected"] = True
    elif tamper == "missing":
        del instance["ignore"]
    elif tamper == "unknown_stratum":
        records[0]["instances"][0]["strata"].append("unknown")
    else:
        records[0]["frame_uid"] = "pilot:999"
    write_jsonl_atomic(manifest, records)

    with pytest.raises(ValueError):
        load_ground_truth_manifest(manifest, frames, config=DetectionQualityConfig())


@pytest.mark.parametrize("duplicate", ("frame", "instance"))
def test_ground_truth_duplicate_identity_is_rejected(tmp_path: Path, duplicate: str) -> None:
    manifest = tmp_path / f"duplicate-{duplicate}.jsonl"
    frames = {frame.frame_uid: frame for frame in (_quality_frame(i) for i in range(100))}
    records = [_ground_truth_payload(i) for i in range(100)]
    if duplicate == "frame":
        records[-1] = copy.deepcopy(records[0])
    else:
        records[1]["instances"][0]["instance_id"] = records[0]["instances"][0]["instance_id"]
    write_jsonl_atomic(manifest, records)

    with pytest.raises(ValueError, match="duplicate"):
        load_ground_truth_manifest(manifest, frames, config=DetectionQualityConfig())


def _quality_quad(left: float, top: float, right: float, bottom: float) -> QuadGeometry:
    return QuadGeometry(points=((left, top), (right, top), (right, bottom), (left, bottom)))


def _quality_prediction(*boxes: QuadGeometry) -> OcrDetectionFrameRecord:
    detections = [
        OcrDetection.model_construct(
            detection_id=f"prediction-{index}",
            source_order=index,
            polygon_xy=box,
        )
        for index, box in enumerate(boxes)
    ]
    return OcrDetectionFrameRecord.model_construct(frame_uid="pilot:0", detections=detections)


def _quality_ground_truth(
    *instances: tuple[str, QuadGeometry, bool] | tuple[str, QuadGeometry, bool, tuple[str, ...]],
) -> GroundTruthFrame:
    frame_strata = ("multi_box",) if sum(not item[2] for item in instances) >= 2 else ()
    default_instance_strata = tuple(name for name in QUALITY_STRATA if name != "multi_box")
    return GroundTruthFrame(
        frame_uid="pilot:0",
        strata=frame_strata,
        instances=tuple(
            GroundTruthInstance(
                instance_id=item[0],
                polygon_xy=item[1],
                ignore=item[2],
                strata=(() if item[2] else item[3] if len(item) == 4 else default_instance_strata),
            )
            for item in instances
        ),
    )


def test_quality_matching_valid_predictions_pass_locked_thresholds() -> None:
    first = _quality_quad(10, 10, 30, 20)
    second = _quality_quad(40, 40, 60, 50)
    metrics = evaluate_detection_quality(
        [_quality_prediction(first, second)],
        [_quality_ground_truth(("first", first, False), ("second", second, False))],
        config=DetectionQualityConfig(),
    )
    enforce_quality_thresholds(metrics, DetectionQualityConfig())
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (2, 0, 0)
    assert all(value == 1.0 for value in metrics["recall_by_stratum"].values())
    assert "status" not in metrics


def test_non_overlapping_box_count_cannot_pass_quality_gate() -> None:
    truth = _quality_quad(10, 10, 30, 20)
    metrics = evaluate_detection_quality(
        [_quality_prediction(_quality_quad(50, 10, 70, 20), _quality_quad(50, 30, 70, 40))],
        [_quality_ground_truth(("truth", truth, False))],
        config=DetectionQualityConfig(),
    )
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (0, 2, 1)
    with pytest.raises(ValueError, match="recall"):
        enforce_quality_thresholds(metrics, DetectionQualityConfig())


def test_quality_metrics_report_tp_fp_fn_and_per_strata_recall() -> None:
    first = _quality_quad(10, 10, 30, 20)
    second = _quality_quad(40, 40, 60, 50)
    metrics = evaluate_detection_quality(
        [_quality_prediction(first, _quality_quad(70, 60, 90, 70))],
        [_quality_ground_truth(("first", first, False), ("second", second, False))],
        config=DetectionQualityConfig(),
    )
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 1, 1)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert set(metrics["recall_by_stratum"].values()) == {0.5}


def test_per_stratum_recall_counts_only_instances_with_that_stratum() -> None:
    horizontal = _quality_quad(10, 10, 30, 20)
    near_vertical = _quality_quad(40, 40, 50, 70)
    ground_truth = _quality_ground_truth(
        ("horizontal", horizontal, False, ("positive_text", "horizontal")),
        ("vertical", near_vertical, False, ("positive_text", "near_vertical")),
    )
    metrics = evaluate_detection_quality(
        [_quality_prediction(horizontal)],
        [ground_truth],
        config=DetectionQualityConfig(),
    )

    assert metrics["recall_by_stratum"]["horizontal"] == 1.0
    assert metrics["recall_by_stratum"]["near_vertical"] == 0.0
    assert metrics["recall_by_stratum"]["positive_text"] == 0.5
    assert metrics["recall_by_stratum"]["multi_box"] == 0.5


def test_real_gate_rejects_detection_mutation_between_verify_and_metric_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "gate_ocr_phase1_real_model.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("real_model_gate_snapshot_test", script)
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    detections = tmp_path / "detections.jsonl"
    detections.write_bytes(b'{"version":"verified"}\n')
    receipt_path_for(detections).write_bytes(b'{"receipt":"verified"}\n')

    def verify_then_mutate() -> None:
        detections.write_bytes(b'{"version":"replacement"}\n')

    with pytest.raises(ValueError, match="changed during real-model gate"):
        gate._verified_detection_snapshot(detections, verify=verify_then_mutate)


def test_duplicate_prediction_does_not_increase_true_positives() -> None:
    truth = _quality_quad(10, 10, 30, 20)
    metrics = evaluate_detection_quality(
        [_quality_prediction(truth, truth)],
        [_quality_ground_truth(("truth", truth, False))],
        config=DetectionQualityConfig(),
    )
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 1, 0)


def test_quality_matching_maximizes_cardinality_deterministically() -> None:
    broad = _quality_quad(0, 0, 20, 10)
    narrow_left = _quality_quad(0, 0, 10, 10)
    narrow_right = _quality_quad(5, 0, 20, 10)
    ground_truth = [
        _quality_ground_truth(("a-broad", broad, False), ("b-left", narrow_left, False))
    ]
    first = evaluate_detection_quality(
        [_quality_prediction(broad, narrow_right)],
        ground_truth,
        config=DetectionQualityConfig(),
    )
    second = evaluate_detection_quality(
        [_quality_prediction(narrow_right, broad)],
        ground_truth,
        config=DetectionQualityConfig(),
    )
    assert (first["tp"], first["fp"], first["fn"]) == (2, 0, 0)
    assert first == second


def test_ignore_region_behavior_is_deterministic() -> None:
    truth = _quality_quad(10, 10, 30, 20)
    ignored = _quality_quad(40, 40, 60, 50)
    extra = _quality_quad(70, 60, 90, 70)
    ground_truth = [_quality_ground_truth(("truth", truth, False), ("ignored", ignored, True))]
    first = evaluate_detection_quality(
        [_quality_prediction(truth, ignored, extra)],
        ground_truth,
        config=DetectionQualityConfig(),
    )
    second = evaluate_detection_quality(
        [_quality_prediction(extra, ignored, truth)],
        ground_truth,
        config=DetectionQualityConfig(),
    )
    assert (first["tp"], first["fp"], first["fn"], first["ignored_predictions"]) == (
        1,
        1,
        0,
        1,
    )
    assert first == second


def _execution_attestation_payload(**updates: object) -> dict:
    payload = {
        "schema_version": "aic26.ocr_phase1.execution_attestation.v1",
        "provider": "kaggle",
        "notebook_kernel_identifier": "owner/kernel",
        "notebook_version_commit_sha": "1" * 40,
        "config_sha256": CONFIG_HASH,
        "detector_revision": IDENTITY.detector_revision,
        "detector_tree_sha256": IDENTITY.detector_tree_sha256,
        "environment_runtime_identity_sha256": IDENTITY.runtime_identity_sha256,
        "internet_enabled": False,
        "accelerator_device": "cpu",
        "created_at": datetime(2026, 8, 19, tzinfo=UTC).isoformat().replace("+00:00", "Z"),
        "approver": "phase1-reviewer",
    }
    payload.update(updates)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _verify_test_attestation(path: Path):
    return load_and_verify_execution_attestation(
        path,
        expected_config_sha256=CONFIG_HASH,
        expected_detector_revision=IDENTITY.detector_revision,
        expected_detector_tree_sha256=IDENTITY.detector_tree_sha256,
        expected_runtime_identity_sha256=IDENTITY.runtime_identity_sha256,
        expected_source_commit_sha="1" * 40,
    )


def test_execution_attestation_is_strict_bound_and_tamper_evident(tmp_path: Path) -> None:
    attestation = tmp_path / "execution-attestation.json"
    atomic_write_json(attestation, _execution_attestation_payload())
    digest, parsed = _verify_test_attestation(attestation)
    assert parsed.internet_enabled is False
    assert digest == sha256_file(attestation)

    tampered = json.loads(attestation.read_text(encoding="utf-8"))
    tampered["approver"] = "tampered"
    atomic_write_json(attestation, tampered)
    with pytest.raises(ValueError, match="invalid strict schema"):
        _verify_test_attestation(attestation)


@pytest.mark.parametrize(
    "case",
    (
        "missing",
        "online",
        "integer_false",
        "config_mismatch",
        "identity_mismatch",
        "commit_mismatch",
        "extra_field",
    ),
)
def test_execution_attestation_missing_online_or_mismatched_is_rejected(
    tmp_path: Path, case: str
) -> None:
    attestation = tmp_path / "execution-attestation.json"
    if case == "missing":
        with pytest.raises(ValueError, match="unavailable"):
            _verify_test_attestation(attestation)
        return
    updates: dict[str, object] = {}
    if case == "online":
        updates["internet_enabled"] = True
    elif case == "integer_false":
        updates["internet_enabled"] = 0
    elif case == "config_mismatch":
        updates["config_sha256"] = "0" * 64
    elif case == "identity_mismatch":
        updates["environment_runtime_identity_sha256"] = "0" * 64
    elif case == "commit_mismatch":
        updates["notebook_version_commit_sha"] = "2" * 40
    payload = _execution_attestation_payload(**updates)
    if case == "extra_field":
        payload["unexpected"] = True
        payload_without_hash = {
            key: value for key, value in payload.items() if key != "payload_sha256"
        }
        payload["payload_sha256"] = hashlib.sha256(
            json.dumps(payload_without_hash, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    atomic_write_json(attestation, payload)
    with pytest.raises(ValueError):
        _verify_test_attestation(attestation)


def _negative_fixture_payload(path: Path, *, mode: str = "I;16") -> dict:
    return {
        "video_id": "negative",
        "frame_uid": "negative:0",
        "keyframe_n": 1,
        "frame_idx": 0,
        "pts_time_s": 0.0,
        "fps": 25.0,
        "frame_relpath": "negative/0.png",
        "source_image_sha256": sha256_file(path),
        "width": 30,
        "height": 20,
        "fixture_id": "unsupported-mode-16bit",
        "expected_error_code": "unsupported_source_mode",
        "expected_reason": f"unsupported source image mode {mode!r}: negative:0",
    }


def test_unsupported_source_only_passes_separate_negative_suite(tmp_path: Path) -> None:
    image_path = tmp_path / "negative" / "0.png"
    image_path.parent.mkdir()
    Image.fromarray(np.full((20, 30), 1024, dtype=np.uint16), mode="I;16").save(image_path)
    fixture = _negative_fixture_payload(image_path)
    negative_manifest = tmp_path / "negative.jsonl"
    receipt_path = tmp_path / "negative.receipt.json"
    write_jsonl_atomic(negative_manifest, [fixture])

    receipt, _baseline = verify_negative_fixture_suite(
        negative_manifest, tmp_path, config_sha256=CONFIG_HASH
    )
    atomic_write_json(receipt_path, receipt.model_dump(mode="json"))
    assert verify_negative_suite_receipt(receipt_path, receipt) == sha256_file(receipt_path)

    successful_manifest = tmp_path / "performance.jsonl"
    frame_fields = set(FrameRef.model_fields)
    write_jsonl_atomic(
        successful_manifest,
        [{key: value for key, value in fixture.items() if key in frame_fields}],
    )
    with pytest.raises(ValueError, match="unsupported source image mode"):
        validate_frame_sources(successful_manifest, tmp_path)

    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_payload["fixture_count"] = 2
    atomic_write_json(receipt_path, receipt_payload)
    with pytest.raises(ValueError, match="mismatch"):
        verify_negative_suite_receipt(receipt_path, receipt)


def test_negative_suite_rejects_unexpectedly_supported_fixture_and_missing_receipt(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "negative" / "0.png"
    image_path.parent.mkdir()
    Image.new("RGB", (30, 20), "white").save(image_path)
    manifest = tmp_path / "negative.jsonl"
    write_jsonl_atomic(manifest, [_negative_fixture_payload(image_path, mode="RGB")])

    with pytest.raises(ValueError, match="unexpectedly accepted"):
        verify_negative_fixture_suite(manifest, tmp_path, config_sha256=CONFIG_HASH)
    with pytest.raises(ValueError, match="unavailable"):
        verify_negative_suite_receipt(tmp_path / "missing-receipt.json", None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_path",
    [r"frames\\video2\\2.png", "C:foo", "C:", "/absolute.png", "../x", "a//b", "a/./b"],
)
def test_manifest_shared_relative_path_contract_fails_during_preflight(
    tmp_path: Path, bad_path: str
) -> None:
    manifest = _manifest(tmp_path, (2,))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["frame_relpath"] = bad_path
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical safe relative path"):
        validate_frame_sources(manifest, tmp_path)


def test_production_detect_route_rejects_fake_identity_injection(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, (2,))
    with pytest.raises(TypeError, match="verified PaddleOcrV6Detector"):
        phase1_module.run_detect_crop(
            frame_manifest=manifest,
            data_root=tmp_path,
            output=tmp_path / "detections.jsonl",
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=FakeDetector(),
            crop_config=CropConfig(),
        )

    forged = object.__new__(PaddleOcrV6Detector)
    object.__setattr__(forged, "_engine", FakeDetector())
    object.__setattr__(
        forged,
        "_verification",
        {
            "detector_id": IDENTITY.detector_id,
            "detector_revision": IDENTITY.detector_revision,
            "detector_tree_sha256": IDENTITY.detector_tree_sha256,
            "runtime_identity_sha256": IDENTITY.runtime_identity_sha256,
            "model_snapshot_verified": True,
        },
    )
    with pytest.raises(TypeError, match="verified PaddleOcrV6Detector"):
        phase1_module.run_detect_crop(
            frame_manifest=manifest,
            data_root=tmp_path,
            output=tmp_path / "forged.jsonl",
            run_id="phase1-test",
            config_sha256=CONFIG_HASH,
            detector=forged,
            crop_config=CropConfig(),
        )


def _load_phase1_cli(monkeypatch: pytest.MonkeyPatch):
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_ocr_phase1.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("run_ocr_phase1_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("command", "provided", "expected"),
    [
        ("detect", {"detections"}, {"detections"}),
        ("track", {"detections", "trajectories"}, {"detections", "trajectories"}),
        ("select", {"trajectories", "representatives"}, {"trajectories", "representatives"}),
        (
            "verify",
            {"detections", "trajectories", "representatives"},
            {"detections", "trajectories", "representatives"},
        ),
    ],
)
def test_cli_stage_paths_are_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    provided: set[str],
    expected: set[str],
) -> None:
    cli = _load_phase1_cli(monkeypatch)
    values = {
        name: (tmp_path / f"{name}.jsonl" if name in provided else None)
        for name in ("detections", "trajectories", "representatives")
    }
    args = argparse.Namespace(command=command, output_root=None, **values)
    assert set(cli._paths(args)) == expected


def test_cli_resume_detects_published_or_partial_stage_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_phase1_cli(monkeypatch)
    output = tmp_path / "artifact.jsonl"
    assert cli._resume_existing_stage(True, output) is False
    output.write_text("published", encoding="utf-8")
    assert cli._resume_existing_stage(True, output) is True
    assert cli._resume_existing_stage(False, output) is False

    committed = OcrPhase1Receipt(
        run_id="phase1-test",
        stage="detect_crop",
        status="running",
        input_artifact_sha256="1" * 64,
        config_sha256="2" * 64,
        detector_revision="3" * 64,
        detector_tree_sha256="4" * 64,
        runtime_identity_sha256="5" * 64,
        resource_limits_sha256=TrackingConfig().resource_limits_sha256,
        shard_id="shard-standalone",
        shard_manifest_sha256="1" * 64,
        record_counts={"frames": 2, "detections": 0},
        output_sha256="6" * 64,
        committed_bytes=10,
        committed_records=2,
        committed_sha256="6" * 64,
    )
    assert cli._detector_required(committed, 2) is False
    assert cli._detector_required(committed, 3) is True


@pytest.mark.parametrize("run_id", [None, False, 7, [], {}])
def test_cli_rejects_non_string_run_id_at_config_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_id: object
) -> None:
    cli = _load_phase1_cli(monkeypatch)
    production_path = (
        Path(__file__).resolve().parents[2] / "configs" / "offline" / "ocr_phase1.yaml"
    )
    config = yaml.safe_load(production_path.read_text(encoding="utf-8"))
    config["run"]["run_id"] = run_id
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty string"):
        cli._settings(argparse.Namespace(config=path))


@pytest.mark.parametrize("profile", [None, "gpu_pinned", "blocked_unverified_runtime"])
def test_cli_rejects_unsupported_execution_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str | None
) -> None:
    cli = _load_phase1_cli(monkeypatch)
    production_path = (
        Path(__file__).resolve().parents[2] / "configs" / "offline" / "ocr_phase1.yaml"
    )
    config = yaml.safe_load(production_path.read_text(encoding="utf-8"))
    config["execution_profile"] = profile
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="supported pinned profile"):
        cli._settings(argparse.Namespace(config=path))


def test_cli_accepts_full_kaggle_gpu_profile_without_inspecting_installed_paddle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_phase1_cli(monkeypatch)
    config_path = (
        Path(__file__).resolve().parents[2] / "configs" / "offline" / "ocr_phase1_kaggle_gpu.yaml"
    )
    config, _, _, run_id, _, identity = cli._settings(argparse.Namespace(config=config_path))

    import aic2026.ocr.detector_only as detector_only_module

    assert config["execution_profile"] == "kaggle_gpu_pinned"
    assert config["model"]["device"] == "gpu:0"
    assert run_id == "ppocrv6-small-det-gpt4o-mini-high-v1-phase1"
    assert (
        identity.runtime_identity_sha256 == detector_only_module.KAGGLE_GPU_RUNTIME_IDENTITY_SHA256
    )


def test_track_select_verify_settings_do_not_inspect_installed_paddle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_phase1_cli(monkeypatch)
    production_path = (
        Path(__file__).resolve().parents[2] / "configs" / "offline" / "ocr_phase1.yaml"
    )

    def forbidden() -> dict[str, str]:
        raise AssertionError("lightweight metadata stage inspected installed Paddle")

    import aic2026.ocr.detector_only as detector_only_module

    monkeypatch.setattr(detector_only_module, "_installed_package_versions", forbidden)
    for command in ("track", "select", "verify"):
        args = argparse.Namespace(config=production_path, command=command)
        _, _, _, run_id, _, identity = cli._settings(args)
        assert run_id
        assert identity.runtime_identity_sha256 == detector_only_module.RUNTIME_IDENTITY_SHA256


@pytest.mark.parametrize("path", ["C:foo", "C:"])
def test_phase1_contract_rejects_windows_drive_relative_paths(path: str) -> None:
    with pytest.raises(ValueError, match="relative"):
        FrameRef(
            video_id="video1",
            frame_uid="video1:0",
            keyframe_n=1,
            frame_idx=0,
            pts_time_s=0,
            fps=25,
            frame_relpath=path,
            source_image_sha256="1" * 64,
            width=32,
            height=18,
        )


@pytest.mark.parametrize(
    "variant", ["run_id", "tracking", "full_config", "missing_model", "changed_model"]
)
def test_cli_verify_rejects_current_config_identity_drift_with_nonzero_exit(
    tmp_path: Path, variant: str
) -> None:
    production_path = (
        Path(__file__).resolve().parents[2] / "configs" / "offline" / "ocr_phase1.yaml"
    )
    production = yaml.safe_load(production_path.read_text(encoding="utf-8"))
    runtime = runtime_identity_from_config(production)
    identity = Phase1Identity(
        detector_id=runtime["detector_id"],
        detector_revision=runtime["detector_revision"],
        detector_tree_sha256=runtime["detector_tree_sha256"],
        runtime_identity_sha256=runtime["runtime_identity_sha256"],
    )
    tracking = TrackingConfig(**production["tracking"])
    _manifest(tmp_path)
    detections, trajectories, representatives = _run_all(
        tmp_path,
        "artifacts",
        run_id=production["run"]["run_id"],
        config_hash=canonical_config_sha256(production),
        identity=identity,
        tracking_config=tracking,
    )
    changed = copy.deepcopy(production)
    if variant == "run_id":
        changed["run"]["run_id"] += "-other"
    elif variant == "tracking":
        changed["tracking"]["max_frame_gap"] -= 1
    elif variant == "full_config":
        changed["seed"] += 1
    elif variant == "missing_model":
        del changed["model"]
    else:
        changed["model"]["candidate_id"] += "-other"
    config_path = tmp_path / f"{variant}.yaml"
    config_path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_ocr_phase1.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "verify",
            "--config",
            str(config_path),
            "--data-root",
            str(tmp_path),
            "--frame-manifest",
            str(tmp_path / "frames.jsonl"),
            "--detections",
            str(detections),
            "--trajectories",
            str(trajectories),
            "--representatives",
            str(representatives),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "verified" not in result.stdout
