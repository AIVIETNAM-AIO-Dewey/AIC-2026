"""Fail-closed deterministic orchestration for OCR Phase 1 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np
from pydantic import BaseModel

from aic2026.common import iter_jsonl, sha256_file
from aic2026.contracts import (
    FrameRef,
    OcrDetection,
    OcrDetectionFrameRecord,
    OcrPhase1Receipt,
    OcrTrajectoryRecord,
    QuadGeometry,
    RawQuadGeometry,
    RepresentativeCropBinding,
)

from .detector_only import (
    DETECTOR_ID,
    DETECTOR_REVISION,
    DETECTOR_TREE_SHA256,
    RUNTIME_IDENTITY_SHA256,
    DetectorPolygon,
    PaddleOcrV6Detector,
    is_production_detector_attested,
)
from .frame_snapshot import decode_canonical_frame
from .geometry import (
    CropConfig,
    canonical_quad,
    clamp_quad,
    encode_crop,
    validate_canonical_quad,
)
from .sharding import _frame_manifest_snapshot
from .tracking import (
    TrackingConfig,
    build_trajectories,
    select_representatives,
    validate_and_sort_detection_frames,
)


class Detector(Protocol):
    def detect(
        self, image_bgr: np.ndarray, *, width: int, height: int
    ) -> list[DetectorPolygon]: ...


FaultInjector = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class Phase1Identity:
    detector_id: str = DETECTOR_ID
    detector_revision: str = DETECTOR_REVISION
    detector_tree_sha256: str = DETECTOR_TREE_SHA256
    runtime_identity_sha256: str = RUNTIME_IDENTITY_SHA256

    def __post_init__(self) -> None:
        for name in (
            "detector_revision",
            "detector_tree_sha256",
            "runtime_identity_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.detector_id != DETECTOR_ID:
            raise ValueError("unsupported detector model ID")


DEFAULT_PHASE1_IDENTITY = Phase1Identity()
DEFAULT_TRACKING_CONFIG = TrackingConfig()


RecordT = TypeVar("RecordT", bound=BaseModel)


def canonical_config_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def receipt_path_for(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".receipt.json")


def _write_receipt(
    path: Path, receipt: OcrPhase1Receipt, fault_injector: FaultInjector | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = (
        json.dumps(
            receipt.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if fault_injector is not None:
        fault_injector("after_receipt_temp_fsync_before_rename")
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_receipt(path: Path) -> OcrPhase1Receipt:
    try:
        return OcrPhase1Receipt.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid Phase 1 receipt: {path}") from error


def _load_records(path: Path, model: type[RecordT]) -> list[RecordT]:
    return [model.model_validate(value) for value in iter_jsonl(path)]


def _check_receipt_identity(
    receipt: OcrPhase1Receipt,
    *,
    stage: str,
    run_id: str,
    input_hash: str,
    config_hash: str,
    identity: Phase1Identity,
    shard_id: str,
    shard_manifest_sha256: str,
    resource_limits_sha256: str,
) -> None:
    expected = (
        stage,
        run_id,
        input_hash,
        config_hash,
        identity.detector_id,
        identity.detector_revision,
        identity.detector_tree_sha256,
        identity.runtime_identity_sha256,
        shard_id,
        shard_manifest_sha256,
        resource_limits_sha256,
    )
    actual = (
        receipt.stage,
        receipt.run_id,
        receipt.input_artifact_sha256,
        receipt.config_sha256,
        receipt.detector_id,
        receipt.detector_revision,
        receipt.detector_tree_sha256,
        receipt.runtime_identity_sha256,
        receipt.shard_id,
        receipt.shard_manifest_sha256,
        receipt.resource_limits_sha256,
    )
    if actual != expected:
        raise ValueError("receipt run/stage/input/config/model identity drift")


def _verify_completed_receipt(
    output: Path,
    receipt_path: Path,
    *,
    stage: str | None = None,
) -> OcrPhase1Receipt:
    if not output.is_file() or not receipt_path.is_file():
        raise ValueError("completed artifact requires both output and receipt")
    receipt = _read_receipt(receipt_path)
    if receipt.status != "completed" or (stage is not None and receipt.stage != stage):
        raise ValueError("artifact receipt is not the expected completed stage")
    if output.stat().st_size != receipt.committed_bytes:
        raise ValueError("artifact byte size differs from completed receipt")
    if sha256_file(output) != receipt.output_sha256:
        raise ValueError("artifact output checksum differs from receipt")
    return receipt


def _load_frame_manifest_with_hash(
    frame_manifest: Path, data_root: Path
) -> tuple[str, list[tuple[FrameRef, Path]]]:
    _, manifest_hash, refs = _frame_manifest_snapshot(frame_manifest)
    root = data_root.resolve()
    validated: list[tuple[FrameRef, Path]] = []
    for ref in refs:
        if ref.source_image_sha256 is None:
            raise ValueError(f"frame lacks source image checksum: {ref.frame_uid}")
        path = (root / ref.frame_relpath).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"frame path escapes data root: {ref.frame_uid}") from error
        if not path.is_file():
            raise ValueError(f"source image is missing: {ref.frame_uid}")
        validated.append((ref, path))
    return manifest_hash, validated


def load_frame_manifest(frame_manifest: Path, data_root: Path) -> list[tuple[FrameRef, Path]]:
    return _load_frame_manifest_with_hash(frame_manifest, data_root)[1]


def validate_frame_sources(frame_manifest: Path, data_root: Path) -> dict[str, int]:
    """Full canonical source validation gate used before detector construction."""

    manifest_hash, frames = _load_frame_manifest_with_hash(frame_manifest, data_root)
    for ref, path in frames:
        decode_canonical_frame(ref, path)
    if sha256_file(frame_manifest) != manifest_hash:
        raise ValueError("frame manifest changed during full source validation")
    return {"frames": len(frames)}


def _verify_all_frame_source_hashes(frames: list[tuple[FrameRef, Path]]) -> None:
    for ref, path in frames:
        if sha256_file(path) != ref.source_image_sha256:
            raise ValueError(f"source image checksum drift: {ref.frame_uid}")


def _record_for_frame(
    ref: FrameRef,
    path: Path,
    *,
    run_id: str,
    config_sha256: str,
    identity: Phase1Identity,
    detector: Detector | None,
    crop_config: CropConfig,
    maximum_detections_per_frame: int,
    remaining_detections_in_shard: int,
) -> OcrDetectionFrameRecord:
    snapshot = decode_canonical_frame(ref, path)
    if detector is None:
        raise RuntimeError("detector construction is required for unfinished frames")
    raw_detections = detector.detect(snapshot.bgr, width=ref.width, height=ref.height)
    if len(raw_detections) > maximum_detections_per_frame:
        raise ValueError("detector output exceeds maximum_detections_per_frame before crop")
    if len(raw_detections) > remaining_detections_in_shard:
        raise ValueError("detector output exceeds remaining shard detection capacity before crop")
    detections: list[OcrDetection] = []
    for item in raw_detections:
        raw_quad = canonical_quad(item.raw_points)
        native_quad = validate_canonical_quad(item.points)
        expected_native = clamp_quad(raw_quad, width=ref.width, height=ref.height)
        if native_quad != expected_native or item.clamped is not (raw_quad != native_quad):
            raise ValueError("detector clamp provenance is inconsistent")
        crop = encode_crop(snapshot.image, native_quad, config=crop_config)
        detections.append(
            OcrDetection(
                detection_id=f"{ref.frame_uid}:det-{item.source_order:04d}",
                source_order=item.source_order,
                polygon_raw_xy=RawQuadGeometry(points=item.raw_points),
                polygon_xy=QuadGeometry(points=native_quad),
                polygon_clamped=item.clamped,
                detector_score=item.score,
                crop=crop.provenance,
            )
        )
    if sha256_file(path) != ref.source_image_sha256:
        raise ValueError(f"source image changed during detector/crop use: {ref.frame_uid}")
    return OcrDetectionFrameRecord(
        **ref.model_dump(),
        run_id=run_id,
        detector_revision=identity.detector_revision,
        detector_tree_sha256=identity.detector_tree_sha256,
        runtime_identity_sha256=identity.runtime_identity_sha256,
        config_sha256=config_sha256,
        canonical_image_sha256=snapshot.canonical_image_sha256,
        detections=detections,
    )


def _write_jsonl_line(stream: Any, record: BaseModel) -> bytes:
    serialized = (
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    stream.write(serialized)
    stream.flush()
    os.fsync(stream.fileno())
    return serialized.encode("utf-8")


def _canonical_jsonl_bytes(records: list[BaseModel]) -> bytes:
    return "".join(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def _quarantine(path: Path) -> Path:
    sequence = 1
    while True:
        quarantine = path.with_suffix(path.suffix + f".uncommitted.{sequence:06d}")
        if not quarantine.exists():
            break
        sequence += 1
    os.replace(path, quarantine)
    _fsync_directory(path.parent)
    return quarantine


def _sha256_state(path: Path) -> Any:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest


def _validate_detection_prefix(
    records: list[OcrDetectionFrameRecord],
    frames: list[tuple[FrameRef, Path]],
    *,
    run_id: str,
    config_sha256: str,
    identity: Phase1Identity,
    crop_config: CropConfig,
) -> None:
    if len(records) > len(frames):
        raise ValueError("partial detection artifact contains extra frame records")
    for index, (record, (ref, path)) in enumerate(zip(records, frames, strict=False)):
        expected_frame = ref.model_dump()
        actual_frame = {key: getattr(record, key) for key in expected_frame}
        if actual_frame != expected_frame:
            raise ValueError(f"partial artifact is not a valid frame prefix at record {index}")
        if (
            record.run_id != run_id
            or record.config_sha256 != config_sha256
            or record.detector_revision != identity.detector_revision
            or record.detector_tree_sha256 != identity.detector_tree_sha256
            or record.runtime_identity_sha256 != identity.runtime_identity_sha256
        ):
            raise ValueError(f"partial artifact identity drift at record {index}")
        snapshot = decode_canonical_frame(ref, path)
        if record.canonical_image_sha256 != snapshot.canonical_image_sha256:
            raise ValueError("canonical source pixel hash mismatch")
        for detection in record.detections:
            if detection.crop.crop_config_sha256 != crop_config.sha256:
                raise ValueError("partial artifact crop config drift")
            raw_quad = canonical_quad(detection.polygon_raw_xy.points)
            expected_native = clamp_quad(raw_quad, width=ref.width, height=ref.height)
            if detection.polygon_xy.points != expected_native:
                raise ValueError("detection canonical/clamped polygon provenance mismatch")
            if detection.polygon_clamped is not (raw_quad != expected_native):
                raise ValueError("detection polygon_clamped flag mismatch")
            expected_crop = encode_crop(snapshot.image, expected_native, config=crop_config)
            if detection.crop != expected_crop.provenance:
                raise ValueError("detection crop provenance or metric mismatch")


def _detection_counts(records: list[OcrDetectionFrameRecord]) -> dict[str, int]:
    return {
        "frames": len(records),
        "detections": sum(len(record.detections) for record in records),
    }


def _validate_detection_stage_input(
    receipt: OcrPhase1Receipt,
    records: list[OcrDetectionFrameRecord],
    *,
    run_id: str,
    config_sha256: str,
    identity: Phase1Identity,
    resource_limits_sha256: str,
) -> None:
    if (
        receipt.run_id != run_id
        or receipt.config_sha256 != config_sha256
        or receipt.detector_revision != identity.detector_revision
        or receipt.detector_tree_sha256 != identity.detector_tree_sha256
        or receipt.runtime_identity_sha256 != identity.runtime_identity_sha256
        or receipt.resource_limits_sha256 != resource_limits_sha256
    ):
        raise ValueError("detection receipt run/config/model identity drift")
    validate_and_sort_detection_frames(records)
    if _detection_counts(records) != receipt.record_counts:
        raise ValueError("detection records differ from receipt counts")
    for record in records:
        if (
            record.run_id != run_id
            or record.config_sha256 != config_sha256
            or record.detector_revision != identity.detector_revision
            or record.detector_tree_sha256 != identity.detector_tree_sha256
            or record.runtime_identity_sha256 != identity.runtime_identity_sha256
        ):
            raise ValueError("detection record run/config/model identity drift")


def _validate_trajectory_stage_input(
    receipt: OcrPhase1Receipt,
    records: list[OcrTrajectoryRecord],
    *,
    run_id: str,
    config_sha256: str,
    identity: Phase1Identity,
    tracking_config_sha256: str,
    resource_limits_sha256: str,
) -> None:
    if (
        receipt.run_id != run_id
        or receipt.config_sha256 != config_sha256
        or receipt.detector_revision != identity.detector_revision
        or receipt.detector_tree_sha256 != identity.detector_tree_sha256
        or receipt.runtime_identity_sha256 != identity.runtime_identity_sha256
        or receipt.resource_limits_sha256 != resource_limits_sha256
    ):
        raise ValueError("trajectory receipt run/config/model identity drift")
    counts = {
        "trajectories": len(records),
        "members": sum(len(record.members) for record in records),
    }
    if counts != receipt.record_counts:
        raise ValueError("trajectory records differ from receipt counts")
    trajectory_ids = [record.trajectory_id for record in records]
    member_ids = [member.detection_id for record in records for member in record.members]
    if len(trajectory_ids) != len(set(trajectory_ids)) or len(member_ids) != len(set(member_ids)):
        raise ValueError("trajectory or member identity is duplicated")
    for record in records:
        if (
            record.run_id != run_id
            or record.config_sha256 != config_sha256
            or record.detector_revision != identity.detector_revision
            or record.detector_tree_sha256 != identity.detector_tree_sha256
            or record.runtime_identity_sha256 != identity.runtime_identity_sha256
            or record.tracking_config_sha256 != tracking_config_sha256
        ):
            raise ValueError("trajectory record run/config/model identity drift")


def _validate_representative_stage_output(
    receipt: OcrPhase1Receipt,
    records: list[RepresentativeCropBinding],
    trajectories: list[OcrTrajectoryRecord],
    *,
    run_id: str,
    config_sha256: str,
    identity: Phase1Identity,
    tracking_config_sha256: str,
) -> None:
    counts = {"representatives": len(records), "trajectories": len(trajectories)}
    if counts != receipt.record_counts:
        raise ValueError("representative records differ from receipt counts")
    keys = [(record.trajectory_id, record.representative_rank) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("representative identity is duplicated")
    member_ids = {
        (trajectory.trajectory_id, member.detection_id)
        for trajectory in trajectories
        for member in trajectory.members
    }
    for record in records:
        if (
            record.run_id != run_id
            or record.config_sha256 != config_sha256
            or record.detector_revision != identity.detector_revision
            or record.detector_tree_sha256 != identity.detector_tree_sha256
            or record.runtime_identity_sha256 != identity.runtime_identity_sha256
            or record.tracking_config_sha256 != tracking_config_sha256
        ):
            raise ValueError("representative record run/config/model identity drift")
        if (record.trajectory_id, record.detection_id) not in member_ids:
            raise ValueError("representative does not bind an input trajectory member")


def _run_detect_crop(
    *,
    frame_manifest: Path,
    data_root: Path,
    output: Path,
    run_id: str,
    config_sha256: str,
    detector: Detector | None,
    crop_config: CropConfig,
    identity: Phase1Identity = DEFAULT_PHASE1_IDENTITY,
    resume: bool = False,
    tracking_config: TrackingConfig = DEFAULT_TRACKING_CONFIG,
    shard_id: str = "shard-standalone",
    shard_manifest_sha256: str | None = None,
    fault_injector: FaultInjector | None = None,
) -> dict[str, int]:
    """Detect/crop frames with a receipt-authenticated durable prefix."""

    input_hash, frames = _load_frame_manifest_with_hash(frame_manifest, data_root)
    if len(frames) > tracking_config.maximum_frames_per_shard:
        raise ValueError("frame manifest exceeds maximum_frames_per_shard")
    if shard_manifest_sha256 is None:
        shard_manifest_sha256 = input_hash
    if shard_manifest_sha256 != input_hash:
        raise ValueError("shard manifest hash differs from detect input manifest")
    receipt_path = receipt_path_for(output)
    receipt_temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    partial = output.with_suffix(output.suffix + ".partial")
    temporary = output.with_suffix(output.suffix + ".tmp")
    completed: list[OcrDetectionFrameRecord]
    malformed_receipt_temporary = False
    if not receipt_path.exists() and receipt_temporary.exists():
        if not resume:
            raise FileExistsError("receipt temporary exists; use --resume")
        try:
            candidate_receipt = _read_receipt(receipt_temporary)
        except ValueError:
            _quarantine(receipt_temporary)
            malformed_receipt_temporary = True
        else:
            _check_receipt_identity(
                candidate_receipt,
                stage="detect_crop",
                run_id=run_id,
                input_hash=input_hash,
                config_hash=config_sha256,
                identity=identity,
                shard_id=shard_id,
                shard_manifest_sha256=shard_manifest_sha256,
                resource_limits_sha256=tracking_config.resource_limits_sha256,
            )
            os.replace(receipt_temporary, receipt_path)
            _fsync_directory(receipt_path.parent)
    elif receipt_path.exists() and receipt_temporary.exists() and resume:
        _quarantine(receipt_temporary)
    if receipt_path.exists():
        receipt = _read_receipt(receipt_path)
        _check_receipt_identity(
            receipt,
            stage="detect_crop",
            run_id=run_id,
            input_hash=input_hash,
            config_hash=config_sha256,
            identity=identity,
            shard_id=shard_id,
            shard_manifest_sha256=shard_manifest_sha256,
            resource_limits_sha256=tracking_config.resource_limits_sha256,
        )
        if receipt.status == "completed":
            if not resume:
                raise FileExistsError("completed detect/crop stage exists; use --resume to verify")
            _verify_completed_receipt(output, receipt_path, stage="detect_crop")
            records = _load_records(output, OcrDetectionFrameRecord)
            _validate_detection_prefix(
                records,
                frames,
                run_id=run_id,
                config_sha256=config_sha256,
                identity=identity,
                crop_config=crop_config,
            )
            if len(records) != len(frames) or _detection_counts(records) != receipt.record_counts:
                raise ValueError("completed detection record count is incomplete or inconsistent")
            _verify_all_frame_source_hashes(frames)
            if sha256_file(frame_manifest) != input_hash:
                raise ValueError("frame manifest changed during completed detection verification")
            return receipt.record_counts
        if not resume:
            raise FileExistsError("partial detect/crop stage exists; use --resume")
        if temporary.exists() or (output.exists() and partial.exists()):
            raise ValueError("running receipt has ambiguous detect/crop outputs")
        if output.is_file():
            _validate_commit_marker(output, receipt, truncate_tail=False)
            completed = _load_records(output, OcrDetectionFrameRecord)
            _validate_detection_prefix(
                completed,
                frames,
                run_id=run_id,
                config_sha256=config_sha256,
                identity=identity,
                crop_config=crop_config,
            )
            if (
                len(completed) != len(frames)
                or _detection_counts(completed) != receipt.record_counts
            ):
                raise ValueError("renamed detection output is not a complete committed artifact")
            completed_receipt = receipt.model_copy(update={"status": "completed"})
            _verify_all_frame_source_hashes(frames)
            _write_receipt(
                receipt_path,
                OcrPhase1Receipt.model_validate(completed_receipt.model_dump()),
            )
            recovered_receipt = _verify_completed_receipt(output, receipt_path, stage="detect_crop")
            _check_receipt_identity(
                recovered_receipt,
                stage="detect_crop",
                run_id=run_id,
                input_hash=input_hash,
                config_hash=config_sha256,
                identity=identity,
                shard_id=shard_id,
                shard_manifest_sha256=shard_manifest_sha256,
                resource_limits_sha256=tracking_config.resource_limits_sha256,
            )
            _verify_all_frame_source_hashes(frames)
            if sha256_file(frame_manifest) != input_hash:
                raise ValueError("frame manifest changed during recovered publication")
            return receipt.record_counts
        if not partial.is_file():
            raise ValueError("running receipt is missing its partial detect/crop output")
        _validate_commit_marker(partial, receipt, truncate_tail=True)
        completed = _load_records(partial, OcrDetectionFrameRecord)
        _validate_detection_prefix(
            completed,
            frames,
            run_id=run_id,
            config_sha256=config_sha256,
            identity=identity,
            crop_config=crop_config,
        )
        if (
            len(completed) != receipt.committed_records
            or _detection_counts(completed) != receipt.record_counts
        ):
            raise ValueError("partial record count differs from running receipt")
    else:
        orphans = [path for path in (output, partial, temporary) if path.exists()]
        recovered_unstarted = False
        if resume and orphans and all(path in {output, partial, temporary} for path in orphans):
            restartable = malformed_receipt_temporary or (
                orphans == [partial] and partial.stat().st_size == 0
            )
            if restartable:
                if detector is None:
                    raise RuntimeError(
                        "non-authenticated detect state requires detector to restart from zero"
                    )
                for orphan in orphans:
                    _quarantine(orphan)
                orphans = []
                recovered_unstarted = True
        if orphans:
            raise FileExistsError(f"detect/crop artifact exists without receipt: {orphans[0]}")
        if resume and not recovered_unstarted:
            raise FileNotFoundError("cannot resume detect/crop without a receipt")
        output.parent.mkdir(parents=True, exist_ok=True)
        partial.touch(exist_ok=False)
        if fault_injector is not None:
            fault_injector("after_partial_create_before_receipt")
        completed = []
        _write_receipt(
            receipt_path,
            _make_receipt(
                run_id=run_id,
                stage="detect_crop",
                status="running",
                input_hash=input_hash,
                config_sha256=config_sha256,
                identity=identity,
                record_counts=_detection_counts(completed),
                output_hash=hashlib.sha256().hexdigest(),
                committed_bytes=0,
                shard_id=shard_id,
                shard_manifest_sha256=shard_manifest_sha256,
                resource_limits_sha256=tracking_config.resource_limits_sha256,
            ),
            fault_injector,
        )

    committed_detection_count = _detection_counts(completed)["detections"]
    if committed_detection_count > tracking_config.maximum_detections_per_shard:
        raise ValueError(
            f"shard {shard_id} committed detection count {committed_detection_count} "
            f"exceeds limit {tracking_config.maximum_detections_per_shard}; "
            "stateful video subsharding "
            "is required"
        )
    partial_digest = _sha256_state(partial)
    with partial.open("a", encoding="utf-8", newline="\n") as stream:
        for ref, path in frames[len(completed) :]:
            record = _record_for_frame(
                ref,
                path,
                run_id=run_id,
                config_sha256=config_sha256,
                identity=identity,
                detector=detector,
                crop_config=crop_config,
                maximum_detections_per_frame=tracking_config.maximum_detections_per_frame,
                remaining_detections_in_shard=(
                    tracking_config.maximum_detections_per_shard - committed_detection_count
                ),
            )
            next_detection_count = committed_detection_count + len(record.detections)
            if next_detection_count > tracking_config.maximum_detections_per_shard:
                raise ValueError(
                    f"shard {shard_id} detection count {next_detection_count} exceeds limit "
                    f"{tracking_config.maximum_detections_per_shard}; reshard only at "
                    "whole-video boundaries. "
                    "If one video exceeds the limit, stateful video subsharding is required"
                )
            partial_digest.update(_write_jsonl_line(stream, record))
            if fault_injector is not None:
                fault_injector("after_record_fsync_before_receipt")
            completed.append(record)
            committed_detection_count = next_detection_count
            _write_receipt(
                receipt_path,
                _make_receipt(
                    run_id=run_id,
                    stage="detect_crop",
                    status="running",
                    input_hash=input_hash,
                    config_sha256=config_sha256,
                    identity=identity,
                    record_counts=_detection_counts(completed),
                    output_hash=partial_digest.hexdigest(),
                    committed_bytes=partial.stat().st_size,
                    shard_id=shard_id,
                    shard_manifest_sha256=shard_manifest_sha256,
                    resource_limits_sha256=tracking_config.resource_limits_sha256,
                ),
                fault_injector,
            )
    os.replace(partial, output)
    _fsync_directory(output.parent)
    if fault_injector is not None:
        fault_injector("after_final_rename_before_receipt")
    counts = _detection_counts(completed)
    _verify_all_frame_source_hashes(frames)
    if sha256_file(frame_manifest) != input_hash:
        raise ValueError("frame manifest changed before detect/crop publication")
    _write_receipt(
        receipt_path,
        _make_receipt(
            run_id=run_id,
            stage="detect_crop",
            status="completed",
            input_hash=input_hash,
            config_sha256=config_sha256,
            identity=identity,
            record_counts=counts,
            output_hash=partial_digest.hexdigest(),
            committed_bytes=output.stat().st_size,
            shard_id=shard_id,
            shard_manifest_sha256=shard_manifest_sha256,
            resource_limits_sha256=tracking_config.resource_limits_sha256,
        ),
        fault_injector,
    )
    if fault_injector is not None:
        fault_injector("after_completed_receipt_before_final_verification")
    final_receipt = _verify_completed_receipt(output, receipt_path, stage="detect_crop")
    _check_receipt_identity(
        final_receipt,
        stage="detect_crop",
        run_id=run_id,
        input_hash=input_hash,
        config_hash=config_sha256,
        identity=identity,
        shard_id=shard_id,
        shard_manifest_sha256=shard_manifest_sha256,
        resource_limits_sha256=tracking_config.resource_limits_sha256,
    )
    _verify_all_frame_source_hashes(frames)
    if final_receipt.record_counts != counts or sha256_file(frame_manifest) != input_hash:
        raise ValueError("detect/crop changed after completed receipt publication")
    return counts


def run_detect_crop(**kwargs: Any) -> dict[str, int]:
    """Production publication route accepting only verified Paddle detector evidence."""

    detector = kwargs.get("detector")
    identity = kwargs.get("identity", DEFAULT_PHASE1_IDENTITY)
    if detector is not None:
        if not isinstance(detector, PaddleOcrV6Detector) or not is_production_detector_attested(
            detector
        ):
            raise TypeError("production detect/crop requires a verified PaddleOcrV6Detector")
        verification = detector.verification
        expected = {
            "detector_id": identity.detector_id,
            "detector_revision": identity.detector_revision,
            "detector_tree_sha256": identity.detector_tree_sha256,
            "runtime_identity_sha256": identity.runtime_identity_sha256,
        }
        if any(verification.get(name) != value for name, value in expected.items()):
            raise ValueError("detector verification evidence differs from Phase1Identity")
        if verification.get("model_snapshot_verified") is not True:
            raise ValueError("detector lacks verified private model snapshot evidence")
    return _run_detect_crop(**kwargs)


def _run_detect_crop_for_test(**kwargs: Any) -> dict[str, int]:
    """Internal fake-detector route; never imported by the production CLI."""

    return _run_detect_crop(**kwargs)


def _make_receipt(
    *,
    run_id: str,
    stage: str,
    status: str,
    input_hash: str,
    config_sha256: str,
    identity: Phase1Identity,
    record_counts: dict[str, int],
    output_hash: str,
    committed_bytes: int,
    shard_id: str,
    shard_manifest_sha256: str,
    resource_limits_sha256: str,
) -> OcrPhase1Receipt:
    primary_key = {
        "detect_crop": "frames",
        "track": "trajectories",
        "select_representatives": "representatives",
    }[stage]
    return OcrPhase1Receipt(
        run_id=run_id,
        stage=stage,
        status=status,
        input_artifact_sha256=input_hash,
        config_sha256=config_sha256,
        detector_revision=identity.detector_revision,
        detector_tree_sha256=identity.detector_tree_sha256,
        runtime_identity_sha256=identity.runtime_identity_sha256,
        resource_limits_sha256=resource_limits_sha256,
        shard_id=shard_id,
        shard_manifest_sha256=shard_manifest_sha256,
        record_counts=record_counts,
        output_sha256=output_hash,
        committed_bytes=committed_bytes,
        committed_records=record_counts[primary_key],
        committed_sha256=output_hash,
    )


def _validate_commit_marker(path: Path, receipt: OcrPhase1Receipt, *, truncate_tail: bool) -> None:
    size = path.stat().st_size
    if size < receipt.committed_bytes:
        raise ValueError("artifact is shorter than its committed byte offset")
    digest = hashlib.sha256()
    remaining = receipt.committed_bytes
    with path.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("artifact ended before committed byte offset")
            digest.update(chunk)
            remaining -= len(chunk)
    if digest.hexdigest() != receipt.committed_sha256:
        raise ValueError("committed artifact prefix hash differs from receipt")
    if size > receipt.committed_bytes:
        if not truncate_tail:
            raise ValueError("completed artifact contains bytes beyond commit marker")
        with path.open("r+b") as stream:
            stream.truncate(receipt.committed_bytes)
            stream.flush()
            os.fsync(stream.fileno())


def _recover_or_publish_derived(
    *,
    output: Path,
    records: list[BaseModel],
    stage: str,
    run_id: str,
    input_hash: str,
    config_sha256: str,
    identity: Phase1Identity,
    record_counts: dict[str, int],
    resume: bool,
    fault_injector: FaultInjector | None,
    shard_id: str,
    shard_manifest_sha256: str,
    input_path: Path,
    resource_limits_sha256: str,
) -> dict[str, int]:
    receipt_path = receipt_path_for(output)
    receipt_temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary = output.with_suffix(output.suffix + ".tmp")
    canonical_bytes = _canonical_jsonl_bytes(records)
    if not receipt_path.exists() and receipt_temporary.exists():
        if not resume:
            raise FileExistsError(f"{stage} receipt temporary exists; use --resume")
        try:
            candidate_receipt = _read_receipt(receipt_temporary)
        except ValueError:
            _quarantine(receipt_temporary)
        else:
            _check_receipt_identity(
                candidate_receipt,
                stage=stage,
                run_id=run_id,
                input_hash=input_hash,
                config_hash=config_sha256,
                identity=identity,
                shard_id=shard_id,
                shard_manifest_sha256=shard_manifest_sha256,
                resource_limits_sha256=resource_limits_sha256,
            )
            os.replace(receipt_temporary, receipt_path)
            _fsync_directory(receipt_path.parent)
    elif receipt_path.exists() and receipt_temporary.exists() and resume:
        _quarantine(receipt_temporary)
    if receipt_path.exists():
        if not resume:
            raise FileExistsError(f"{stage} stage receipt exists; use --resume to verify")
        receipt = _verify_completed_receipt(output, receipt_path, stage=stage)
        _check_receipt_identity(
            receipt,
            stage=stage,
            run_id=run_id,
            input_hash=input_hash,
            config_hash=config_sha256,
            identity=identity,
            shard_id=shard_id,
            shard_manifest_sha256=shard_manifest_sha256,
            resource_limits_sha256=resource_limits_sha256,
        )
        if temporary.exists():
            _quarantine(temporary)
        if output.read_bytes() != canonical_bytes:
            raise ValueError(f"completed {stage} output differs from deterministic replay")
        if receipt.record_counts != record_counts:
            raise ValueError(f"completed {stage} receipt counts differ from replay")
        if sha256_file(input_path) != input_hash:
            raise ValueError(f"{stage} input changed during completed verification")
        return record_counts

    existing = [path for path in (output, temporary) if path.exists()]
    if existing:
        if not resume:
            raise FileExistsError(f"{stage} artifact exists without receipt: {existing[0]}")
        if len(existing) != 1:
            raise ValueError(f"{stage} recovery has ambiguous final/temporary outputs")
        candidate = existing[0]
        if candidate == temporary:
            _quarantine(temporary)
            existing = []
        elif candidate.read_bytes() != canonical_bytes:
            raise ValueError(f"orphan {stage} final output differs from canonical replay bytes")
    elif resume:
        raise FileNotFoundError(f"cannot resume {stage} without an artifact or receipt")
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as stream:
            stream.write(canonical_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        if fault_injector is not None:
            fault_injector("after_derived_temp_fsync_before_publish")
        os.replace(temporary, output)
        _fsync_directory(output.parent)

    if sha256_file(input_path) != input_hash:
        raise ValueError(f"{stage} upstream artifact changed before publication")
    output_hash = sha256_file(output)
    if fault_injector is not None:
        fault_injector("after_output_publish_before_receipt")
    if sha256_file(input_path) != input_hash:
        raise ValueError(f"{stage} upstream artifact changed before receipt publication")
    _write_receipt(
        receipt_path,
        _make_receipt(
            run_id=run_id,
            stage=stage,
            status="completed",
            input_hash=input_hash,
            config_sha256=config_sha256,
            identity=identity,
            record_counts=record_counts,
            output_hash=output_hash,
            committed_bytes=output.stat().st_size,
            shard_id=shard_id,
            shard_manifest_sha256=shard_manifest_sha256,
            resource_limits_sha256=resource_limits_sha256,
        ),
        fault_injector,
    )
    if fault_injector is not None:
        fault_injector("after_completed_receipt_before_final_verification")
    final_receipt = _verify_completed_receipt(output, receipt_path, stage=stage)
    _check_receipt_identity(
        final_receipt,
        stage=stage,
        run_id=run_id,
        input_hash=input_hash,
        config_hash=config_sha256,
        identity=identity,
        shard_id=shard_id,
        shard_manifest_sha256=shard_manifest_sha256,
        resource_limits_sha256=resource_limits_sha256,
    )
    if (
        final_receipt.record_counts != record_counts
        or output.read_bytes() != canonical_bytes
        or sha256_file(input_path) != input_hash
    ):
        raise ValueError(f"{stage} changed after completed receipt publication")
    return record_counts


def run_tracking(
    *,
    detections: Path,
    output: Path,
    run_id: str,
    config_sha256: str,
    tracking_config: TrackingConfig,
    identity: Phase1Identity = DEFAULT_PHASE1_IDENTITY,
    resume: bool = False,
    fault_injector: FaultInjector | None = None,
) -> dict[str, int]:
    detection_receipt = _verify_completed_receipt(
        detections, receipt_path_for(detections), stage="detect_crop"
    )
    detection_hash = detection_receipt.output_sha256
    records = _load_records(detections, OcrDetectionFrameRecord)
    _validate_detection_stage_input(
        detection_receipt,
        records,
        run_id=run_id,
        config_sha256=config_sha256,
        identity=identity,
        resource_limits_sha256=tracking_config.resource_limits_sha256,
    )
    trajectories = build_trajectories(records, config=tracking_config)
    counts = {
        "trajectories": len(trajectories),
        "members": sum(len(item.members) for item in trajectories),
    }
    return _recover_or_publish_derived(
        output=output,
        records=trajectories,
        stage="track",
        run_id=run_id,
        input_hash=detection_hash,
        config_sha256=config_sha256,
        identity=identity,
        record_counts=counts,
        resume=resume,
        fault_injector=fault_injector,
        shard_id=detection_receipt.shard_id,
        shard_manifest_sha256=detection_receipt.shard_manifest_sha256,
        input_path=detections,
        resource_limits_sha256=tracking_config.resource_limits_sha256,
    )


def run_representative_selection(
    *,
    trajectories: Path,
    output: Path,
    run_id: str,
    config_sha256: str,
    tracking_config: TrackingConfig,
    identity: Phase1Identity = DEFAULT_PHASE1_IDENTITY,
    resume: bool = False,
    fault_injector: FaultInjector | None = None,
) -> dict[str, int]:
    trajectory_receipt = _verify_completed_receipt(
        trajectories, receipt_path_for(trajectories), stage="track"
    )
    trajectory_hash = trajectory_receipt.output_sha256
    records = _load_records(trajectories, OcrTrajectoryRecord)
    _validate_trajectory_stage_input(
        trajectory_receipt,
        records,
        run_id=run_id,
        config_sha256=config_sha256,
        identity=identity,
        tracking_config_sha256=tracking_config.sha256,
        resource_limits_sha256=tracking_config.resource_limits_sha256,
    )
    representatives = select_representatives(records, config=tracking_config)
    counts = {
        "representatives": len(representatives),
        "trajectories": len(records),
    }
    return _recover_or_publish_derived(
        output=output,
        records=representatives,
        stage="select_representatives",
        run_id=run_id,
        input_hash=trajectory_hash,
        config_sha256=config_sha256,
        identity=identity,
        record_counts=counts,
        resume=resume,
        fault_injector=fault_injector,
        shard_id=trajectory_receipt.shard_id,
        shard_manifest_sha256=trajectory_receipt.shard_manifest_sha256,
        input_path=trajectories,
        resource_limits_sha256=tracking_config.resource_limits_sha256,
    )


def verify_detection_artifact(
    *,
    output: Path,
    frame_manifest: Path,
    data_root: Path,
    crop_config: CropConfig,
    expected_run_id: str,
    expected_config_sha256: str,
    expected_identity: Phase1Identity,
    expected_shard_id: str = "shard-standalone",
    tracking_config: TrackingConfig = DEFAULT_TRACKING_CONFIG,
) -> dict[str, int]:
    receipt = _verify_completed_receipt(output, receipt_path_for(output), stage="detect_crop")
    input_hash, frames = _load_frame_manifest_with_hash(frame_manifest, data_root)
    _check_receipt_identity(
        receipt,
        stage="detect_crop",
        run_id=expected_run_id,
        input_hash=input_hash,
        config_hash=expected_config_sha256,
        identity=expected_identity,
        shard_id=expected_shard_id,
        shard_manifest_sha256=input_hash,
        resource_limits_sha256=tracking_config.resource_limits_sha256,
    )
    records = _load_records(output, OcrDetectionFrameRecord)
    _validate_detection_prefix(
        records,
        frames,
        run_id=expected_run_id,
        config_sha256=expected_config_sha256,
        identity=expected_identity,
        crop_config=crop_config,
    )
    counts = _detection_counts(records)
    if len(records) != len(frames) or counts != receipt.record_counts:
        raise ValueError("detection artifact has missing/duplicate records or count drift")
    if sha256_file(frame_manifest) != input_hash or sha256_file(output) != receipt.output_sha256:
        raise ValueError("detection verification input changed during semantic replay")
    return counts


def verify_tracking_artifact(
    *,
    detections: Path,
    trajectories: Path,
    expected_run_id: str,
    expected_config_sha256: str,
    expected_identity: Phase1Identity,
    tracking_config: TrackingConfig,
) -> dict[str, int]:
    """Replay and verify a completed detect -> track boundary without selection."""

    detection_receipt = _verify_completed_receipt(
        detections, receipt_path_for(detections), stage="detect_crop"
    )
    trajectory_receipt = _verify_completed_receipt(
        trajectories, receipt_path_for(trajectories), stage="track"
    )
    _check_receipt_identity(
        trajectory_receipt,
        stage="track",
        run_id=expected_run_id,
        input_hash=sha256_file(detections),
        config_hash=expected_config_sha256,
        identity=expected_identity,
        shard_id=detection_receipt.shard_id,
        shard_manifest_sha256=detection_receipt.shard_manifest_sha256,
        resource_limits_sha256=tracking_config.resource_limits_sha256,
    )
    detection_records = _load_records(detections, OcrDetectionFrameRecord)
    trajectory_records = _load_records(trajectories, OcrTrajectoryRecord)
    _validate_detection_stage_input(
        detection_receipt,
        detection_records,
        run_id=expected_run_id,
        config_sha256=expected_config_sha256,
        identity=expected_identity,
        resource_limits_sha256=tracking_config.resource_limits_sha256,
    )
    _validate_trajectory_stage_input(
        trajectory_receipt,
        trajectory_records,
        run_id=expected_run_id,
        config_sha256=expected_config_sha256,
        identity=expected_identity,
        tracking_config_sha256=tracking_config.sha256,
        resource_limits_sha256=tracking_config.resource_limits_sha256,
    )
    expected = build_trajectories(detection_records, config=tracking_config)
    expected_bytes = _canonical_jsonl_bytes(expected)
    counts = {
        "trajectories": len(expected),
        "members": sum(len(item.members) for item in expected),
    }
    if trajectory_receipt.record_counts != counts or trajectories.read_bytes() != expected_bytes:
        raise ValueError("trajectory artifact differs from deterministic replay")
    if (
        sha256_file(detections) != detection_receipt.output_sha256
        or sha256_file(trajectories) != trajectory_receipt.output_sha256
    ):
        raise ValueError("tracking verification input changed during replay")
    return counts


def verify_linked_artifacts(
    *,
    detections: Path,
    trajectories: Path,
    representatives: Path,
    expected_run_id: str,
    expected_config_sha256: str,
    expected_identity: Phase1Identity,
    tracking_config: TrackingConfig,
) -> dict[str, int]:
    detection_receipt = _verify_completed_receipt(
        detections, receipt_path_for(detections), stage="detect_crop"
    )
    trajectory_receipt = _verify_completed_receipt(
        trajectories, receipt_path_for(trajectories), stage="track"
    )
    representative_receipt = _verify_completed_receipt(
        representatives, receipt_path_for(representatives), stage="select_representatives"
    )
    _check_receipt_identity(
        trajectory_receipt,
        stage="track",
        run_id=expected_run_id,
        input_hash=sha256_file(detections),
        config_hash=expected_config_sha256,
        identity=expected_identity,
        shard_id=detection_receipt.shard_id,
        shard_manifest_sha256=detection_receipt.shard_manifest_sha256,
        resource_limits_sha256=tracking_config.resource_limits_sha256,
    )
    _check_receipt_identity(
        representative_receipt,
        stage="select_representatives",
        run_id=expected_run_id,
        input_hash=sha256_file(trajectories),
        config_hash=expected_config_sha256,
        identity=expected_identity,
        shard_id=trajectory_receipt.shard_id,
        shard_manifest_sha256=trajectory_receipt.shard_manifest_sha256,
        resource_limits_sha256=tracking_config.resource_limits_sha256,
    )
    if (
        detection_receipt.run_id != expected_run_id
        or detection_receipt.config_sha256 != expected_config_sha256
        or detection_receipt.detector_revision != expected_identity.detector_revision
        or detection_receipt.detector_tree_sha256 != expected_identity.detector_tree_sha256
        or detection_receipt.runtime_identity_sha256 != expected_identity.runtime_identity_sha256
    ):
        raise ValueError("detection receipt differs from expected config identity")
    detection_records = _load_records(detections, OcrDetectionFrameRecord)
    trajectory_records = _load_records(trajectories, OcrTrajectoryRecord)
    representative_records = _load_records(representatives, RepresentativeCropBinding)
    detection_ids = {
        detection.detection_id for frame in detection_records for detection in frame.detections
    }
    trajectory_ids = [item.trajectory_id for item in trajectory_records]
    member_ids = [member.detection_id for item in trajectory_records for member in item.members]
    representative_keys = [
        (item.trajectory_id, item.representative_rank) for item in representative_records
    ]
    _validate_detection_stage_input(
        detection_receipt,
        detection_records,
        run_id=expected_run_id,
        config_sha256=expected_config_sha256,
        identity=expected_identity,
        resource_limits_sha256=tracking_config.resource_limits_sha256,
    )
    _validate_trajectory_stage_input(
        trajectory_receipt,
        trajectory_records,
        run_id=expected_run_id,
        config_sha256=expected_config_sha256,
        identity=expected_identity,
        tracking_config_sha256=tracking_config.sha256,
        resource_limits_sha256=tracking_config.resource_limits_sha256,
    )
    _validate_representative_stage_output(
        representative_receipt,
        representative_records,
        trajectory_records,
        run_id=expected_run_id,
        config_sha256=expected_config_sha256,
        identity=expected_identity,
        tracking_config_sha256=tracking_config.sha256,
    )
    actual_detection_counts = _detection_counts(detection_records)
    actual_trajectory_counts = {
        "trajectories": len(trajectory_records),
        "members": len(member_ids),
    }
    actual_representative_counts = {
        "representatives": len(representative_records),
        "trajectories": len(trajectory_records),
    }
    if actual_detection_counts != detection_receipt.record_counts:
        raise ValueError("detection receipt record count drift")
    if actual_trajectory_counts != trajectory_receipt.record_counts:
        raise ValueError("trajectory receipt record count drift")
    if actual_representative_counts != representative_receipt.record_counts:
        raise ValueError("representative receipt record count drift")
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("duplicate trajectory record")
    if len(member_ids) != len(set(member_ids)) or set(member_ids) != detection_ids:
        raise ValueError("trajectory members are missing or duplicate detection records")
    if len(representative_keys) != len(set(representative_keys)):
        raise ValueError("duplicate representative binding")
    known_trajectories = set(trajectory_ids)
    if any(item.trajectory_id not in known_trajectories for item in representative_records):
        raise ValueError("representative references a missing trajectory")
    members_by_id = {
        member.detection_id: (trajectory.trajectory_id, member)
        for trajectory in trajectory_records
        for member in trajectory.members
    }
    representatives_by_trajectory: dict[str, list[RepresentativeCropBinding]] = {}
    for representative in representative_records:
        linked = members_by_id.get(representative.detection_id)
        if linked is None or linked[0] != representative.trajectory_id:
            raise ValueError("representative does not bind a member of its trajectory")
        member = linked[1]
        shared_fields = (
            "video_id",
            "frame_uid",
            "frame_idx",
            "frame_relpath",
            "source_image_sha256",
            "canonical_image_sha256",
            "source_width",
            "source_height",
            "detection_id",
            "detector_score",
            "polygon_xy",
            "crop",
        )
        if any(getattr(representative, name) != getattr(member, name) for name in shared_fields):
            raise ValueError("representative member provenance drift")
        representatives_by_trajectory.setdefault(representative.trajectory_id, []).append(
            representative
        )
    for trajectory in trajectory_records:
        selected = representatives_by_trajectory.get(trajectory.trajectory_id, [])
        expected_count = min(len(trajectory.members), 3)
        actual_ranks = [item.representative_rank for item in selected]
        expected_ranks = list(range(1, expected_count + 1))
        if len(selected) != expected_count or actual_ranks != expected_ranks:
            raise ValueError("trajectory has missing or non-contiguous representative ranks")
    replayed_trajectories = build_trajectories(detection_records, config=tracking_config)
    if trajectory_records != replayed_trajectories:
        raise ValueError("trajectory artifact differs from deterministic replay")
    replayed_representatives = select_representatives(replayed_trajectories, config=tracking_config)
    if representative_records != replayed_representatives:
        raise ValueError("representative artifact differs from deterministic replay")
    if (
        sha256_file(detections) != detection_receipt.output_sha256
        or sha256_file(trajectories) != trajectory_receipt.output_sha256
        or sha256_file(representatives) != representative_receipt.output_sha256
    ):
        raise ValueError("linked artifact changed during semantic verification")
    return {
        "frames": len(detection_records),
        "detections": len(detection_ids),
        "trajectories": len(trajectory_records),
        "representatives": len(representative_records),
    }
