"""Portable, crash-safe checkpoint bundles for OCR Phase 1 Kaggle runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from aic2026.common import sha256_file
from aic2026.contracts import (
    OcrCheckpointArtifact,
    OcrCheckpointCounts,
    OcrCheckpointFile,
    OcrDetectionFrameRecord,
    OcrGlobalShardManifest,
    OcrPhase1Checkpoint,
    OcrPhase1Receipt,
)

from .geometry import CropConfig
from .phase1 import (
    Phase1Identity,
    _check_receipt_identity,
    _detection_counts,
    _load_frame_manifest_with_hash,
    _load_records,
    _read_receipt,
    _validate_detection_prefix,
    _verify_all_frame_source_hashes,
    receipt_path_for,
    verify_detection_artifact,
    verify_linked_artifacts,
    verify_tracking_artifact,
)
from .sharding import global_shard_receipt_path, verify_global_shard_structure
from .tracking import TrackingConfig

FaultInjector = Callable[[str], None]
CHECKPOINT_MARKER = "checkpoint.json"

_SECRET_MARKERS = (
    b"openai_api_key",
    b"sk-",
    b"google_oauth",
    b"refresh_token",
    b"access_token",
    b"client_secret",
    b"rclone.conf",
    b"rclone_config",
    b"password_command",
    b"ya29.",
)


@dataclass(frozen=True, slots=True)
class CheckpointArtifactPaths:
    detections: Path
    trajectories: Path
    representatives: Path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _marker_bytes(checkpoint: OcrPhase1Checkpoint) -> bytes:
    return (
        json.dumps(
            checkpoint.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _replay_identity(checkpoint: OcrPhase1Checkpoint) -> dict:
    payload = checkpoint.model_dump(mode="json")
    for field in ("created_at", "checkpoint_sequence", "previous_checkpoint_sha256"):
        payload.pop(field)
    return payload


def _assert_secret_free(payload: bytes, *, label: str) -> None:
    lowered = payload.lower()
    if any(marker in lowered for marker in _SECRET_MARKERS):
        raise ValueError(f"checkpoint input contains forbidden credential material: {label}")
    # Phase 1 artifacts are JSON and never need bearer/API tokens. Keep this
    # deliberately narrow to avoid treating ordinary OCR data as a credential.
    if b'"authorization"' in lowered and b"bearer " in lowered:
        raise ValueError(f"checkpoint input contains forbidden bearer credential: {label}")


def _safe_child(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("checkpoint path escapes its root") from error
    return target


def _artifact_relative(path: Path, artifact_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(artifact_root.resolve())
    except ValueError as error:
        raise ValueError("checkpoint artifact path escapes artifact root") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("checkpoint artifact path is not a safe relative path")
    return relative.as_posix()


def _write_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _read_exact_prefix(path: Path, byte_count: int) -> bytes:
    with path.open("rb") as stream:
        payload = stream.read(byte_count)
    if len(payload) != byte_count:
        raise ValueError("artifact is shorter than its committed receipt offset")
    return payload


def _checkpoint_chain(root: Path) -> list[tuple[Path, OcrPhase1Checkpoint, str]]:
    if (root / CHECKPOINT_MARKER).is_file():
        candidates = [root]
    else:
        candidates = sorted(
            child
            for child in root.iterdir()
            if child.is_dir() and child.name.startswith("checkpoint-")
        )
    parsed: list[tuple[Path, OcrPhase1Checkpoint, str]] = []
    for candidate in candidates:
        marker = candidate / CHECKPOINT_MARKER
        if not marker.is_file():
            raise ValueError(f"checkpoint bundle is missing its commit marker: {candidate}")
        payload = marker.read_bytes()
        try:
            checkpoint = OcrPhase1Checkpoint.model_validate_json(payload)
        except ValueError as error:
            raise ValueError(f"invalid checkpoint commit marker: {marker}") from error
        parsed.append((candidate, checkpoint, hashlib.sha256(payload).hexdigest()))
    if not parsed:
        raise ValueError("checkpoint root contains no committed checkpoint bundle")
    parsed.sort(key=lambda item: item[1].checkpoint_sequence)
    sequences = [item[1].checkpoint_sequence for item in parsed]
    if len(sequences) != len(set(sequences)):
        raise ValueError("checkpoint sequence is duplicated")
    if len(parsed) > 1:
        for previous, current in zip(parsed, parsed[1:], strict=False):
            if (
                current[1].checkpoint_sequence != previous[1].checkpoint_sequence + 1
                or current[1].previous_checkpoint_sha256 != previous[2]
            ):
                raise ValueError("checkpoint hash chain is broken")
    return parsed


def resolve_checkpoint_bundle(root: Path) -> Path:
    """Resolve an exact bundle or select the latest bundle in a shard checkpoint root."""

    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError("checkpoint root is unavailable")
    return _checkpoint_chain(resolved)[-1][0]


def _shard_binding(
    *,
    source_manifest: Path,
    global_manifest: Path,
    frame_manifest: Path,
    shard_id: str,
    config_sha256: str,
    tracking_config: TrackingConfig,
) -> tuple[OcrGlobalShardManifest, tuple[str, ...]]:
    verify_global_shard_structure(
        source_manifest=source_manifest,
        global_manifest=global_manifest,
        expected_config_sha256=config_sha256,
        tracking_config=tracking_config,
    )
    manifest = OcrGlobalShardManifest.model_validate_json(
        global_manifest.read_text(encoding="utf-8")
    )
    matches = [item for item in manifest.shards if item.shard_id == shard_id]
    if len(matches) != 1:
        raise ValueError("checkpoint shard is absent or duplicated in global manifest")
    shard = matches[0]
    if sha256_file(frame_manifest) != shard.manifest_sha256:
        raise ValueError("checkpoint frame manifest differs from global shard binding")
    return manifest, tuple(shard.video_ids)


def _verify_running_detection(
    *,
    logical_output: Path,
    payload: Path,
    receipt_file: Path,
    frame_manifest: Path,
    data_root: Path,
    crop_config: CropConfig,
    run_id: str,
    config_sha256: str,
    identity: Phase1Identity,
    shard_id: str,
    tracking_config: TrackingConfig,
) -> tuple[OcrPhase1Receipt, dict[str, int], list[str]]:
    del logical_output
    receipt = _read_receipt(receipt_file)
    input_hash, frames = _load_frame_manifest_with_hash(frame_manifest, data_root)
    _check_receipt_identity(
        receipt,
        stage="detect_crop",
        run_id=run_id,
        input_hash=input_hash,
        config_hash=config_sha256,
        identity=identity,
        shard_id=shard_id,
        shard_manifest_sha256=input_hash,
        resource_limits_sha256=tracking_config.resource_limits_sha256,
    )
    if receipt.status != "running":
        raise ValueError("detection checkpoint does not contain a running receipt")
    if payload.stat().st_size != receipt.committed_bytes:
        raise ValueError("checkpoint detection prefix differs from committed byte offset")
    if sha256_file(payload) != receipt.committed_sha256:
        raise ValueError("checkpoint detection prefix hash differs from running receipt")
    records = _load_records(payload, OcrDetectionFrameRecord)
    _validate_detection_prefix(
        records,
        frames,
        run_id=run_id,
        config_sha256=config_sha256,
        identity=identity,
        crop_config=crop_config,
    )
    counts = _detection_counts(records)
    if len(records) != receipt.committed_records or counts != receipt.record_counts:
        raise ValueError("checkpoint detection prefix count differs from running receipt")
    _verify_all_frame_source_hashes(frames)
    if sha256_file(frame_manifest) != input_hash:
        raise ValueError("frame manifest changed during checkpoint detection replay")
    return receipt, counts, [ref.frame_uid for ref, _ in frames]


def _copy_metadata(role: str, relpath: str, payload: bytes) -> OcrCheckpointFile:
    return OcrCheckpointFile(
        role=role,
        bundle_relpath=relpath,
        byte_size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _capture_artifact(
    *,
    stage: str,
    logical_path: Path,
    artifact_root: Path,
    receipt: OcrPhase1Receipt,
    payload: bytes,
    receipt_payload: bytes,
) -> OcrCheckpointArtifact:
    relative = _artifact_relative(logical_path, artifact_root)
    if receipt.status == "running":
        payload_relpath = f"state/{relative}.partial"
    else:
        payload_relpath = f"state/{relative}"
    return OcrCheckpointArtifact(
        stage=stage,
        artifact_relpath=relative,
        payload_relpath=payload_relpath,
        byte_size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        committed_records=receipt.committed_records,
        receipt_relpath=f"state/{relative}.receipt.json",
        receipt_sha256=hashlib.sha256(receipt_payload).hexdigest(),
        receipt_status=receipt.status,
    )


def _validate_and_capture_state(
    *,
    artifacts: CheckpointArtifactPaths,
    artifact_root: Path,
    frame_manifest: Path,
    data_root: Path,
    crop_config: CropConfig,
    run_id: str,
    config_sha256: str,
    identity: Phase1Identity,
    tracking_config: TrackingConfig,
    shard_id: str,
) -> tuple[
    str,
    str | None,
    OcrCheckpointCounts,
    list[tuple[OcrCheckpointArtifact, bytes, bytes]],
]:
    detection_receipt_path = receipt_path_for(artifacts.detections)
    detection_receipt = _read_receipt(detection_receipt_path)
    captured: list[tuple[OcrCheckpointArtifact, bytes, bytes]] = []
    if detection_receipt.status == "running":
        partial = artifacts.detections.with_suffix(artifacts.detections.suffix + ".partial")
        payload = _read_exact_prefix(partial, detection_receipt.committed_bytes)
        if hashlib.sha256(payload).hexdigest() != detection_receipt.committed_sha256:
            raise ValueError("working detection prefix differs from committed receipt")
        receipt_payload = detection_receipt_path.read_bytes()
        # Verify a tail-free snapshot, not the mutable working file itself.
        with tempfile.TemporaryDirectory(prefix="ocr-checkpoint-prefix-") as directory:
            snapshot = Path(directory) / "detections.jsonl.partial"
            snapshot.write_bytes(payload)
            _, detect_counts, frame_uids = _verify_running_detection(
                logical_output=artifacts.detections,
                payload=snapshot,
                receipt_file=detection_receipt_path,
                frame_manifest=frame_manifest,
                data_root=data_root,
                crop_config=crop_config,
                run_id=run_id,
                config_sha256=config_sha256,
                identity=identity,
                shard_id=shard_id,
                tracking_config=tracking_config,
            )
        metadata = _capture_artifact(
            stage="detect",
            logical_path=artifacts.detections,
            artifact_root=artifact_root,
            receipt=detection_receipt,
            payload=payload,
            receipt_payload=receipt_payload,
        )
        captured.append((metadata, payload, receipt_payload))
        next_frame = (
            frame_uids[detection_receipt.committed_records]
            if detection_receipt.committed_records < len(frame_uids)
            else None
        )
        return (
            "detect",
            next_frame,
            OcrCheckpointCounts(
                frames=detect_counts["frames"],
                detections=detect_counts["detections"],
                trajectories=0,
                representatives=0,
            ),
            captured,
        )

    detect_counts = verify_detection_artifact(
        output=artifacts.detections,
        frame_manifest=frame_manifest,
        data_root=data_root,
        crop_config=crop_config,
        expected_run_id=run_id,
        expected_config_sha256=config_sha256,
        expected_identity=identity,
        expected_shard_id=shard_id,
        tracking_config=tracking_config,
    )
    detect_payload = artifacts.detections.read_bytes()
    detect_receipt_payload = detection_receipt_path.read_bytes()
    captured.append(
        (
            _capture_artifact(
                stage="detect",
                logical_path=artifacts.detections,
                artifact_root=artifact_root,
                receipt=detection_receipt,
                payload=detect_payload,
                receipt_payload=detect_receipt_payload,
            ),
            detect_payload,
            detect_receipt_payload,
        )
    )

    trajectory_receipt_path = receipt_path_for(artifacts.trajectories)
    if not trajectory_receipt_path.is_file():
        return (
            "track",
            None,
            OcrCheckpointCounts(**detect_counts, trajectories=0, representatives=0),
            captured,
        )
    trajectory_receipt = _read_receipt(trajectory_receipt_path)
    if trajectory_receipt.status != "completed":
        raise ValueError("checkpoint cannot contain a running trajectory receipt")
    track_counts = verify_tracking_artifact(
        detections=artifacts.detections,
        trajectories=artifacts.trajectories,
        expected_run_id=run_id,
        expected_config_sha256=config_sha256,
        expected_identity=identity,
        tracking_config=tracking_config,
    )
    trajectory_payload = artifacts.trajectories.read_bytes()
    trajectory_receipt_payload = trajectory_receipt_path.read_bytes()
    captured.append(
        (
            _capture_artifact(
                stage="track",
                logical_path=artifacts.trajectories,
                artifact_root=artifact_root,
                receipt=trajectory_receipt,
                payload=trajectory_payload,
                receipt_payload=trajectory_receipt_payload,
            ),
            trajectory_payload,
            trajectory_receipt_payload,
        )
    )

    representative_receipt_path = receipt_path_for(artifacts.representatives)
    if not representative_receipt_path.is_file():
        return (
            "select",
            None,
            OcrCheckpointCounts(
                **detect_counts,
                trajectories=track_counts["trajectories"],
                representatives=0,
            ),
            captured,
        )
    representative_receipt = _read_receipt(representative_receipt_path)
    if representative_receipt.status != "completed":
        raise ValueError("checkpoint cannot contain a running representative receipt")
    linked_counts = verify_linked_artifacts(
        detections=artifacts.detections,
        trajectories=artifacts.trajectories,
        representatives=artifacts.representatives,
        expected_run_id=run_id,
        expected_config_sha256=config_sha256,
        expected_identity=identity,
        tracking_config=tracking_config,
    )
    representative_payload = artifacts.representatives.read_bytes()
    representative_receipt_payload = representative_receipt_path.read_bytes()
    captured.append(
        (
            _capture_artifact(
                stage="select",
                logical_path=artifacts.representatives,
                artifact_root=artifact_root,
                receipt=representative_receipt,
                payload=representative_payload,
                receipt_payload=representative_receipt_payload,
            ),
            representative_payload,
            representative_receipt_payload,
        )
    )
    return (
        "completed",
        None,
        OcrCheckpointCounts(
            **detect_counts,
            trajectories=linked_counts["trajectories"],
            representatives=linked_counts["representatives"],
        ),
        captured,
    )


def publish_checkpoint(
    *,
    checkpoint_root: Path,
    artifact_root: Path,
    artifacts: CheckpointArtifactPaths,
    source_manifest: Path,
    global_manifest: Path,
    frame_manifest: Path,
    data_root: Path,
    run_id: str,
    config_sha256: str,
    git_commit_sha: str,
    identity: Phase1Identity,
    crop_config: CropConfig,
    tracking_config: TrackingConfig,
    shard_id: str,
    created_at: datetime | None = None,
    fault_injector: FaultInjector | None = None,
) -> Path:
    """Verify and atomically publish a new immutable checkpoint bundle."""

    source_manifest = source_manifest.resolve()
    global_manifest = global_manifest.resolve()
    frame_manifest = frame_manifest.resolve()
    artifact_root = artifact_root.resolve()
    checkpoint_root = checkpoint_root.resolve()
    manifest, video_ids = _shard_binding(
        source_manifest=source_manifest,
        global_manifest=global_manifest,
        frame_manifest=frame_manifest,
        shard_id=shard_id,
        config_sha256=config_sha256,
        tracking_config=tracking_config,
    )
    _, source_frames = _load_frame_manifest_with_hash(frame_manifest, data_root)
    _verify_all_frame_source_hashes(source_frames)
    structural_paths = [
        source_manifest,
        global_manifest,
        global_shard_receipt_path(global_manifest),
        *[(global_manifest.parent / item.manifest_relpath).resolve() for item in manifest.shards],
        *[path for _, path in source_frames],
    ]
    structural_baseline = [(path, sha256_file(path)) for path in structural_paths]
    stage, next_frame, counts, captured_artifacts = _validate_and_capture_state(
        artifacts=artifacts,
        artifact_root=artifact_root,
        frame_manifest=frame_manifest,
        data_root=data_root,
        crop_config=crop_config,
        run_id=run_id,
        config_sha256=config_sha256,
        identity=identity,
        tracking_config=tracking_config,
        shard_id=shard_id,
    )

    manifest_sources = (
        ("source_manifest", "manifests/source.frames.jsonl", source_manifest),
        ("global_manifest", "manifests/global-shards.json", global_manifest),
        (
            "global_receipt",
            "manifests/global-shards.json.receipt.json",
            global_shard_receipt_path(global_manifest),
        ),
        ("shard_manifest", "manifests/shard.frames.jsonl", frame_manifest),
    )
    captured_files: list[tuple[OcrCheckpointFile, bytes]] = []
    for role, relpath, path in manifest_sources:
        payload = path.read_bytes()
        _assert_secret_free(payload, label=role)
        captured_files.append((_copy_metadata(role, relpath, payload), payload))
    for metadata, payload, receipt_payload in captured_artifacts:
        _assert_secret_free(payload, label=f"{metadata.stage} artifact")
        _assert_secret_free(receipt_payload, label=f"{metadata.stage} receipt")

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    _fsync_directory(checkpoint_root.parent)
    committed_children = [
        child
        for child in checkpoint_root.iterdir()
        if child.is_dir() and child.name.startswith("checkpoint-")
    ]
    previous_entries = _checkpoint_chain(checkpoint_root) if committed_children else []
    sequence = previous_entries[-1][1].checkpoint_sequence + 1 if previous_entries else 1
    previous_hash = previous_entries[-1][2] if previous_entries else None
    timestamp = created_at or datetime.now(UTC).replace(microsecond=0)
    checkpoint = OcrPhase1Checkpoint(
        run_id=run_id,
        config_sha256=config_sha256,
        git_commit_sha=git_commit_sha,
        detector_id=identity.detector_id,
        detector_revision=identity.detector_revision,
        detector_tree_sha256=identity.detector_tree_sha256,
        runtime_identity_sha256=identity.runtime_identity_sha256,
        resource_limits_sha256=tracking_config.resource_limits_sha256,
        source_manifest_sha256=manifest.source_manifest_sha256,
        global_manifest_sha256=sha256_file(global_manifest),
        shard_manifest_sha256=sha256_file(frame_manifest),
        shard_id=shard_id,
        video_ids=video_ids,
        stage=stage,
        files=tuple(item[0] for item in captured_files),
        artifacts=tuple(item[0] for item in captured_artifacts),
        counts=counts,
        next_frame_uid=next_frame,
        next_stage=stage,
        created_at=timestamp,
        checkpoint_sequence=sequence,
        previous_checkpoint_sha256=previous_hash,
    )
    if previous_entries and _replay_identity(previous_entries[-1][1]) == _replay_identity(
        checkpoint
    ):
        verify_checkpoint_bundle(
            checkpoint_root=previous_entries[-1][0],
            source_manifest=source_manifest,
            global_manifest=global_manifest,
            frame_manifest=frame_manifest,
            data_root=data_root,
            run_id=run_id,
            config_sha256=config_sha256,
            git_commit_sha=git_commit_sha,
            identity=identity,
            crop_config=crop_config,
            tracking_config=tracking_config,
            shard_id=shard_id,
        )
        return previous_entries[-1][0]
    marker_payload = _marker_bytes(checkpoint)
    _assert_secret_free(marker_payload, label=CHECKPOINT_MARKER)
    marker_digest = hashlib.sha256(marker_payload).hexdigest()
    final = checkpoint_root / f"checkpoint-{sequence:06d}-{marker_digest[:16]}"
    if final.exists():
        raise FileExistsError(
            "checkpoint target already exists; immutable bundles are not overwritten"
        )
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-publishing-", dir=checkpoint_root))
    for metadata, payload in captured_files:
        _write_file(_safe_child(temporary, metadata.bundle_relpath), payload)
    for metadata, payload, receipt_payload in captured_artifacts:
        _write_file(_safe_child(temporary, metadata.payload_relpath), payload)
        _write_file(_safe_child(temporary, metadata.receipt_relpath), receipt_payload)
    _fsync_directory(temporary)
    if fault_injector is not None:
        fault_injector("after_checkpoint_files_fsync")

    # Recheck every captured source immediately before publishing the commit marker.
    for metadata, payload in captured_files:
        source_path = dict((role, path) for role, _, path in manifest_sources)[metadata.role]
        if source_path.read_bytes() != payload:
            raise ValueError(f"checkpoint source changed before publication: {metadata.role}")
    for metadata, payload, receipt_payload in captured_artifacts:
        logical = _safe_child(artifact_root, metadata.artifact_relpath)
        source_payload = (
            logical.with_suffix(logical.suffix + ".partial")
            if metadata.receipt_status == "running"
            else logical
        )
        if metadata.receipt_status == "running":
            if _read_exact_prefix(source_payload, metadata.byte_size) != payload:
                raise ValueError("committed detection prefix changed before checkpoint publication")
        elif source_payload.read_bytes() != payload:
            raise ValueError("artifact changed before checkpoint publication")
        if receipt_path_for(logical).read_bytes() != receipt_payload:
            raise ValueError("artifact receipt changed before checkpoint publication")
    if any(sha256_file(path) != digest for path, digest in structural_baseline):
        raise ValueError("source/shard input changed before checkpoint publication")
    verify_global_shard_structure(
        source_manifest=source_manifest,
        global_manifest=global_manifest,
        expected_config_sha256=config_sha256,
        tracking_config=tracking_config,
    )
    if any(sha256_file(path) != digest for path, digest in structural_baseline):
        raise ValueError("source/shard input changed during checkpoint structural replay")
    if fault_injector is not None:
        fault_injector("before_checkpoint_commit_marker")
    _write_file(temporary / CHECKPOINT_MARKER, marker_payload)
    _fsync_directory(temporary)
    if fault_injector is not None:
        fault_injector("after_checkpoint_marker_fsync_before_rename")
    os.rename(temporary, final)
    _fsync_directory(checkpoint_root)
    if fault_injector is not None:
        fault_injector("after_checkpoint_directory_rename")
    return final


def _validate_checkpoint_files(bundle: Path, checkpoint: OcrPhase1Checkpoint) -> dict[str, Path]:
    expected_relpaths = {CHECKPOINT_MARKER}
    expected_relpaths.update(item.bundle_relpath for item in checkpoint.files)
    expected_relpaths.update(
        path
        for item in checkpoint.artifacts
        for path in (item.payload_relpath, item.receipt_relpath)
    )
    actual_relpaths = {
        path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()
    }
    if actual_relpaths != expected_relpaths:
        raise ValueError("checkpoint bundle contains missing or undeclared files")
    paths: dict[str, Path] = {}
    for item in checkpoint.files:
        path = _safe_child(bundle, item.bundle_relpath)
        if not path.is_file() or path.stat().st_size != item.byte_size:
            raise ValueError(f"checkpoint file is missing or truncated: {item.role}")
        if sha256_file(path) != item.sha256:
            raise ValueError(f"checkpoint file checksum drift: {item.role}")
        _assert_secret_free(path.read_bytes(), label=item.role)
        paths[item.role] = path
    for item in checkpoint.artifacts:
        payload = _safe_child(bundle, item.payload_relpath)
        receipt = _safe_child(bundle, item.receipt_relpath)
        if (
            not payload.is_file()
            or not receipt.is_file()
            or payload.stat().st_size != item.byte_size
        ):
            raise ValueError(f"checkpoint {item.stage} artifact is missing or truncated")
        if sha256_file(payload) != item.sha256 or sha256_file(receipt) != item.receipt_sha256:
            raise ValueError(f"checkpoint {item.stage} artifact/receipt checksum drift")
        receipt_model = _read_receipt(receipt)
        expected_receipt_stage = {
            "detect": "detect_crop",
            "track": "track",
            "select": "select_representatives",
        }[item.stage]
        if (
            receipt_model.stage != expected_receipt_stage
            or receipt_model.status != item.receipt_status
            or receipt_model.committed_bytes != item.byte_size
            or receipt_model.committed_records != item.committed_records
            or receipt_model.committed_sha256 != item.sha256
        ):
            raise ValueError(f"checkpoint {item.stage} metadata differs from its receipt")
        _assert_secret_free(payload.read_bytes(), label=f"{item.stage} artifact")
        _assert_secret_free(receipt.read_bytes(), label=f"{item.stage} receipt")
    return paths


def verify_checkpoint_bundle(
    *,
    checkpoint_root: Path,
    source_manifest: Path,
    global_manifest: Path,
    frame_manifest: Path,
    data_root: Path,
    run_id: str,
    config_sha256: str,
    git_commit_sha: str,
    identity: Phase1Identity,
    crop_config: CropConfig,
    tracking_config: TrackingConfig,
    shard_id: str,
) -> OcrPhase1Checkpoint:
    """Verify identity, hashes and semantic replay without modifying the bundle."""

    bundle = resolve_checkpoint_bundle(checkpoint_root)
    marker = bundle / CHECKPOINT_MARKER
    checkpoint = OcrPhase1Checkpoint.model_validate_json(marker.read_bytes())
    expected_identity = (
        run_id,
        config_sha256,
        git_commit_sha,
        identity.detector_id,
        identity.detector_revision,
        identity.detector_tree_sha256,
        identity.runtime_identity_sha256,
        tracking_config.resource_limits_sha256,
        shard_id,
    )
    actual_identity = (
        checkpoint.run_id,
        checkpoint.config_sha256,
        checkpoint.git_commit_sha,
        checkpoint.detector_id,
        checkpoint.detector_revision,
        checkpoint.detector_tree_sha256,
        checkpoint.runtime_identity_sha256,
        checkpoint.resource_limits_sha256,
        checkpoint.shard_id,
    )
    if actual_identity != expected_identity:
        raise ValueError("checkpoint run/config/git/model/runtime/resource/shard identity drift")
    manifest, video_ids = _shard_binding(
        source_manifest=source_manifest,
        global_manifest=global_manifest,
        frame_manifest=frame_manifest,
        shard_id=shard_id,
        config_sha256=config_sha256,
        tracking_config=tracking_config,
    )
    if (
        checkpoint.source_manifest_sha256 != manifest.source_manifest_sha256
        or checkpoint.global_manifest_sha256 != sha256_file(global_manifest)
        or checkpoint.shard_manifest_sha256 != sha256_file(frame_manifest)
        or checkpoint.video_ids != video_ids
    ):
        raise ValueError("checkpoint source/global/shard/video ownership binding drift")

    # Capture one immutable baseline before any byte or semantic verification.
    baseline_paths = [marker]
    baseline_paths += [_safe_child(bundle, item.bundle_relpath) for item in checkpoint.files]
    baseline_paths += [
        _safe_child(bundle, path)
        for item in checkpoint.artifacts
        for path in (item.payload_relpath, item.receipt_relpath)
    ]
    baseline_paths += [
        source_manifest,
        global_manifest,
        global_shard_receipt_path(global_manifest),
    ]
    global_root = global_manifest.parent.resolve()
    baseline_paths += [(global_root / item.manifest_relpath).resolve() for item in manifest.shards]
    _, frames = _load_frame_manifest_with_hash(frame_manifest, data_root)
    baseline_paths += [path for _, path in frames]
    baseline = [(path, sha256_file(path)) for path in baseline_paths]

    bundled = _validate_checkpoint_files(bundle, checkpoint)
    expected_manifest_hashes = {
        "source_manifest": sha256_file(source_manifest),
        "global_manifest": sha256_file(global_manifest),
        "global_receipt": sha256_file(global_shard_receipt_path(global_manifest)),
        "shard_manifest": sha256_file(frame_manifest),
    }
    if any(
        sha256_file(bundled[role]) != digest for role, digest in expected_manifest_hashes.items()
    ):
        raise ValueError("checkpoint manifest snapshot differs from current verified inputs")

    logical: dict[str, Path] = {}
    for artifact in checkpoint.artifacts:
        payload = _safe_child(bundle, artifact.payload_relpath)
        logical[artifact.stage] = (
            payload.with_suffix("") if artifact.receipt_status == "running" else payload
        )
    detect = checkpoint.artifacts[0]
    if detect.receipt_status == "running":
        receipt, detect_counts, frame_uids = _verify_running_detection(
            logical_output=logical["detect"],
            payload=_safe_child(bundle, detect.payload_relpath),
            receipt_file=_safe_child(bundle, detect.receipt_relpath),
            frame_manifest=frame_manifest,
            data_root=data_root,
            crop_config=crop_config,
            run_id=run_id,
            config_sha256=config_sha256,
            identity=identity,
            shard_id=shard_id,
            tracking_config=tracking_config,
        )
        next_frame = (
            frame_uids[receipt.committed_records]
            if receipt.committed_records < len(frame_uids)
            else None
        )
    else:
        detect_counts = verify_detection_artifact(
            output=logical["detect"],
            frame_manifest=frame_manifest,
            data_root=data_root,
            crop_config=crop_config,
            expected_run_id=run_id,
            expected_config_sha256=config_sha256,
            expected_identity=identity,
            expected_shard_id=shard_id,
            tracking_config=tracking_config,
        )
        next_frame = None
    trajectory_count = 0
    representative_count = 0
    if checkpoint.stage == "select":
        track_counts = verify_tracking_artifact(
            detections=logical["detect"],
            trajectories=logical["track"],
            expected_run_id=run_id,
            expected_config_sha256=config_sha256,
            expected_identity=identity,
            tracking_config=tracking_config,
        )
        trajectory_count = track_counts["trajectories"]
    elif checkpoint.stage == "completed":
        linked_counts = verify_linked_artifacts(
            detections=logical["detect"],
            trajectories=logical["track"],
            representatives=logical["select"],
            expected_run_id=run_id,
            expected_config_sha256=config_sha256,
            expected_identity=identity,
            tracking_config=tracking_config,
        )
        trajectory_count = linked_counts["trajectories"]
        representative_count = linked_counts["representatives"]
    expected_counts = OcrCheckpointCounts(
        frames=detect_counts["frames"],
        detections=detect_counts["detections"],
        trajectories=trajectory_count,
        representatives=representative_count,
    )
    if checkpoint.counts != expected_counts or checkpoint.next_frame_uid != next_frame:
        raise ValueError("checkpoint counts or next-frame position differ from semantic replay")

    # Final immutable-baseline recheck covers every bundle byte and every current
    # manifest/source image used by semantic replay.
    verify_global_shard_structure(
        source_manifest=source_manifest,
        global_manifest=global_manifest,
        expected_config_sha256=config_sha256,
        tracking_config=tracking_config,
    )
    if any(sha256_file(path) != digest for path, digest in baseline):
        raise ValueError("checkpoint verification input changed during semantic replay")
    return checkpoint


def _restore_file(source: Path, destination: Path, expected_sha256: str) -> None:
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != expected_sha256:
            raise ValueError(f"restore target conflicts with checkpoint: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.restore-{os.getpid()}.tmp")
    if temporary.exists():
        raise ValueError(f"stale restore temporary requires operator review: {temporary}")
    with source.open("rb") as reader, temporary.open("xb") as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    if sha256_file(temporary) != expected_sha256:
        raise ValueError("restored temporary checksum mismatch")
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def restore_checkpoint_history(
    *,
    checkpoint_root: Path,
    checkpoint_destination_root: Path,
    source_manifest: Path,
    global_manifest: Path,
    frame_manifest: Path,
    data_root: Path,
    run_id: str,
    config_sha256: str,
    git_commit_sha: str,
    identity: Phase1Identity,
    crop_config: CropConfig,
    tracking_config: TrackingConfig,
    shard_id: str,
) -> Path:
    """Copy a verified checkpoint chain into writable storage without rewriting it."""

    source_root = checkpoint_root.expanduser().resolve()
    destination_root = checkpoint_destination_root.expanduser().resolve()
    try:
        destination_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("checkpoint history destination cannot be inside its read-only source")
    source_chain = _checkpoint_chain(source_root)
    if (source_root / CHECKPOINT_MARKER).is_file() and (
        source_chain[0][1].checkpoint_sequence != 1
        or source_chain[0][1].previous_checkpoint_sha256 is not None
    ):
        raise ValueError("cross-session restore requires the complete checkpoint history root")
    if source_chain[0][1].checkpoint_sequence != 1 or (
        source_chain[0][1].previous_checkpoint_sha256 is not None
    ):
        raise ValueError("checkpoint history does not begin at sequence 1")
    for bundle, _, _ in source_chain:
        verify_checkpoint_bundle(
            checkpoint_root=bundle,
            source_manifest=source_manifest,
            global_manifest=global_manifest,
            frame_manifest=frame_manifest,
            data_root=data_root,
            run_id=run_id,
            config_sha256=config_sha256,
            git_commit_sha=git_commit_sha,
            identity=identity,
            crop_config=crop_config,
            tracking_config=tracking_config,
            shard_id=shard_id,
        )

    destination_root.mkdir(parents=True, exist_ok=True)
    _fsync_directory(destination_root.parent)
    destination_children = [
        child
        for child in destination_root.iterdir()
        if child.is_dir() and child.name.startswith("checkpoint-")
    ]
    destination_chain = _checkpoint_chain(destination_root) if destination_children else []
    if len(destination_chain) > len(source_chain):
        raise ValueError("writable checkpoint history is ahead of the restore source")
    for destination_item, source_item in zip(destination_chain, source_chain, strict=False):
        if destination_item[2] != source_item[2]:
            raise ValueError("writable checkpoint history diverges from restore source")

    for source_bundle, checkpoint, _ in source_chain[len(destination_chain) :]:
        target = destination_root / source_bundle.name
        if target.exists():
            raise FileExistsError("checkpoint history target exists outside the verified prefix")
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-restoring-", dir=destination_root))
        regular_relpaths = [item.bundle_relpath for item in checkpoint.files]
        regular_relpaths += [
            path
            for item in checkpoint.artifacts
            for path in (item.payload_relpath, item.receipt_relpath)
        ]
        for relative in regular_relpaths:
            _write_file(
                _safe_child(temporary, relative),
                _safe_child(source_bundle, relative).read_bytes(),
            )
        _write_file(
            temporary / CHECKPOINT_MARKER,
            (source_bundle / CHECKPOINT_MARKER).read_bytes(),
        )
        _fsync_directory(temporary)
        os.rename(temporary, target)
        _fsync_directory(destination_root)
    copied_chain = _checkpoint_chain(destination_root)
    if copied_chain[-1][2] != source_chain[-1][2]:
        raise ValueError("restored checkpoint history does not match its source")
    return copied_chain[-1][0]


def restore_checkpoint(
    *,
    checkpoint_root: Path,
    artifact_root: Path,
    source_manifest: Path,
    global_manifest: Path,
    frame_manifest: Path,
    data_root: Path,
    run_id: str,
    config_sha256: str,
    git_commit_sha: str,
    identity: Phase1Identity,
    crop_config: CropConfig,
    tracking_config: TrackingConfig,
    shard_id: str,
    checkpoint_destination_root: Path | None = None,
) -> OcrPhase1Checkpoint:
    """Verify a read-only bundle, then idempotently copy it into working storage."""

    bundle = resolve_checkpoint_bundle(checkpoint_root)
    artifact_root = artifact_root.resolve()
    try:
        artifact_root.relative_to(bundle.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("restore target cannot be inside the read-only checkpoint bundle")
    checkpoint = verify_checkpoint_bundle(
        checkpoint_root=checkpoint_root,
        source_manifest=source_manifest,
        global_manifest=global_manifest,
        frame_manifest=frame_manifest,
        data_root=data_root,
        run_id=run_id,
        config_sha256=config_sha256,
        git_commit_sha=git_commit_sha,
        identity=identity,
        crop_config=crop_config,
        tracking_config=tracking_config,
        shard_id=shard_id,
    )
    if checkpoint_destination_root is not None:
        restore_checkpoint_history(
            checkpoint_root=checkpoint_root,
            checkpoint_destination_root=checkpoint_destination_root,
            source_manifest=source_manifest,
            global_manifest=global_manifest,
            frame_manifest=frame_manifest,
            data_root=data_root,
            run_id=run_id,
            config_sha256=config_sha256,
            git_commit_sha=git_commit_sha,
            identity=identity,
            crop_config=crop_config,
            tracking_config=tracking_config,
            shard_id=shard_id,
        )
    artifact_root.mkdir(parents=True, exist_ok=True)
    payload_destinations: list[tuple[Path, Path, str]] = []
    receipt_destinations: list[tuple[Path, Path, str]] = []
    for artifact in checkpoint.artifacts:
        logical = _safe_child(artifact_root, artifact.artifact_relpath)
        destination = (
            logical.with_suffix(logical.suffix + ".partial")
            if artifact.receipt_status == "running"
            else logical
        )
        payload_destinations.append(
            (_safe_child(bundle, artifact.payload_relpath), destination, artifact.sha256)
        )
        receipt_destinations.append(
            (
                _safe_child(bundle, artifact.receipt_relpath),
                receipt_path_for(logical),
                artifact.receipt_sha256,
            )
        )
    for source, destination, digest in payload_destinations:
        _restore_file(source, destination, digest)
    # Receipts are commit markers in the working tree and are always restored last.
    for source, destination, digest in receipt_destinations:
        _restore_file(source, destination, digest)
    _fsync_directory(artifact_root)
    return checkpoint
