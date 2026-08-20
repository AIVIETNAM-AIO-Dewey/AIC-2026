"""Deterministic whole-video shard planning and global coverage verification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from aic2026.common import atomic_write_json, sha256_file, write_jsonl_atomic
from aic2026.contracts import (
    FrameRef,
    OcrFrameShard,
    OcrGlobalShardManifest,
    OcrGlobalShardReceipt,
)

from .geometry import CropConfig
from .tracking import TrackingConfig, natural_key


@dataclass(frozen=True, slots=True)
class OcrShardArtifactBundle:
    detections: Path
    trajectories: Path
    representatives: Path


def _frame_manifest_snapshot(path: Path) -> tuple[bytes, str, list[FrameRef]]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"frame manifest is unavailable: {path}") from error
    digest = hashlib.sha256(payload).hexdigest()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("frame manifest must be UTF-8") from error
    records: list[FrameRef] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid frame manifest JSON at line {line_number}") from error
        records.append(FrameRef.model_validate(value))
    if not records:
        raise ValueError("frame manifest is empty")
    frame_uids = [item.frame_uid for item in records]
    frame_keys = [(item.video_id, item.frame_idx) for item in records]
    if len(frame_uids) != len(set(frame_uids)):
        raise ValueError("duplicate frame_uid in frame manifest")
    if len(frame_keys) != len(set(frame_keys)):
        raise ValueError("duplicate video_id/frame_idx in frame manifest")
    ordered = sorted(
        records,
        key=lambda item: (natural_key(item.video_id), item.frame_idx, natural_key(item.frame_uid)),
    )
    return payload, digest, ordered


def global_shard_receipt_path(global_manifest: Path) -> Path:
    return global_manifest.with_suffix(global_manifest.suffix + ".receipt.json")


def plan_frame_shards(
    *,
    source_manifest: Path,
    output_dir: Path,
    config_sha256: str,
    tracking_config: TrackingConfig,
) -> tuple[Path, OcrGlobalShardManifest]:
    """First-fit-decreasing bin packing without ever splitting a video."""

    _, source_hash, frames = _frame_manifest_snapshot(source_manifest)
    maximum_frames_per_shard = tracking_config.maximum_frames_per_shard
    groups: dict[str, list[FrameRef]] = {}
    for frame in frames:
        groups.setdefault(frame.video_id, []).append(frame)
    oversized = [
        video_id for video_id, members in groups.items() if len(members) > maximum_frames_per_shard
    ]
    if oversized:
        video_id = min(oversized, key=natural_key)
        raise ValueError(
            f"video {video_id!r} has {len(groups[video_id])} frames, exceeding shard limit "
            f"{maximum_frames_per_shard}; stateful cross-shard tracking is required"
        )

    bins: list[list[str]] = []
    bin_counts: list[int] = []
    for video_id in sorted(groups, key=lambda key: (-len(groups[key]), natural_key(key))):
        count = len(groups[video_id])
        destination = next(
            (
                index
                for index, current in enumerate(bin_counts)
                if current + count <= maximum_frames_per_shard
            ),
            None,
        )
        if destination is None:
            bins.append([video_id])
            bin_counts.append(count)
        else:
            bins[destination].append(video_id)
            bin_counts[destination] += count

    source_resolved = source_manifest.resolve()
    output_resolved = output_dir.resolve()
    shard_targets = [
        (output_resolved / f"shard-{index:06d}.frames.jsonl").resolve()
        for index in range(1, len(bins) + 1)
    ]
    manifest_path = (output_resolved / "global-shards.json").resolve()
    receipt_target = global_shard_receipt_path(manifest_path).resolve()
    targets = [*shard_targets, manifest_path, receipt_target]
    if len(targets) != len(set(targets)) or source_resolved in targets:
        raise ValueError("shard planner target collides with source or another commit artifact")
    if output_resolved.exists():
        raise FileExistsError("shard planner requires a fresh output directory")

    planned_shards: list[tuple[str, list[str], list[FrameRef], Path]] = []
    preflight_entries: list[OcrFrameShard] = []
    for index, video_ids in enumerate(bins, start=1):
        shard_id = f"shard-{index:06d}"
        membership = set(video_ids)
        shard_frames = [item for item in frames if item.video_id in membership]
        shard_path = shard_targets[index - 1]
        planned_shards.append((shard_id, video_ids, shard_frames, shard_path))
        preflight_entries.append(
            OcrFrameShard(
                shard_id=shard_id,
                manifest_relpath=shard_path.relative_to(output_resolved).as_posix(),
                manifest_sha256="0" * 64,
                video_ids=sorted(video_ids),
                frame_uids=[item.frame_uid for item in shard_frames],
                frame_count=len(shard_frames),
            )
        )
    OcrGlobalShardManifest(
        source_manifest_sha256=source_hash,
        config_sha256=config_sha256,
        maximum_frames_per_shard=tracking_config.maximum_frames_per_shard,
        shards=preflight_entries,
    )
    OcrGlobalShardReceipt(
        source_manifest_sha256=source_hash,
        config_sha256=config_sha256,
        global_manifest_sha256="0" * 64,
        shard_count=len(preflight_entries),
        frame_count=len(frames),
    )
    if sha256_file(source_resolved) != source_hash:
        raise ValueError("source manifest changed before shard publication")
    output_resolved.mkdir(parents=True, exist_ok=False)
    entries: list[OcrFrameShard] = []
    for shard_id, video_ids, shard_frames, shard_path in planned_shards:
        write_jsonl_atomic(shard_path, shard_frames)
        entries.append(
            OcrFrameShard(
                shard_id=shard_id,
                manifest_relpath=shard_path.relative_to(output_resolved).as_posix(),
                manifest_sha256=sha256_file(shard_path),
                video_ids=sorted(video_ids),
                frame_uids=[item.frame_uid for item in shard_frames],
                frame_count=len(shard_frames),
            )
        )
    global_manifest = OcrGlobalShardManifest(
        source_manifest_sha256=source_hash,
        config_sha256=config_sha256,
        maximum_frames_per_shard=tracking_config.maximum_frames_per_shard,
        shards=entries,
    )
    if sha256_file(source_resolved) != source_hash:
        raise ValueError("source manifest changed during shard planning")
    atomic_write_json(manifest_path, global_manifest.model_dump(mode="json"))
    if sha256_file(source_resolved) != source_hash:
        raise ValueError("source manifest changed before shard receipt publication")
    atomic_write_json(
        global_shard_receipt_path(manifest_path),
        OcrGlobalShardReceipt(
            source_manifest_sha256=source_hash,
            config_sha256=config_sha256,
            global_manifest_sha256=sha256_file(manifest_path),
            shard_count=len(entries),
            frame_count=len(frames),
        ).model_dump(mode="json"),
    )
    if sha256_file(source_resolved) != source_hash:
        raise ValueError("source manifest changed after shard publication")
    return manifest_path, global_manifest


def verify_global_shard_structure(
    *,
    source_manifest: Path,
    global_manifest: Path,
    expected_config_sha256: str,
    tracking_config: TrackingConfig,
) -> dict[str, int]:
    """Structural-only frame/video coverage check; not a production artifact gate."""

    _, source_hash, source_frames = _frame_manifest_snapshot(source_manifest)
    manifest = OcrGlobalShardManifest.model_validate_json(
        global_manifest.read_text(encoding="utf-8")
    )
    receipt = OcrGlobalShardReceipt.model_validate_json(
        global_shard_receipt_path(global_manifest).read_text(encoding="utf-8")
    )
    if (
        manifest.source_manifest_sha256 != source_hash
        or manifest.config_sha256 != expected_config_sha256
        or receipt.source_manifest_sha256 != source_hash
        or receipt.config_sha256 != expected_config_sha256
        or receipt.global_manifest_sha256 != sha256_file(global_manifest)
        or receipt.shard_count != len(manifest.shards)
        or receipt.frame_count != len(source_frames)
        or manifest.maximum_frames_per_shard != tracking_config.maximum_frames_per_shard
    ):
        raise ValueError("global shard manifest/receipt identity drift")

    expected_by_uid = {item.frame_uid: item for item in source_frames}
    seen_frames: set[str] = set()
    video_owner: dict[str, str] = {}
    root = global_manifest.parent.resolve()
    for shard in manifest.shards:
        shard_path = (root / shard.manifest_relpath).resolve()
        try:
            shard_path.relative_to(root)
        except ValueError as error:
            raise ValueError("shard manifest path escapes global manifest root") from error
        _, shard_hash, shard_frames = _frame_manifest_snapshot(shard_path)
        if shard_hash != shard.manifest_sha256:
            raise ValueError(f"shard manifest checksum drift: {shard.shard_id}")
        if len(shard_frames) > tracking_config.maximum_frames_per_shard:
            raise ValueError(f"shard exceeds frame limit: {shard.shard_id}")
        if [item.frame_uid for item in shard_frames] != shard.frame_uids:
            raise ValueError(f"shard ordered frame membership drift: {shard.shard_id}")
        if sorted({item.video_id for item in shard_frames}) != shard.video_ids:
            raise ValueError(f"shard video membership drift: {shard.shard_id}")
        for frame in shard_frames:
            if frame.frame_uid in seen_frames or expected_by_uid.get(frame.frame_uid) != frame:
                raise ValueError("global shards contain duplicate or foreign frame")
            seen_frames.add(frame.frame_uid)
            owner = video_owner.setdefault(frame.video_id, shard.shard_id)
            if owner != shard.shard_id:
                raise ValueError(f"video {frame.video_id!r} spans multiple shards")
    if seen_frames != set(expected_by_uid):
        raise ValueError("global shard union has missing source frames")

    if sha256_file(source_manifest) != source_hash:
        raise ValueError("source manifest changed during global verification")
    return {
        "shards": len(manifest.shards),
        "videos": len(video_owner),
        "frames": len(seen_frames),
        "trajectories": 0,
    }


def verify_global_shards(
    *,
    source_manifest: Path,
    global_manifest: Path,
    expected_config_sha256: str,
    tracking_config: TrackingConfig,
    expected_run_id: str,
    expected_identity: object,
    shard_bundles: Mapping[str, OcrShardArtifactBundle],
    data_root: Path,
    crop_config: CropConfig,
    _test_fault_injector: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Production gate replaying source bytes against materialized detections."""

    from aic2026.contracts import OcrDetectionFrameRecord, OcrTrajectoryRecord

    from .phase1 import (
        Phase1Identity,
        _load_records,
        _verify_completed_receipt,
        receipt_path_for,
        verify_detection_artifact,
        verify_linked_artifacts,
    )

    if not isinstance(expected_identity, Phase1Identity):
        raise TypeError("expected_identity must be Phase1Identity")
    if not isinstance(crop_config, CropConfig):
        raise TypeError("crop_config must be CropConfig")

    # Capture committed manifests/artifacts plus legacy manifest-bound sources.
    # Lazy sources are read once during semantic replay and checked against the
    # SHA-256 already materialized in their detection frame record.
    source_resolved = source_manifest.resolve()
    global_resolved = global_manifest.resolve()
    _, _, source_frames = _frame_manifest_snapshot(source_resolved)
    try:
        global_payload = global_resolved.read_bytes()
    except OSError as error:
        raise ValueError("global shard manifest is unavailable") from error
    manifest = OcrGlobalShardManifest.model_validate_json(global_payload)
    shards_by_id = {item.shard_id: item for item in manifest.shards}
    if len(shards_by_id) != len(manifest.shards):
        raise ValueError("duplicate shard ID in global manifest")
    if set(shard_bundles) != set(shards_by_id):
        raise ValueError("artifact bundle shard set differs from global manifest")

    baseline_paths: list[Path] = [
        source_resolved,
        global_resolved,
        global_shard_receipt_path(global_resolved).resolve(),
    ]
    root = global_resolved.parent
    for shard in manifest.shards:
        shard_path = (root / shard.manifest_relpath).resolve()
        try:
            shard_path.relative_to(root)
        except ValueError as error:
            raise ValueError("shard manifest path escapes global manifest root") from error
        baseline_paths.append(shard_path)
        bundle = shard_bundles[shard.shard_id]
        for artifact in (bundle.detections, bundle.trajectories, bundle.representatives):
            artifact_resolved = artifact.resolve()
            baseline_paths.extend(
                (artifact_resolved, receipt_path_for(artifact_resolved).resolve())
            )
    source_root = data_root.resolve()
    source_paths: list[Path] = []
    for frame in source_frames:
        source_image = (source_root / frame.frame_relpath).resolve()
        try:
            source_image.relative_to(source_root)
        except ValueError as error:
            raise ValueError(f"frame path escapes data root: {frame.frame_uid}") from error
        source_paths.append(source_image)
        if frame.source_image_sha256 is not None:
            baseline_paths.append(source_image)
    if len(source_paths) != len(set(source_paths)):
        raise ValueError("global verifier source frame paths collide")
    if len(baseline_paths) != len(set(baseline_paths)):
        raise ValueError("global verifier inputs contain colliding file paths")
    try:
        baseline = tuple((path, sha256_file(path)) for path in baseline_paths)
    except OSError as error:
        raise ValueError("global verifier baseline input is unavailable") from error
    baseline_hashes = dict(baseline)
    for frame, source_image in zip(source_frames, source_paths, strict=True):
        if (
            frame.source_image_sha256 is not None
            and baseline_hashes[source_image] != frame.source_image_sha256
        ):
            raise ValueError(f"source image checksum drift: {frame.frame_uid}")
    counts = verify_global_shard_structure(
        source_manifest=source_resolved,
        global_manifest=global_resolved,
        expected_config_sha256=expected_config_sha256,
        tracking_config=tracking_config,
    )
    if _test_fault_injector is not None:
        _test_fault_injector("after_structural_verification")
    global_trajectory_ids: set[str] = set()
    total_trajectories = 0
    for shard_id in sorted(shard_bundles, key=natural_key):
        shard = shards_by_id[shard_id]
        bundle = shard_bundles[shard_id]
        shard_manifest_path = (root / shard.manifest_relpath).resolve()
        detection_receipt = _verify_completed_receipt(
            bundle.detections,
            receipt_path_for(bundle.detections),
            stage="detect_crop",
        )
        if (
            detection_receipt.shard_id != shard_id
            or detection_receipt.shard_manifest_sha256 != shard.manifest_sha256
            or detection_receipt.input_artifact_sha256 != shard.manifest_sha256
            or detection_receipt.resource_limits_sha256 != tracking_config.resource_limits_sha256
        ):
            raise ValueError("detection receipt/global shard binding drift")
        verify_detection_artifact(
            output=bundle.detections,
            frame_manifest=shard_manifest_path,
            data_root=source_root,
            crop_config=crop_config,
            expected_run_id=expected_run_id,
            expected_config_sha256=expected_config_sha256,
            expected_identity=expected_identity,
            expected_shard_id=shard_id,
            tracking_config=tracking_config,
        )
        verify_linked_artifacts(
            detections=bundle.detections,
            trajectories=bundle.trajectories,
            representatives=bundle.representatives,
            expected_run_id=expected_run_id,
            expected_config_sha256=expected_config_sha256,
            expected_identity=expected_identity,
            tracking_config=tracking_config,
        )
        shard_frames = {
            item.frame_uid: item for item in _frame_manifest_snapshot(shard_manifest_path)[2]
        }
        detection_records = _load_records(bundle.detections, OcrDetectionFrameRecord)
        if {record.frame_uid for record in detection_records} != set(shard_frames):
            raise ValueError("detection bundle frame membership differs from shard")
        for record in detection_records:
            frame = shard_frames[record.frame_uid]
            expected_frame = frame.model_dump()
            actual_frame = {key: getattr(record, key) for key in expected_frame}
            if frame.source_image_sha256 is None:
                expected_frame["source_image_sha256"] = record.source_image_sha256
            if record.video_id not in shard.video_ids or actual_frame != expected_frame:
                raise ValueError("detection frame provenance is foreign to its shard")
        trajectories = _load_records(bundle.trajectories, OcrTrajectoryRecord)
        for trajectory in trajectories:
            for member in trajectory.members:
                frame = shard_frames.get(member.frame_uid)
                if (
                    frame is None
                    or member.video_id != frame.video_id
                    or member.frame_idx != frame.frame_idx
                    or member.frame_relpath != frame.frame_relpath
                    or (
                        frame.source_image_sha256 is not None
                        and member.source_image_sha256 != frame.source_image_sha256
                    )
                ):
                    raise ValueError("trajectory member is foreign or missing from its shard")
            if trajectory.trajectory_id in global_trajectory_ids:
                raise ValueError("trajectory ID is duplicated across shards")
            global_trajectory_ids.add(trajectory.trajectory_id)
        total_trajectories += len(trajectories)
    if _test_fault_injector is not None:
        _test_fault_injector("after_linked_verification")
    counts["trajectories"] = total_trajectories
    for path, expected_hash in baseline:
        if sha256_file(path) != expected_hash:
            raise ValueError(f"global verification input changed during replay: {path}")
    return counts
