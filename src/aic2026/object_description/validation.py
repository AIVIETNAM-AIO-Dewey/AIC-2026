"""Cheap input validation that must run before loading SAM or DAM weights."""

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Literal

from PIL import Image

from aic2026.common import iter_jsonl
from aic2026.contracts import FrameRef, ObjectFrameRecord

from .organizer import index_object_files, load_organizer_detections
from .rle import decode_mask


def _safe_image_path(data_root: Path, relative: str) -> Path:
    root = data_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Frame path escapes data root: {relative}") from error
    return path


def validate_mask_stage_inputs(
    *,
    frame_manifest: Path,
    data_root: Path,
    video_id: str,
    limit: int | None,
    objects_dir: Path | None = None,
) -> dict[str, int]:
    refs = [FrameRef.model_validate(raw) for raw in iter_jsonl(frame_manifest)]
    if limit is not None:
        refs = refs[:limit]
    if not refs:
        raise ValueError("Frame manifest is empty")
    uids = [ref.frame_uid for ref in refs]
    if len(uids) != len(set(uids)):
        raise ValueError("Frame manifest contains duplicate frame_uid values")
    object_files = index_object_files(objects_dir) if objects_dir is not None else None
    if object_files is not None and limit is None:
        surplus = sorted(set(object_files).difference(ref.keyframe_n for ref in refs))
        if surplus:
            raise ValueError(
                f"Objects directory contains n values absent from manifest: {surplus[:5]}"
            )
    detection_count = 0
    for ref in refs:
        if ref.video_id != video_id:
            raise ValueError(
                f"Frame manifest video_id {ref.video_id!r} does not match {video_id!r}"
            )
        if object_files is not None:
            object_path = object_files.get(ref.keyframe_n)
            if object_path is None:
                raise ValueError(f"Missing Objects JSON for keyframe n={ref.keyframe_n}")
            detection_count += len(load_organizer_detections(object_path))
        image_path = _safe_image_path(data_root, ref.frame_relpath)
        with Image.open(image_path) as image:
            if image.size != (ref.width, ref.height):
                raise ValueError(f"Frame dimensions do not match manifest: {image_path}")
            image.verify()
    return {"frames": len(refs), "source_detections": detection_count}


def validate_description_stage_inputs(
    *,
    mask_artifact: Path,
    data_root: Path,
    video_id: str,
    limit: int | None,
    expected_run_id: str | None = None,
) -> dict[str, int]:
    raw_records = iter_jsonl(mask_artifact)
    if limit is not None:
        raw_records = islice(raw_records, limit)
    uids: set[str] = set()
    frame_count = 0
    region_count = 0
    for raw in raw_records:
        record = ObjectFrameRecord.model_validate(raw)
        if record.frame_uid in uids:
            raise ValueError("Mask artifact contains duplicate frame_uid values")
        uids.add(record.frame_uid)
        frame_count += 1
        if record.video_id != video_id:
            raise ValueError(
                f"Mask artifact video_id {record.video_id!r} does not match {video_id!r}"
            )
        if expected_run_id is not None and record.run_id != expected_run_id:
            raise ValueError(
                f"Mask artifact run_id {record.run_id!r} does not match {expected_run_id!r}"
            )
        image_path = _safe_image_path(data_root, record.frame_relpath)
        with Image.open(image_path) as image:
            if image.size != (record.width, record.height):
                raise ValueError(f"Frame dimensions do not match artifact: {image_path}")
            image.verify()
        for region in record.regions:
            if region.caption.status != "pending":
                raise ValueError(f"Mask input already contains caption: {region.region_id}")
            mask = decode_mask(region.segmentation.mask_rle.model_dump(mode="json"))
            if not mask.any():
                raise ValueError(f"Decoded mask is empty: {region.region_id}")
            region_count += 1
    if frame_count == 0:
        raise ValueError("Mask artifact is empty")
    return {"frames": frame_count, "regions": region_count}


def validate_published_object_artifact(
    *,
    artifact: Path,
    data_root: Path,
    video_id: str,
    expected_frame_uids: list[str],
    caption_mode: Literal["pending", "finished"],
    expected_run_id: str | None = None,
) -> dict[str, int]:
    """Validate a final mask/description shard before repairing its sidecar."""
    counters = {
        "frames": 0,
        "regions": 0,
        "bbox_fallbacks": 0,
        "captions_ok": 0,
        "captions_error": 0,
        "captions_oom": 0,
    }
    for raw in iter_jsonl(artifact):
        record = ObjectFrameRecord.model_validate(raw)
        index = counters["frames"]
        if index >= len(expected_frame_uids) or record.frame_uid != expected_frame_uids[index]:
            raise ValueError("Published artifact frame order/completeness differs from its input")
        counters["frames"] += 1
        if record.video_id != video_id:
            raise ValueError(f"Artifact video_id {record.video_id!r} does not match {video_id!r}")
        if expected_run_id is not None and record.run_id != expected_run_id:
            raise ValueError(
                f"Artifact run_id {record.run_id!r} does not match {expected_run_id!r}"
            )
        image_path = _safe_image_path(data_root, record.frame_relpath)
        with Image.open(image_path) as image:
            if image.size != (record.width, record.height):
                raise ValueError(f"Frame dimensions do not match artifact: {image_path}")
            image.verify()
        for region in record.regions:
            mask = decode_mask(region.segmentation.mask_rle.model_dump(mode="json"))
            if not mask.any():
                raise ValueError(f"Decoded mask is empty: {region.region_id}")
            status = region.caption.status
            if caption_mode == "pending" and status != "pending":
                raise ValueError(f"Mask artifact contains caption: {region.region_id}")
            if caption_mode == "finished" and status == "pending":
                raise ValueError(f"Description artifact has pending caption: {region.region_id}")
            counters["regions"] += 1
            counters["bbox_fallbacks"] += int(region.segmentation.mask_source == "bbox_fallback")
            if status != "pending":
                counters[f"captions_{status}"] += 1
    if counters["frames"] != len(expected_frame_uids):
        raise ValueError("Published artifact frame order/completeness differs from its input")
    return counters
