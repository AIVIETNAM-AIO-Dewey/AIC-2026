"""Resumable JSONL pipelines for SAM masks and DAM descriptions."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from itertools import islice
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from aic2026.common import iter_jsonl, write_jsonl_atomic
from aic2026.contracts import (
    CaptionResult,
    DetectorMetadata,
    FrameRef,
    MaskRLE,
    ObjectFrameRecord,
    ObjectRegion,
    SegmentationResult,
)
from aic2026.contracts.models import CaptionError

from .caption import normalize_caption
from .geometry import FilterConfig, filter_detections, normalized_to_pixels
from .organizer import index_object_files, load_organizer_detections
from .rle import decode_mask, encode_mask
from .sam_backend import MaskPrediction


class MaskBackend(Protocol):
    def generate(
        self, image: Image.Image, boxes_xyxy: list[tuple[int, int, int, int]]
    ) -> list[MaskPrediction]: ...


class CaptionBackend(Protocol):
    def describe(
        self, image: Image.Image, mask: Image.Image, *, max_new_tokens: int = 48
    ) -> str: ...


class _ResumableJsonl:
    def __init__(
        self, output: Path, resume: bool, *, successful_captions_only: bool = False
    ) -> None:
        self.output = output
        self.partial = output.with_suffix(output.suffix + ".partial")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if not resume:
                raise FileExistsError(f"Output already exists; use --resume: {output}")
            records = self._validate_records(list(iter_jsonl(output)), require_non_empty=True)
            if successful_captions_only and self._first_failed_caption_record(records) is not None:
                raise ValueError("Completed description artifact contains failed captions")
            self.completed = True
            self._set_metadata(records)
            self.stream = None
            return
        self.completed = False
        valid_records: list[dict[str, Any]] = []
        recovered_torn_write = False
        if resume and self.partial.exists():
            try:
                valid_records = list(iter_jsonl(self.partial))
            except (UnicodeDecodeError, ValueError) as partial_error:
                # Recover complete lines and discard only a torn final write.
                recovered_torn_write = True
                raw_lines = self.partial.read_bytes().splitlines(keepends=True)
                for line_number, raw_line in enumerate(raw_lines, start=1):
                    is_final_line = line_number == len(raw_lines)
                    try:
                        line = raw_line.decode("utf-8")
                    except UnicodeDecodeError as line_error:
                        if is_final_line:
                            break
                        raise ValueError("Partial JSONL contains invalid UTF-8") from line_error
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as line_error:
                        if not is_final_line and any(
                            rest.strip() for rest in raw_lines[line_number:]
                        ):
                            raise ValueError(
                                "Partial JSONL is corrupt before its final non-empty line"
                            ) from line_error
                        break
                    if not isinstance(value, dict):
                        raise ValueError("Partial JSONL records must be objects") from partial_error
                    valid_records.append(value)
            if self.partial.stat().st_size and not self.partial.read_bytes().endswith(b"\n"):
                recovered_torn_write = True
            validated_records = self._validate_records(valid_records, require_non_empty=False)
            failed_at = (
                self._first_failed_caption_record(validated_records)
                if successful_captions_only
                else None
            )
            if failed_at is not None:
                validated_records = validated_records[:failed_at]
                recovered_torn_write = True
            valid_records = [record.model_dump(mode="json") for record in validated_records]
            # Normalize a recovered torn write before reopening in append mode.
            if recovered_torn_write:
                write_jsonl_atomic(self.partial, valid_records)
        elif self.partial.exists():
            raise FileExistsError(f"Partial output exists; use --resume: {self.partial}")
        validated_records = self._validate_records(valid_records, require_non_empty=False)
        self._set_metadata(validated_records)
        self.stream = self.partial.open("a", encoding="utf-8", newline="\n")

    @staticmethod
    def _validate_records(
        records: list[dict[str, Any]], *, require_non_empty: bool
    ) -> list[ObjectFrameRecord]:
        validated = [ObjectFrameRecord.model_validate(record) for record in records]
        if require_non_empty and not validated:
            raise ValueError("Completed object artifact cannot be empty")
        frame_uids = [record.frame_uid for record in validated]
        if len(frame_uids) != len(set(frame_uids)):
            raise ValueError("Resumable object artifact contains duplicate frame_uid values")
        return validated

    @staticmethod
    def _first_failed_caption_record(records: list[ObjectFrameRecord]) -> int | None:
        for index, record in enumerate(records):
            if any(region.caption.status != "ok" for region in record.regions):
                return index
        return None

    def _set_metadata(self, records: list[ObjectFrameRecord]) -> None:
        self.record_metadata = [
            (
                record.frame_uid,
                record.run_id,
                frozenset(region.caption.status for region in record.regions),
            )
            for record in records
        ]
        self.seen = {item[0] for item in self.record_metadata}

    def validate_prefix(
        self,
        expected: list[tuple[str, str]],
        *,
        caption_mode: str,
        require_complete: bool = False,
    ) -> None:
        actual = [(frame_uid, run_id) for frame_uid, run_id, _ in self.record_metadata]
        expected_slice = expected if require_complete else expected[: len(actual)]
        if actual != expected_slice:
            qualifier = "complete sequence" if require_complete else "ordered prefix"
            raise ValueError(f"Resumable artifact is not an exact {qualifier} of its input")
        for _, _, statuses in self.record_metadata:
            if caption_mode == "pending" and statuses.difference({"pending"}):
                raise ValueError("Mask partial contains non-pending captions")
            if caption_mode == "ok" and statuses.difference({"ok"}):
                raise ValueError("Description partial contains failed captions")

    def append(self, record: ObjectFrameRecord) -> None:
        if self.stream is None:
            return
        if record.frame_uid in self.seen:
            raise ValueError(f"Duplicate frame_uid during resume: {record.frame_uid}")
        self.stream.write(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.seen.add(record.frame_uid)
        self.record_metadata.append(
            (
                record.frame_uid,
                record.run_id,
                frozenset(region.caption.status for region in record.regions),
            )
        )

    def finalize(self) -> None:
        if self.stream is None:
            return
        self.stream.close()
        self.partial.replace(self.output)
        self.completed = True


def _frame_records(path: Path) -> list[FrameRef]:
    return [FrameRef.model_validate(record) for record in iter_jsonl(path)]


def prepare_masks(
    *,
    frame_manifest: Path,
    objects_dir: Path,
    data_root: Path,
    output: Path,
    run_id: str,
    mask_backend: MaskBackend,
    filter_config: FilterConfig | None = None,
    resume: bool = False,
    limit: int | None = None,
    progress: Callable[[ObjectFrameRecord], None] | None = None,
) -> dict[str, int]:
    filter_config = filter_config or FilterConfig()
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    writer = _ResumableJsonl(output, resume)
    if writer.completed:
        return {"frames": len(writer.seen), "regions": 0, "resumed_complete": 1}
    frame_refs = _frame_records(frame_manifest)
    if limit is not None:
        frame_refs = frame_refs[:limit]
    if not frame_refs:
        raise ValueError("frame manifest is empty")
    expected_records = [(frame.frame_uid, run_id) for frame in frame_refs]
    writer.validate_prefix(expected_records, caption_mode="pending")
    object_files = index_object_files(objects_dir)
    counters = {
        "frames": 0,
        "regions": 0,
        "bbox_fallbacks": 0,
        "frames_without_regions": 0,
        "frames_at_region_cap": 0,
        "skipped": 0,
    }
    for frame in frame_refs:
        if frame.frame_uid in writer.seen:
            counters["skipped"] += 1
            continue
        object_path = object_files.get(frame.keyframe_n)
        if object_path is None:
            raise ValueError(f"Missing Objects JSON for keyframe n={frame.keyframe_n}")
        detections = filter_detections(load_organizer_detections(object_path), filter_config)
        counters["frames_without_regions"] += int(not detections)
        counters["frames_at_region_cap"] += int(len(detections) == filter_config.maximum_regions)
        image_path = (data_root / frame.frame_relpath).resolve()
        try:
            image_path.relative_to(data_root.resolve())
        except ValueError as error:
            raise ValueError(f"Frame path escapes data root: {image_path}") from error
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        if image.size != (frame.width, frame.height):
            raise ValueError(f"Frame dimensions changed since manifest: {image_path}")
        boxes = [
            normalized_to_pixels(detection.bbox_yxyx_norm, frame.width, frame.height)
            for detection in detections
        ]
        predictions = mask_backend.generate(image, boxes)
        if len(predictions) != len(detections):
            raise ValueError("SAM backend returned a different number of masks than boxes")
        regions: list[ObjectRegion] = []
        for detection, box, prediction in zip(detections, boxes, predictions, strict=True):
            rle = encode_mask(prediction.mask)
            region_id = f"{frame.video_id}:{frame.frame_idx}:d{detection.source_index:03d}"
            regions.append(
                ObjectRegion(
                    region_id=region_id,
                    source_detection_index=detection.source_index,
                    bbox_yxyx_norm=detection.bbox_yxyx_norm,
                    bbox_xyxy_px=box,
                    detector=DetectorMetadata(
                        class_name=detection.class_name,
                        class_entity=detection.class_entity,
                        class_label=detection.class_label,
                        score=detection.score,
                    ),
                    segmentation=SegmentationResult(
                        mask_source=prediction.source,
                        sam_iou_score=prediction.iou_score,
                        mask_rle=MaskRLE.model_validate(rle),
                    ),
                )
            )
            counters["bbox_fallbacks"] += int(prediction.source == "bbox_fallback")
        record = ObjectFrameRecord(**frame.model_dump(), run_id=run_id, regions=regions)
        writer.append(record)
        counters["frames"] += 1
        counters["regions"] += len(regions)
        if progress:
            progress(record)
    writer.validate_prefix(expected_records, caption_mode="pending", require_complete=True)
    writer.finalize()
    totals = _summarize_object_artifact(output, region_cap=filter_config.maximum_regions)
    return {
        "frames": totals["frames"],
        "regions": totals["regions"],
        "bbox_fallbacks": totals["bbox_fallbacks"],
        "frames_without_regions": totals["frames_without_regions"],
        "frames_at_region_cap": totals["frames_at_region_cap"],
        "skipped": counters["skipped"],
    }


def _is_cuda_oom(error: BaseException) -> bool:
    return error.__class__.__name__ == "OutOfMemoryError" and "torch" in error.__class__.__module__


def _summarize_object_artifact(path: Path, *, region_cap: int = 20) -> dict[str, int]:
    counters = {
        "frames": 0,
        "regions": 0,
        "bbox_fallbacks": 0,
        "frames_without_regions": 0,
        "frames_at_region_cap": 0,
        "captions_ok": 0,
        "captions_error": 0,
        "captions_oom": 0,
    }
    for raw in iter_jsonl(path):
        record = ObjectFrameRecord.model_validate(raw)
        counters["frames"] += 1
        counters["regions"] += len(record.regions)
        counters["frames_without_regions"] += int(not record.regions)
        counters["frames_at_region_cap"] += int(len(record.regions) == region_cap)
        for region in record.regions:
            counters["bbox_fallbacks"] += int(region.segmentation.mask_source == "bbox_fallback")
            if region.caption.status != "pending":
                counters[f"captions_{region.caption.status}"] += 1
    return counters


def _caption_error_tail(path: Path) -> tuple[int, int]:
    consecutive_errors = 0
    consecutive_oom = 0
    if not path.exists():
        return consecutive_errors, consecutive_oom
    for raw in iter_jsonl(path):
        record = ObjectFrameRecord.model_validate(raw)
        for region in record.regions:
            if region.caption.status == "error":
                consecutive_errors += 1
                consecutive_oom = 0
            elif region.caption.status == "oom":
                consecutive_oom += 1
                consecutive_errors = 0
            elif region.caption.status == "ok":
                consecutive_errors = 0
                consecutive_oom = 0
    return consecutive_errors, consecutive_oom


def validate_description_settings(
    *,
    maximum_words: int,
    max_new_tokens: int,
    oom_retry_max_new_tokens: int,
    maximum_consecutive_oom: int,
    maximum_consecutive_errors: int,
) -> None:
    if not 1 <= maximum_words <= 20:
        raise ValueError("maximum_words must be within [1, 20] for schema v1")
    if max_new_tokens < 1 or oom_retry_max_new_tokens < 1:
        raise ValueError("generation token limits must be positive")
    if oom_retry_max_new_tokens >= max_new_tokens:
        raise ValueError("OOM retry token limit must be below the primary token limit")
    if maximum_consecutive_oom < 1 or maximum_consecutive_errors < 1:
        raise ValueError("consecutive failure limits must be positive")


def run_descriptions(
    *,
    mask_artifact: Path,
    data_root: Path,
    output: Path,
    caption_backend: CaptionBackend,
    resume: bool = False,
    limit: int | None = None,
    max_new_tokens: int = 48,
    oom_retry_max_new_tokens: int = 32,
    maximum_words: int = 20,
    maximum_consecutive_oom: int = 3,
    maximum_consecutive_errors: int = 3,
) -> dict[str, int]:
    validate_description_settings(
        maximum_words=maximum_words,
        max_new_tokens=max_new_tokens,
        oom_retry_max_new_tokens=oom_retry_max_new_tokens,
        maximum_consecutive_oom=maximum_consecutive_oom,
        maximum_consecutive_errors=maximum_consecutive_errors,
    )
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    writer = _ResumableJsonl(output, resume, successful_captions_only=True)
    if writer.completed:
        return {"frames": len(writer.seen), "captions_ok": 0, "resumed_complete": 1}
    metadata_records = iter_jsonl(mask_artifact)
    if limit is not None:
        metadata_records = islice(metadata_records, limit)
    expected_records = []
    for raw in metadata_records:
        record = ObjectFrameRecord.model_validate(raw)
        expected_records.append((record.frame_uid, record.run_id))
    if not expected_records:
        raise ValueError("mask artifact is empty")
    writer.validate_prefix(expected_records, caption_mode="ok")
    counters = {
        "frames": 0,
        "captions_ok": 0,
        "captions_error": 0,
        "captions_oom": 0,
        "oom_retries": 0,
        "skipped": 0,
    }
    consecutive_errors, consecutive_oom = _caption_error_tail(writer.partial)
    input_records = iter_jsonl(mask_artifact)
    if limit is not None:
        input_records = islice(input_records, limit)
    for raw in input_records:
        record = ObjectFrameRecord.model_validate(raw)
        if record.frame_uid in writer.seen:
            counters["skipped"] += 1
            continue
        image_path = (data_root / record.frame_relpath).resolve()
        try:
            image_path.relative_to(data_root.resolve())
        except ValueError as error:
            raise ValueError(f"Frame path escapes data root: {image_path}") from error
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        described_regions: list[ObjectRegion] = []
        for region in record.regions:
            mask_array = decode_mask(region.segmentation.mask_rle.model_dump(mode="json"))
            mask = Image.fromarray(np.asarray(mask_array, dtype=np.uint8) * 255, mode="L")
            try:
                retry_after_oom = False
                try:
                    text = caption_backend.describe(image, mask, max_new_tokens=max_new_tokens)
                except BaseException as error:
                    if not _is_cuda_oom(error):
                        raise
                    retry_after_oom = True
                if retry_after_oom:
                    counters["oom_retries"] += 1
                    # Retry outside the exception handler so traceback-held model
                    # tensors can be released before clearing the CUDA allocator.
                    import gc

                    gc.collect()
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except (ImportError, RuntimeError):
                        pass
                    text = caption_backend.describe(
                        image, mask, max_new_tokens=oom_retry_max_new_tokens
                    )
                caption = normalize_caption(text, maximum_words=maximum_words)
                consecutive_errors = 0
                consecutive_oom = 0
                counters["captions_ok"] += 1
            except BaseException as error:
                if isinstance(error, KeyboardInterrupt | SystemExit):
                    raise
                if _is_cuda_oom(error):
                    consecutive_errors = 0
                    consecutive_oom += 1
                    counters["captions_oom"] += 1
                    caption = CaptionResult(
                        status="oom",
                        error=CaptionError(
                            type=error.__class__.__name__,
                            message=str(error) or "CUDA out of memory",
                            retryable=True,
                        ),
                    )
                else:
                    consecutive_errors += 1
                    consecutive_oom = 0
                    counters["captions_error"] += 1
                    caption = CaptionResult(
                        status="error",
                        error=CaptionError(
                            type=error.__class__.__name__,
                            message=str(error) or "DAM caption failed",
                            retryable=False,
                        ),
                    )
            described_regions.append(region.model_copy(update={"caption": caption}))
            if consecutive_oom >= maximum_consecutive_oom:
                raise RuntimeError(
                    f"Stopping shard after {consecutive_oom} consecutive DAM OOM failures"
                )
            if consecutive_errors >= maximum_consecutive_errors:
                raise RuntimeError(
                    f"Stopping shard after {consecutive_errors} consecutive DAM region failures"
                )
        described = record.model_copy(update={"regions": described_regions})
        writer.append(described)
        counters["frames"] += 1
    writer.validate_prefix(expected_records, caption_mode="any", require_complete=True)
    totals = _summarize_object_artifact(writer.partial)
    failed_captions = totals["captions_error"] + totals["captions_oom"]
    if failed_captions > 0:
        raise RuntimeError("DAM produced failed captions; refusing to publish shard")
    writer.finalize()
    return {
        "frames": totals["frames"],
        "regions": totals["regions"],
        "captions_ok": totals["captions_ok"],
        "captions_error": totals["captions_error"],
        "captions_oom": totals["captions_oom"],
        "oom_retries": counters["oom_retries"],
        "skipped": counters["skipped"],
    }
