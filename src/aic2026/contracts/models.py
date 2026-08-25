"""Pydantic models for frame, object-region, and run artifacts."""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
        allow_inf_nan=False,
    )


class FrameRef(StrictModel):
    video_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    frame_uid: str = Field(min_length=3)
    keyframe_n: PositiveInt
    frame_idx: int = Field(ge=0)
    pts_time_s: float = Field(ge=0)
    fps: PositiveFloat
    frame_relpath: str = Field(min_length=1)
    width: PositiveInt
    height: PositiveInt

    @model_validator(mode="after")
    def validate_uid(self) -> FrameRef:
        expected = f"{self.video_id}:{self.frame_idx}"
        if self.frame_uid != expected:
            raise ValueError(f"frame_uid must equal {expected!r}")
        normalized_path = self.frame_relpath.replace("\\", "/")
        if (
            normalized_path.startswith("/")
            or re.match(r"^[A-Za-z]:/", normalized_path)
            or ".." in PurePosixPath(normalized_path).parts
        ):
            raise ValueError(
                "frame_relpath must be relative, never a machine-specific absolute path"
            )
        return self


class ShotRecord(StrictModel):
    schema_version: Literal["aic26.shot.v1"] = "aic26.shot.v1"
    video_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    shot_id: str = Field(min_length=1)
    shot_start_idx: int = Field(ge=0)
    shot_end_idx: int = Field(ge=0)
    start_time_s: float = Field(ge=0)
    end_time_s: float = Field(ge=0)
    fps: PositiveFloat
    source_video: str = Field(min_length=1)
    source: Literal["transnetv2"] = "transnetv2"

    @model_validator(mode="after")
    def validate_shot_bounds(self) -> ShotRecord:
        if self.shot_end_idx < self.shot_start_idx:
            raise ValueError("shot_end_idx must be >= shot_start_idx")
        if self.end_time_s < self.start_time_s:
            raise ValueError("end_time_s must be >= start_time_s")
        return self


class DetectorMetadata(StrictModel):
    source: Literal["organizer_frcnn"] = "organizer_frcnn"
    class_name: str = Field(min_length=1)
    class_entity: str = Field(min_length=1)
    class_label: int = Field(ge=0)
    score: float = Field(ge=0, le=1)


class MaskRLE(StrictModel):
    size: tuple[PositiveInt, PositiveInt]
    counts: str = Field(min_length=1)

    @field_validator("size", mode="before")
    @classmethod
    def accept_json_array(cls, value: Any) -> Any:
        # JSON has no tuple type; arrays are the canonical wire representation.
        return tuple(value) if isinstance(value, list) else value


class SegmentationResult(StrictModel):
    mask_source: Literal["sam", "bbox_fallback"]
    # SAM predicts mask quality with an unconstrained linear head. Preserve the
    # finite raw score instead of pretending it is mathematically bounded to [0, 1].
    sam_iou_score: float | None = None
    mask_rle: MaskRLE

    @model_validator(mode="after")
    def validate_score_source(self) -> SegmentationResult:
        if self.mask_source == "sam" and self.sam_iou_score is None:
            raise ValueError("SAM masks require the raw predicted quality score")
        if self.mask_source == "bbox_fallback" and self.sam_iou_score is not None:
            raise ValueError("bbox fallback masks cannot claim a SAM quality score")
        return self


class CaptionError(StrictModel):
    type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False


class CaptionResult(StrictModel):
    status: Literal["pending", "ok", "error", "oom"] = "pending"
    description_en: str | None = None
    word_count: int = Field(default=0, ge=0, le=50)
    truncated: bool = False
    error: CaptionError | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> CaptionResult:
        if self.status == "ok":
            if not self.description_en or not self.description_en.strip():
                raise ValueError("caption status 'ok' requires a non-empty description_en")
            actual = len(self.description_en.split())
            if actual != self.word_count:
                raise ValueError("word_count must match description_en")
            if actual > 50:
                raise ValueError("description_en must contain at most 50 words")
            if self.error is not None:
                raise ValueError("successful caption cannot contain error details")
        else:
            if self.description_en is not None or self.word_count != 0 or self.truncated:
                raise ValueError("non-successful caption cannot contain description text")
            if self.status == "pending" and self.error is not None:
                raise ValueError("pending caption cannot contain error details")
            if self.status in {"error", "oom"} and self.error is None:
                raise ValueError(
                    f"caption status {self.status!r} requires structured error details"
                )
        return self


class ObjectRegion(StrictModel):
    region_id: str = Field(min_length=1)
    source_detection_index: int = Field(ge=0)
    bbox_yxyx_norm: tuple[float, float, float, float]
    bbox_xyxy_px: tuple[int, int, int, int]
    coordinate_convention: Literal["xyxy_half_open"] = "xyxy_half_open"
    detector: DetectorMetadata
    segmentation: SegmentationResult
    caption: CaptionResult = Field(default_factory=CaptionResult)

    @field_validator("bbox_yxyx_norm", "bbox_xyxy_px", mode="before")
    @classmethod
    def accept_json_box_arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_boxes(self) -> ObjectRegion:
        ymin, xmin, ymax, xmax = self.bbox_yxyx_norm
        if not (0 <= ymin < ymax <= 1 and 0 <= xmin < xmax <= 1):
            raise ValueError("bbox_yxyx_norm must be ordered inside [0, 1]")
        x1, y1, x2, y2 = self.bbox_xyxy_px
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy_px must be a non-empty half-open box")
        return self


class ObjectFrameRecord(FrameRef):
    schema_version: Literal["aic26.object_regions.v1"] = "aic26.object_regions.v1"
    run_id: str = Field(min_length=1)
    regions: list[ObjectRegion]

    @model_validator(mode="after")
    def validate_region_identity(self) -> ObjectFrameRecord:
        ids = [region.region_id for region in self.regions]
        if len(ids) != len(set(ids)):
            raise ValueError("region_id values must be unique within a frame")
        for region in self.regions:
            height, width = region.segmentation.mask_rle.size
            if (width, height) != (self.width, self.height):
                raise ValueError("mask size must equal frame dimensions")
            x1, y1, x2, y2 = region.bbox_xyxy_px
            if x2 > self.width or y2 > self.height:
                raise ValueError("pixel bbox exceeds frame dimensions")
            ymin, xmin, ymax, xmax = region.bbox_yxyx_norm
            expected_x1 = max(0, min(self.width - 1, math.floor(xmin * self.width)))
            expected_y1 = max(0, min(self.height - 1, math.floor(ymin * self.height)))
            expected_box = (
                expected_x1,
                expected_y1,
                max(expected_x1 + 1, min(self.width, math.ceil(xmax * self.width))),
                max(expected_y1 + 1, min(self.height, math.ceil(ymax * self.height))),
            )
            if region.bbox_xyxy_px != expected_box:
                raise ValueError("pixel bbox is inconsistent with normalized bbox and frame size")
        return self


class ArtifactInput(StrictModel):
    name: str
    source_id: str
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class ModelRevision(StrictModel):
    model_id: str
    revision: str
    license: str


class RunManifest(StrictModel):
    schema_version: Literal["aic26.run_manifest.v1"] = "aic26.run_manifest.v1"
    run_id: str = Field(min_length=1)
    stage: Literal["frame_manifest", "sam_masks", "dam_descriptions", "scene_embeddings"]
    status: Literal["running", "completed", "failed"]
    git_sha: str | None = None
    git_dirty: bool | None = None
    platform: dict[str, Any]
    resolved_config: dict[str, Any]
    config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    seed: int
    inputs: list[ArtifactInput] = Field(default_factory=list)
    outputs: list[ArtifactInput] = Field(default_factory=list)
    models: list[ModelRevision] = Field(default_factory=list)
    started_at: datetime
    ended_at: datetime | None = None
    counters: dict[str, int] = Field(default_factory=dict)
    completed_shards: list[str] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_completion(self) -> RunManifest:
        if self.status in {"completed", "failed"} and self.ended_at is None:
            raise ValueError("terminal manifests require ended_at")
        if self.status == "running" and self.ended_at is not None:
            raise ValueError("running manifests cannot have ended_at")
        return self
