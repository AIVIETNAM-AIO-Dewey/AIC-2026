"""Canonical Pydantic contracts for the Unified Multi-Modal Frame Pipeline."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
        allow_inf_nan=False,
    )


class DamRegionCaption(StrictModel):
    region_id: str = Field(min_length=1)
    class_label: str = Field(min_length=1)
    bbox_xyxy_px: tuple[int, int, int, int]
    sam_iou: float | None = None
    caption_en: str = Field(min_length=1)
    word_count: int = Field(ge=1, le=50)

    @model_validator(mode="after")
    def validate_word_count(self) -> DamRegionCaption:
        actual = len(self.caption_en.split())
        if actual != self.word_count:
            raise ValueError(f"word_count {self.word_count} must match actual length {actual}")
        if actual > 50:
            raise ValueError("caption_en must contain at most 50 words")
        return self


class UnifiedOcrSpan(StrictModel):
    line_id: str
    raw_text: str
    normalized_text: str
    confidence: float
    polygon_xy: list[tuple[float, float]]
    normalized_polygon_xy: list[tuple[float, float]]


class UnifiedOcrResult(StrictModel):
    full_text: str
    spans: list[UnifiedOcrSpan] = Field(default_factory=list)


class UnifiedFrameRecord(StrictModel):
    schema_version: Literal["aic26.unified_frame.v1"] = "aic26.unified_frame.v1"
    video_id: str = Field(min_length=1)
    frame_uid: str = Field(min_length=3)
    keyframe_n: PositiveInt
    frame_idx: int = Field(ge=0)
    pts_time_s: float = Field(ge=0)
    fps: PositiveFloat
    shot_id: str | None = None
    image_relpath: str = Field(min_length=1)
    siglip_embedding: list[float] | None = None  # 768 float unit vector
    ocr: UnifiedOcrResult = Field(default_factory=lambda: UnifiedOcrResult(full_text=""))
    dam_descriptions: list[DamRegionCaption] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_frame_identity(self) -> UnifiedFrameRecord:
        expected_uid = f"{self.video_id}:{self.frame_idx}"
        if self.frame_uid != expected_uid:
            raise ValueError(f"frame_uid must equal {expected_uid!r}")
        if self.siglip_embedding is not None and len(self.siglip_embedding) != 768:
            raise ValueError(f"SigLIP embedding must have exactly 768 dimensions; got {len(self.siglip_embedding)}")
        return self
