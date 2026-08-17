"""Versioned EasyOCR output records consumed by online indexing."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .models import FrameRef, StrictModel


class OcrText(StrictModel):
    raw_text: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    polygon_xy: list[tuple[float, float]] = Field(min_length=4)

    @field_validator("polygon_xy", mode="before")
    @classmethod
    def accept_json_points(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [tuple(point) if isinstance(point, list) else point for point in value]
        return value


class OcrFrameRecord(FrameRef):
    schema_version: Literal["aic26.ocr.v1"] = "aic26.ocr.v1"
    run_id: str = Field(min_length=1)
    texts: list[OcrText] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_keyframe(self) -> OcrFrameRecord:
        if self.keyframe_n is None:
            raise ValueError("OCR records require an organizer keyframe_n")
        return self
