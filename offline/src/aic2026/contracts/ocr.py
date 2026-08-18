"""Versioned OCR records consumed by online indexing.

The v2 contract preserves every native-frame polygon and emits exactly one
terminal record (success, empty, or error) for every input keyframe.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .models import FrameRef, StrictModel


class OcrError(StrictModel):
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=500)


class OcrText(StrictModel):
    line_id: str = Field(min_length=1)
    raw_text: str
    normalized_text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_semantics: Literal["engine_native_score", "not_provided"]
    accepted: bool
    polygon_raw_xy: list[tuple[float, float]] | None = None
    polygon_xy: list[tuple[float, float]] | None = None
    normalized_polygon_xy: list[tuple[float, float]] | None = None
    polygon_clamped: bool = False
    geometry_warning: str | None = None
    source_order: int = Field(ge=0)
    reading_order: int = Field(ge=0)

    @field_validator("polygon_raw_xy", "polygon_xy", "normalized_polygon_xy", mode="before")
    @classmethod
    def accept_json_points(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [tuple(point) if isinstance(point, list) else point for point in value]
        return value

    @model_validator(mode="after")
    def validate_line(self) -> OcrText:
        if self.accepted and not self.normalized_text:
            raise ValueError("accepted OCR text must be non-empty")
        if self.confidence is None and self.confidence_semantics != "not_provided":
            raise ValueError("null confidence must use not_provided semantics")
        if self.confidence is not None and self.confidence_semantics != "engine_native_score":
            raise ValueError("numeric confidence must use engine_native_score semantics")
        if self.polygon_xy is None:
            if self.normalized_polygon_xy is not None or not self.geometry_warning:
                raise ValueError("missing polygon requires a geometry warning")
        else:
            if len(self.polygon_xy) < 3 or self.normalized_polygon_xy is None:
                raise ValueError("native polygon requires matching normalized geometry")
            if self.polygon_raw_xy is None:
                raise ValueError("native polygon requires the engine polygon provenance")
            if len(self.polygon_xy) != len(self.polygon_raw_xy):
                raise ValueError("raw and native polygons must have equal length")
            if len(self.polygon_xy) != len(self.normalized_polygon_xy):
                raise ValueError("native and normalized polygons must have equal length")
            if self.geometry_warning is not None:
                raise ValueError("valid native polygon cannot carry a geometry warning")
        return self


class LegacyOcrText(StrictModel):
    """Read-only compatibility contract for previously published EasyOCR artifacts."""

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


class LegacyOcrFrameRecord(FrameRef):
    schema_version: Literal["aic26.ocr.v1"] = "aic26.ocr.v1"
    run_id: str = Field(min_length=1)
    texts: list[LegacyOcrText] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_keyframe(self) -> LegacyOcrFrameRecord:
        if self.keyframe_n is None:
            raise ValueError("OCR records require an organizer keyframe_n")
        return self


class OcrFrameRecord(FrameRef):
    schema_version: Literal["aic26.ocr.v2"] = "aic26.ocr.v2"
    run_id: str = Field(min_length=1)
    terminal_status: Literal["success", "empty", "error"]
    full_text: str = ""
    texts: list[OcrText] = Field(default_factory=list)
    error: OcrError | None = None

    @model_validator(mode="after")
    def validate_terminal_record(self) -> OcrFrameRecord:
        if self.keyframe_n is None:
            raise ValueError("OCR records require an organizer keyframe_n")
        if self.width < 2 or self.height < 2:
            raise ValueError("OCR geometry requires frame dimensions of at least 2x2")
        if self.source_image_sha256 is None:
            raise ValueError("OCR records require the source image SHA-256")
        ids = [line.line_id for line in self.texts]
        source_orders = [line.source_order for line in self.texts]
        reading_orders = [line.reading_order for line in self.texts]
        if len(ids) != len(set(ids)):
            raise ValueError("line_id values must be unique within a frame")
        if len(source_orders) != len(set(source_orders)):
            raise ValueError("source_order values must be unique within a frame")
        if len(reading_orders) != len(set(reading_orders)):
            raise ValueError("reading_order values must be unique within a frame")

        for line in self.texts:
            if line.polygon_xy is None:
                continue
            assert line.normalized_polygon_xy is not None
            assert line.polygon_raw_xy is not None
            expected_native = [
                (
                    min(max(x, 0.0), self.width - 1.0),
                    min(max(y, 0.0), self.height - 1.0),
                )
                for x, y in line.polygon_raw_xy
            ]
            if line.polygon_xy != expected_native:
                raise ValueError("native polygon must be the canonical frame-bounds clamp")
            if line.polygon_clamped is not (line.polygon_raw_xy != expected_native):
                raise ValueError("polygon_clamped is inconsistent with native geometry")
            for (x, y), (nx, ny) in zip(line.polygon_xy, line.normalized_polygon_xy, strict=True):
                if not (0 <= x <= self.width - 1 and 0 <= y <= self.height - 1):
                    raise ValueError("native polygon exceeds frame bounds")
                if not (
                    math.isclose(nx, x / (self.width - 1), abs_tol=1e-9)
                    and math.isclose(ny, y / (self.height - 1), abs_tol=1e-9)
                ):
                    raise ValueError("normalized polygon must derive from native coordinates")

        accepted = sorted(
            (line for line in self.texts if line.accepted), key=lambda line: line.reading_order
        )
        expected_text = " ".join(line.normalized_text for line in accepted)
        if self.terminal_status == "success":
            if not accepted or self.full_text != expected_text or self.error is not None:
                raise ValueError("success OCR terminal payload is inconsistent")
        elif self.terminal_status == "empty":
            if self.texts or self.full_text or self.error is not None:
                raise ValueError("empty OCR terminal payload is inconsistent")
        elif self.texts or self.full_text or self.error is None:
            raise ValueError("error OCR terminal payload is inconsistent")
        return self
