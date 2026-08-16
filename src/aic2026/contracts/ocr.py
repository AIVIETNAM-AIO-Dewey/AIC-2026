"""Versioned EasyOCR output records consumed by online indexing."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .models import FrameRef, StrictModel


class OcrText(StrictModel):
    raw_text: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    polygon_xy: list[tuple[float, float]] = Field(min_length=4)


class OcrFrameRecord(FrameRef):
    schema_version: Literal["aic26.ocr.v1"] = "aic26.ocr.v1"
    run_id: str = Field(min_length=1)
    texts: list[OcrText] = Field(default_factory=list)
