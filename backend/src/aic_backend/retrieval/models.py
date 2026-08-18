"""Online domain models independent from FastAPI and Qdrant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Modality = Literal["scene", "object", "ocr", "asr", "dense"]


@dataclass(frozen=True)
class OcrLine:
    line_id: str
    raw_text: str
    normalized_text: str
    confidence: float | None
    accepted: bool
    polygon_xy: tuple[tuple[float, float], ...] | None
    polygon_clamped: bool = False
    reading_order: int = 0


@dataclass(frozen=True)
class StructuredOcr:
    terminal_status: Literal["success", "empty", "error"]
    full_text: str
    width: int
    height: int
    run_id: str
    model_revisions: tuple[str, ...]
    source_image_sha256: str | None
    lines: tuple[OcrLine, ...] = ()


@dataclass(frozen=True)
class Evidence:
    modality: Modality
    text: str | None = None
    source_id: str | None = None
    score: float = 0.0


@dataclass(frozen=True)
class FrameCandidate:
    video_id: str
    frame_idx: int
    pts_time_s: float
    keyframe_n: int | None = None
    score: float = 0.0
    modality: Modality = "scene"
    evidence: Evidence | None = None
    region_id: str | None = None
    object_slot: int | None = None
    ocr: StructuredOcr | None = None

    @property
    def frame_uid(self) -> str:
        return f"{self.video_id}:{self.frame_idx}"


@dataclass(frozen=True)
class SearchHit:
    video_id: str
    frame_idx: int
    pts_time_s: float
    keyframe_n: int | None
    score: float
    modality_scores: dict[str, float] = field(default_factory=dict)
    evidence: tuple[Evidence, ...] = ()
    ocr: StructuredOcr | None = None

    @property
    def frame_uid(self) -> str:
        return f"{self.video_id}:{self.frame_idx}"


@dataclass(frozen=True)
class EventFrame:
    event_index: int
    frame: SearchHit


@dataclass(frozen=True)
class TrakeSequence:
    video_id: str
    score: float
    events: tuple[EventFrame, ...]
