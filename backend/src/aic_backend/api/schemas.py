"""Stable OpenAPI DTOs. frame_idx is always the organizer canonical ID."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    task_type: Literal["kis", "qa", "trake"]
    raw_query_vi: str = Field(min_length=1)
    top_k: int = Field(default=100, ge=1, le=100)
    use_images_for_answer: bool = True


class EvidenceResponse(BaseModel):
    modality: str
    text: str | None = None
    source_id: str | None = None
    score: float


class OcrLineResponse(BaseModel):
    line_id: str
    raw_text: str
    normalized_text: str
    confidence: float | None
    accepted: bool
    polygon_xy: list[tuple[float, float]] | None
    polygon_clamped: bool
    reading_order: int


class StructuredOcrResponse(BaseModel):
    terminal_status: Literal["success", "empty", "error"]
    full_text: str
    width: int
    height: int
    run_id: str
    model_revisions: list[str]
    source_image_sha256: str | None
    lines: list[OcrLineResponse]


class OcrMatchResponse(BaseModel):
    query: str
    normalized_query: str
    matched_text: str
    lexical_score: float
    fuzzy_similarity: float | None
    final_score: float
    match_type: Literal["exact", "accent_folded", "fuzzy", "trigram_candidate"]
    fuzzy_enabled: bool


class FrameHitResponse(BaseModel):
    rank: int
    score: float
    video_id: str
    frame_idx: int
    keyframe_n: int | None = None
    pts_time_s: float
    image_url: str
    modality_scores: dict[str, float]
    evidence: list[EvidenceResponse]
    ocr: StructuredOcrResponse | None = None
    ocr_match: OcrMatchResponse | None = None


class TrakeEventResponse(BaseModel):
    event_index: int
    frame: FrameHitResponse


class TrakeSequenceResponse(BaseModel):
    rank: int
    video_id: str
    score: float
    events: list[TrakeEventResponse]


class SearchResponse(BaseModel):
    request_id: str
    task_type: str
    latency_ms: float
    stage_latency_ms: dict[str, float]
    degraded: bool = False
    results: list[FrameHitResponse] = Field(default_factory=list)
    sequences: list[TrakeSequenceResponse] = Field(default_factory=list)
    answer: str | None = None
    confidence: float | None = None
    evidence_frame_uids: list[str] = Field(default_factory=list)


class OcrSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=25, ge=1, le=100)
    fuzzy: bool = True


class OcrSearchResponse(BaseModel):
    request_id: str
    task_type: Literal["ocr"] = "ocr"
    query: str
    normalized_query: str
    fuzzy_enabled: bool
    strategies: list[str]
    latency_ms: float
    results: list[FrameHitResponse] = Field(default_factory=list)


class OcrJobRunRequest(BaseModel):
    manifest_id: str = Field(min_length=1, max_length=100)


class OcrDatasetStatusResponse(BaseModel):
    manifest_id: str
    status: Literal["not_started", "running", "interrupted", "failed", "completed"]
    total_frames: int
    processed_frames: int
    remaining_frames: int
    counters: dict[str, int]
    output_exists: bool


class OcrJobsResponse(BaseModel):
    enabled: bool
    model_id: str
    active_manifest_id: str | None
    started_at: str | None
    last_exit_code: int | None
    datasets: list[OcrDatasetStatusResponse]


class SubmissionRenderRequest(BaseModel):
    task_type: Literal["kis", "qa", "trake"]
    frames: list[FrameHitResponse] = Field(default_factory=list)
    answer: str | None = None
    sequences: list[TrakeSequenceResponse] = Field(default_factory=list)


class CapabilitiesResponse(BaseModel):
    qdrant_ready: bool
    openai_configured: bool
    image_answers_enabled: bool
    search_ready: bool
    tasks: dict[str, dict[str, object]]
    collections: dict[str, bool]
    models: dict[str, bool]
