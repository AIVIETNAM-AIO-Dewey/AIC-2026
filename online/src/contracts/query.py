"""Pydantic Contracts & Schemas for Online Retrieval Sub-Queries and Results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TaskType = Literal["KIS", "TRAKE", "VQA"]


class TrakeEvent(BaseModel):
    """Sub-event description in a chronological action sequence (TRAKE)."""

    order: int = Field(description="Sequence index (1, 2, 3, ...)")
    description: str = Field(description="Original description of this event")
    scene_en: str = Field(
        default="", description="Visual setting/camera framing in English for SigLIP"
    )
    objects_en: list[str] = Field(
        default_factory=list,
        description="Key objects, tools, actions, ingredients in English for DAM",
    )
    speech_vi: str = Field(default="", description="Spoken speech keywords in Vietnamese for ASR")
    ocr_keywords: list[str] = Field(
        default_factory=list, description="On-screen text in Vietnamese for OCR"
    )


class ParsedQuery(BaseModel):
    """Structured Sub-Queries output from LLM Query Decomposer."""

    model_config = ConfigDict(extra="forbid")

    task_type: TaskType = Field(default="KIS", description="Identified task: KIS, TRAKE, or VQA")
    language: str = Field(default="vi", description="Input query language (vi, en, mixed)")
    original_query: str = Field(
        default="",
        description="Raw user query text; optional for manually authored direct JSON",
    )
    session_id: str = Field(default="", description="Session ID for caching branch search hits")

    # 4-Channel Sub-Queries with exact language mapping:
    global_scene_en: str = Field(
        default="",
        description="Global background, lighting, camera angle, setting in English for SigLIP",
    )
    objects_en: list[str] = Field(
        default_factory=list,
        description="List of specific visual objects, people, clothing, ingredients in English for DAM",
    )
    speech_vi: str = Field(
        default="",
        description="Spoken dialogue keywords, voiceover topic in Vietnamese for Audio ASR",
    )
    ocr_keywords: list[str] = Field(
        default_factory=list,
        description="Exact on-screen text, subtitles, numbers, brand names, recipes in Vietnamese for OCR",
    )

    # TRAKE Sequential Sub-Events
    is_temporal_trake: bool = Field(
        default=False, description="True if query describes chronological events"
    )
    trake_events: list[TrakeEvent] = Field(
        default_factory=list, description="Sequential sub-event list for TRAKE"
    )

    # VQA Question String
    vqa_question: str = Field(
        default="", description="The specific question being asked if task is VQA"
    )

    # Dynamic Channel Weights (Sum = 1.0)
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "vis": 0.45,
            "dam": 0.40,
            "asr": 0.15,
            "ocr": 0.00,
        },
        description="Dynamic modality weights",
    )


class MatchedObject(BaseModel):
    """Explainability metadata for detected DAM objects."""

    region_id: str | int = 1
    class_entity: str = "Object"
    description_en: str = ""
    score: float = 0.0
    bbox: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])


class SpeechEvidence(BaseModel):
    """Spoken dialogue evidence for keyframe."""

    start_s: float = 0.0
    end_s: float = 0.0
    transcript_raw: str = ""
    score: float = 0.0


class SearchResult(BaseModel):
    """Single candidate keyframe result card."""

    rank: int = 1
    video_id: str
    keyframe_n: int = 1
    frame_idx: int
    pts_time_s: float = 0.0
    submission_string: str = ""  # Format: "<VIDEO_ID>, <FRAME_IDX>"

    # Fusion & Multi-Stage Scores
    final_score: float = 0.0
    stage1_score: float = 0.0
    stage2_rerank_score: float = 0.0
    visual_similarity: float = 0.0

    # Image Display
    image_relpath: str = ""
    image_available: bool = True
    best_matching_objects: list[MatchedObject] = Field(default_factory=list)
    dam_full_captions: list[str] = Field(default_factory=list)

    # Speech & Text Evidence
    has_speech: bool = False
    speech_evidence: SpeechEvidence | None = None
    ocr_text: str = ""

    # Navigation & Reasoning
    adjacent_keyframes: list[int] = Field(default_factory=list)
    trake_matched_frames: list[int] = Field(default_factory=list)
    vqa_answer: str | None = None


class SearchResponse(BaseModel):
    """Full search response payload returned to Frontend UI."""

    task_type: TaskType = "KIS"
    original_query: str = ""
    parsed_query: ParsedQuery | None = None
    execution_time_ms: float = 0.0
    total_candidates_evaluated: int = 0
    results: list[SearchResult] = Field(default_factory=list)
    vqa_answer: str | None = None
