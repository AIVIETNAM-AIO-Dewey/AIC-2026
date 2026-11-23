"""Pydantic HTTP request contracts for the online retrieval workbench."""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from online.src.contracts.query import ParsedQuery, TaskType
from online.src.retrieval.branches.branch1.contracts import (
    BRANCH1_FINAL_TOP_K,
    MODEL_SPECS,
    QUERY_ROLES,
)
from online.src.retrieval.branches.final_fusion.contracts import DEFAULT_BRANCH_WEIGHTS


class ParseRequest(BaseModel):
    query: str
    task_type: TaskType | None = "KIS"
    engine: Literal["local", "direct", "rule"] = "local"


class SearchRequest(BaseModel):
    parsed_query: ParsedQuery
    session_id: str | None = None
    top_k: int = Field(default=20, ge=1, le=100)


class Branch1Query(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["original", "entity", "action", "context", "synonym", "keyword"]
    vi: str = Field(min_length=1, max_length=4096)
    en: str = Field(min_length=1, max_length=4096)

    @field_validator("vi", "en")
    @classmethod
    def require_nonempty_language(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned


class Branch1QueryBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["branch1.query.v1"]
    queries: list[Branch1Query] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def require_exact_roles(self):
        roles = [query.role for query in self.queries]
        if len(set(roles)) != 6 or set(roles) != set(QUERY_ROLES):
            raise ValueError(
                f"queries must contain each role exactly once: {', '.join(QUERY_ROLES)}"
            )
        return self


class Branch1ModelWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    siglip2: float = Field(default=0.45, ge=0)
    metaclip2: float = Field(default=0.30, ge=0)
    beit3: float = Field(default=0.25, ge=0)

    @model_validator(mode="after")
    def require_positive_sum(self):
        values = (self.siglip2, self.metaclip2, self.beit3)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("model weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("model weight sum must be greater than zero")
        return self

    def normalized(self) -> dict[str, float]:
        values = self.model_dump()
        total = sum(values.values())
        return {name: float(values[name]) / total for name in MODEL_SPECS}


class Branch1SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_bundle: Branch1QueryBundle
    model_weights: Branch1ModelWeights = Field(default_factory=Branch1ModelWeights)
    per_stream_top_k: int = Field(default=2000, ge=1, le=2000)
    final_top_k: Literal[1500] = BRANCH1_FINAL_TOP_K


class Branch2Weights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dense: float = Field(default=0.70, ge=0)
    sparse: float = Field(default=0.30, ge=0)

    @model_validator(mode="after")
    def positive_sum(self):
        values = (self.dense, self.sparse)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("hybrid weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("hybrid weight sum must be greater than zero")
        return self

    def normalized(self) -> dict[str, float]:
        total = self.dense + self.sparse
        return {"dense": self.dense / total, "sparse": self.sparse / total}


class Branch2RerankWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beit3: float = Field(default=0.40, ge=0)
    previous: float = Field(default=0.60, ge=0)

    @model_validator(mode="after")
    def positive_sum(self):
        values = (self.beit3, self.previous)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("rerank weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("rerank weight sum must be greater than zero")
        return self

    def normalized(self) -> dict[str, float]:
        total = self.beit3 + self.previous
        return {"beit3": self.beit3 / total, "previous": self.previous / total}


class Branch2SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_bundle: Branch1QueryBundle
    hybrid_weights: Branch2Weights = Field(default_factory=Branch2Weights)
    rerank_weights: Branch2RerankWeights = Field(default_factory=Branch2RerankWeights)
    per_stream_top_k: int = Field(default=2000, ge=1, le=2000)
    pre_rerank_top_k: Literal[500] = 500
    # BEiT-3 is a restricted candidate validator, never a second retrieval
    # stage.  Keep the public boundary aligned with the fixed top-100 policy;
    # the service and reranker repeat this check defensively.
    rerank_top_k: int = Field(default=100, ge=1, le=100)

    @model_validator(mode="after")
    def valid_rerank_limit(self):
        if self.rerank_top_k > self.pre_rerank_top_k:
            raise ValueError("rerank_top_k cannot exceed pre_rerank_top_k")
        return self


class Branch3AsrSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_bundle: Branch1QueryBundle
    per_stream_top_k: int = Field(default=2000, ge=1, le=2000)
    # Branch 3 exposes a bounded API pool: callers may request a smaller
    # result set for inspection, but the modality can never return more than
    # the 500-frame gate.  The UI uses the full gate (500).
    final_top_k: int = Field(default=500, ge=1, le=500)


class Branch3OcrSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_bundle: Branch1QueryBundle
    per_stream_top_k: int = Field(default=2000, ge=1, le=2000)
    final_top_k: int = Field(default=500, ge=1, le=500)


class KisBranchWeights(BaseModel):
    """Positive four-voter weights; the service normalizes them server-side."""

    model_config = ConfigDict(extra="forbid")

    branch1: float = Field(default=DEFAULT_BRANCH_WEIGHTS["branch1"], gt=0)
    branch2: float = Field(default=DEFAULT_BRANCH_WEIGHTS["branch2"], gt=0)
    ocr: float = Field(default=DEFAULT_BRANCH_WEIGHTS["ocr"], gt=0)
    asr: float = Field(default=DEFAULT_BRANCH_WEIGHTS["asr"], gt=0)

    @field_validator("branch1", "branch2", "ocr", "asr", mode="before")
    @classmethod
    def reject_boolean_weight(cls, value: Any) -> Any:
        # JSON booleans are not numeric slider weights.  Pydantic's normal
        # float coercion would otherwise turn ``true`` into ``1.0``.
        if isinstance(value, bool):
            raise ValueError("KIS branch weights must be numeric, not boolean")
        return value

    @model_validator(mode="after")
    def finite_positive(self):
        values = (self.branch1, self.branch2, self.ocr, self.asr)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("KIS branch weights must be finite and greater than zero")
        if sum(values) <= 0:
            raise ValueError("KIS branch weight sum must be greater than zero")
        return self

    def normalized(self) -> dict[str, float]:
        total = self.branch1 + self.branch2 + self.ocr + self.asr
        return {
            "branch1": self.branch1 / total,
            "branch2": self.branch2 / total,
            "ocr": self.ocr / total,
            "asr": self.asr / total,
        }


class KisFusionSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_bundle: Branch1QueryBundle
    branch_weights: KisBranchWeights = Field(default_factory=KisBranchWeights)


class KisQueryPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4096)
    task_type: Literal["KIS", "VQA", "TRAKE"] = "KIS"
    query_bundle: Branch1QueryBundle | None = None


class KisTemporalEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int = Field(ge=1, le=6)
    description: str = Field(min_length=1, max_length=4096)
    vi: str | None = Field(default=None, max_length=4096)
    en: str | None = Field(default=None, max_length=4096)

    @field_validator("description", "vi", "en")
    @classmethod
    def clean_event_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned


class KisTemporalSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["KIS", "VQA", "TRAKE"] = "KIS"
    query_bundle: Branch1QueryBundle
    events: list[KisTemporalEventRequest] = Field(min_length=2, max_length=6)
    branch_weights: KisBranchWeights = Field(default_factory=KisBranchWeights)
    top_k_sequences: int = Field(default=100, ge=1, le=100)
    max_gap_seconds: float | None = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def require_contiguous_event_order(self):
        orders = sorted(event.order for event in self.events)
        if orders != list(range(1, len(self.events) + 1)):
            raise ValueError("event orders must be contiguous from 1")
        return self


class VideoVisualSearchRequest(BaseModel):
    parsed_query: ParsedQuery
    top_k: int = Field(default=50, ge=1, le=100)


class VideoVisualFusionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4096)
    query_bundle: Branch1QueryBundle | None = None
    top_k: int = Field(default=50, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def require_visual_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned


class DiscoveryCascadeRequest(BaseModel):
    parsed_query: ParsedQuery
    dam_top_frames_per_object: int = Field(default=20, ge=1, le=50)
    siglip_top_frames_per_video: int = Field(default=10, ge=1, le=20)


class TemporalVisualEventRequest(BaseModel):
    order: int = Field(ge=1)
    description: str
    global_scene_en: str

    @field_validator("description", "global_scene_en")
    @classmethod
    def require_nonempty_event_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned


class TemporalIntersectionRequest(BaseModel):
    events: list[TemporalVisualEventRequest] = Field(min_length=2, max_length=6)
    anchor_query: str | None = None
    top_k_per_event: int = Field(default=300, ge=1, le=1000)
    top_k_sequences: int = Field(default=20, ge=1, le=100)
    max_gap_seconds: float | None = Field(default=30.0, gt=0)
    anchor_event_order: int | None = None
    paths_per_video: int = Field(default=1, ge=1, le=10)
    sequence_reservoir_size: int | None = Field(default=None, ge=1, le=500)
    path_beam_width: int | None = Field(default=None, ge=1, le=2048)
    path_diversity_min_events: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_options(self):
        orders = [event.order for event in self.events]
        if len(set(orders)) != len(orders):
            raise ValueError("event orders must be unique")
        if self.anchor_event_order is not None and self.anchor_event_order not in set(orders):
            raise ValueError("anchor_event_order must identify one supplied event")
        if (
            self.sequence_reservoir_size is not None
            and self.sequence_reservoir_size < self.top_k_sequences
        ):
            raise ValueError("sequence_reservoir_size cannot be smaller than top_k_sequences")
        if self.path_diversity_min_events > len(self.events):
            raise ValueError("path_diversity_min_events cannot exceed the event count")
        self.anchor_query = (self.anchor_query or "").strip() or None
        return self


class SubmissionPrepareRequest(BaseModel):
    task_type: Literal["KIS", "VQA", "QA", "TRAKE"] = "KIS"
    mode: str | None = Field(default=None, max_length=64)
    query_id: str = Field(default="1", min_length=1, max_length=64)
    target_rows: int = Field(default=100, ge=1, le=100)
    manual_selections: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    candidate_reservoir: list[dict[str, Any]] = Field(default_factory=list, max_length=5000)
    vqa_answer: str | None = Field(default=None, max_length=100)
    event_count: int | None = Field(default=None, ge=2, le=100)
    manual_sequences: list[Any] = Field(default_factory=list, max_length=100)
    candidate_sequences: list[Any] = Field(default_factory=list, max_length=500)

    @field_validator("query_id")
    @classmethod
    def validate_query_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", cleaned):
            raise ValueError("query_id may contain only letters, numbers, underscores, and hyphens")
        return cleaned


class RelatedFramesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1, max_length=32)
    frame_idx: int = Field(ge=0)
    limit: int = Field(default=99, ge=1, le=99)


__all__ = [
    "Branch1ModelWeights",
    "Branch1Query",
    "Branch1QueryBundle",
    "Branch1SearchRequest",
    "Branch2RerankWeights",
    "Branch2SearchRequest",
    "Branch2Weights",
    "Branch3AsrSearchRequest",
    "Branch3OcrSearchRequest",
    "DiscoveryCascadeRequest",
    "KisBranchWeights",
    "KisFusionSearchRequest",
    "KisQueryPlanRequest",
    "KisTemporalEventRequest",
    "KisTemporalSearchRequest",
    "ParseRequest",
    "RelatedFramesRequest",
    "SearchRequest",
    "SubmissionPrepareRequest",
    "TemporalIntersectionRequest",
    "TemporalVisualEventRequest",
    "VideoVisualFusionRequest",
    "VideoVisualSearchRequest",
]
