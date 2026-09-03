"""CPU-only workbench server backed by local Qdrant and deterministic parsing."""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from online.src.contracts.query import ParsedQuery, TaskType
from online.src.retrieval.encoders.worker_manager import (
    EncoderWorkerManager,
    ProcessBranch1Encoders,
    ProcessCpuTextEncoders,
)
from online.src.retrieval.branches.branch1.health import branch1_health
from online.src.retrieval.branches.branch1.service import Branch1Search
from online.src.retrieval.infrastructure.persistent_cache import PersistentQueryEmbeddingCache
from online.src.retrieval.branches.branch1.contracts import (
    BRANCH1_FINAL_TOP_K,
    MODEL_SPECS,
    QUERY_ROLES,
)
from online.src.retrieval.branches.branch2.service import Branch2Search
from online.src.retrieval.branches.branch3.service import Branch3AsrSearch, Branch3OcrSearch
from online.src.retrieval.branches.final_fusion.service import KisFusionSearch
from online.src.retrieval.branches.final_fusion.contracts import DEFAULT_BRANCH_WEIGHTS
from online.src.retrieval.infrastructure.query_parser import LocalQueryParser
from online.src.retrieval.modalities.temporal import IndependentModalitySearch
from online.src.retrieval.infrastructure.qdrant import QdrantHttpClient
from online.src.retrieval.modalities.asr import AsrFtsIndex
from online.src.retrieval.modalities.visual import CpuQdrantSearch
from online.src.retrieval.infrastructure.metadata import FrameMetadataStore
from online.src.retrieval.infrastructure.resources import current_process_rss_bytes, resource_qualification
from online.src.retrieval.modalities.ocr import OcrFtsIndex
from online.src.submission import prepare_submission


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("aic_cpu_workbench")
DATA_ROOT = Path(os.environ.get("AIC_DATA_ROOT", "/data"))
STATE_ROOT = Path(os.environ.get("AIC_STATE_ROOT", "/state"))
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
KEYFRAMES_ROOT = DATA_ROOT / "keyframes"
MEDIA_INFO_ROOT = Path(os.environ.get("AIC_MEDIA_INFO_ROOT", str(DATA_ROOT / "media_info")))
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend" / "dist"
MAX_IMAGE_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MAX_IMAGE_DIMENSION = 8192
ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

parser = LocalQueryParser()
encoder_workers: EncoderWorkerManager | None = None
encoders: ProcessCpuTextEncoders | None = None
searcher: CpuQdrantSearch | None = None
workbench_search: IndependentModalitySearch | None = None
ocr_index: OcrFtsIndex | None = None
asr_index: AsrFtsIndex | None = None
metadata_store: FrameMetadataStore | None = None
branch1_encoders: ProcessBranch1Encoders | None = None
branch1_searcher: Branch1Search | None = None
branch1_qdrant: QdrantHttpClient | None = None
branch2_searcher: Branch2Search | None = None
branch3_asr_searcher: Branch3AsrSearch | None = None
branch3_ocr_searcher: Branch3OcrSearch | None = None
kis_fusion_searcher: KisFusionSearch | None = None
qdrant_client: QdrantHttpClient | None = None
query_cache: PersistentQueryEmbeddingCache | None = None
heavy_search_lock = threading.Lock()
_health_cache_lock = threading.RLock()
_health_cache: tuple[float, tuple[tuple[str, int, int], ...], dict[str, Any]] | None = None
HEALTH_CACHE_SECONDS = 5.0


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
            raise ValueError(f"queries must contain each role exactly once: {', '.join(QUERY_ROLES)}")
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


class VideoVisualSearchRequest(BaseModel):
    parsed_query: ParsedQuery
    top_k: int = Field(default=50, ge=1, le=100)


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
        if self.sequence_reservoir_size is not None and self.sequence_reservoir_size < self.top_k_sequences:
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


def _canonical_video_id(value: str) -> str:
    canonical = value.strip().upper().replace("-", "_")
    if not re.fullmatch(r"[A-Z0-9]+_V\d+", canonical):
        raise HTTPException(status_code=400, detail="Invalid video ID")
    return canonical


def _pool(
    modality: str,
    display_name: str,
    query: str | list[str],
    query_source: str,
    score_type: str,
    score_description: str,
    function: Any | None,
    reason: str = "",
) -> dict[str, Any]:
    model_identity = {
        "siglip": (
            "google/siglip2-base-patch16-224",
            "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        ),
        "dam": ("BAAI/bge-m3", "local-cache"),
        "ocr": ("sqlite-fts5", "local"),
        "asr": ("sqlite-fts5", "local"),
    }.get(modality, ("", ""))
    if function is None:
        return {
            "modality": modality,
            "display_name": display_name,
            "status": "not_run",
            "reason": reason,
            "query": query,
            "query_source": query_source,
            "score_type": score_type,
            "score_description": score_description,
            "provenance": {
                "query_modality": query_source,
                "query_tower": modality,
                "index_modality": modality,
                "embedding_model_id": model_identity[0],
                "embedding_model_revision": model_identity[1],
                "fusion_applied": False,
                "reranking_applied": False,
            },
            "result_count": 0,
            "execution_time_ms": 0.0,
            "results": [],
        }
    started = time.perf_counter()
    results = function()
    return {
        "modality": modality,
        "display_name": display_name,
        "status": "ok",
        "reason": "",
        "query": query,
        "query_source": query_source,
        "score_type": score_type,
        "score_description": score_description,
        "provenance": {
            "query_modality": query_source,
            "query_tower": modality,
            "index_modality": modality,
            "embedding_model_id": model_identity[0],
            "embedding_model_revision": model_identity[1],
            "fusion_applied": False,
            "reranking_applied": False,
        },
        "result_count": len(results),
        "execution_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "results": results,
    }


def _execute_search(parsed: ParsedQuery, top_k: int) -> dict[str, dict[str, Any]]:
    # Text modalities are optional.  The generic workbench search must not
    # make an OCR-only request depend on an ASR SQLite handle being present;
    # each populated modality pool enforces its own readiness at execution.
    assert searcher is not None
    visual_query = parsed.global_scene_en.strip() or parsed.original_query.strip()
    object_queries = [value.strip() for value in parsed.objects_en if value.strip()]
    ocr_keywords = [value.strip() for value in parsed.ocr_keywords if value.strip()]
    # ASR is opt-in from the parser contract.  Falling back to the visual
    # description would run an audio search for every ordinary query and would
    # also make an absent ASR index look like a hard dependency.
    speech_query = parsed.speech_vi.strip()
    return {
        "siglip": _pool(
            "siglip", "SigLIP2 visual scene (CPU + Qdrant)", visual_query,
            "global_scene_en", "cosine",
            "Qdrant cosine between multilingual SigLIP2 text and full-frame embeddings",
            (lambda: searcher.search_siglip(visual_query, top_k)) if visual_query else None,
            "The query is empty.",
        ),
        "dam": _pool(
            "dam", "DAM regions (CPU BGE-M3 + Qdrant)", object_queries,
            "objects_en", "mean_best_region_cosine",
            "Best DAM region per phrase, aggregated at parent-frame level",
            (lambda: searcher.search_dam_queries(object_queries, top_k)) if object_queries else None,
            "No object query is available.",
        ),
        "ocr": _pool(
            "ocr", "OCR text (local SQLite FTS5)", ocr_keywords,
            "ocr_keywords", "keyword_match_ratio",
            "Local FTS5 candidates ranked by exact keyword coverage and BM25",
            (
                lambda: branch3_ocr_searcher.execute_single(
                    " ".join(ocr_keywords), top_k
                )
            )
            if ocr_keywords
            else None,
            "No OCR keyword could be extracted locally.",
        ),
        "asr": _pool(
            "asr", "ASR spoken speech (local BM25 + n-gram)", speech_query,
            "speech_vi", "bm25_ngram",
            "SQLite FTS5 Okapi BM25 combined with token and adjacent n-gram coverage",
            (lambda: searcher.search_speech(speech_query, top_k=top_k)) if speech_query else None,
            "No speech query is available.",
        ),
    }


def _decode_image(payload: bytes, content_type: str | None) -> tuple[Image.Image, dict[str, Any]]:
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    if len(payload) > MAX_IMAGE_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image is too large")
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP are supported")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            if max(source.size) > MAX_IMAGE_DIMENSION or source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Uploaded image dimensions are too large")
            image = ImageOps.exif_transpose(source).convert("RGB").copy()
            image_format = (source.format or "unknown").lower()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Uploaded image could not be decoded") from error
    return image, {"format": image_format, "width": image.width, "height": image.height, "byte_count": len(payload)}


def _normalize_frame(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    if "video_id" in result:
        result["video_id"] = _canonical_video_id(str(result["video_id"]))
    return result


def _normalize_sequence(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_frame(frame) for frame in value]
    if not isinstance(value, dict):
        return value
    result = dict(value)
    for field in ("matched_events", "events", "frames"):
        if isinstance(result.get(field), list):
            result[field] = [_normalize_frame(frame) for frame in result[field]]
    return result


def _normalize_submission(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    for field in ("manual_selections", "candidate_reservoir"):
        result[field] = [_normalize_frame(frame) for frame in result.get(field, [])]
    for field in ("manual_sequences", "candidate_sequences"):
        result[field] = [_normalize_sequence(sequence) for sequence in result.get(field, [])]
    return result


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global encoders, searcher, workbench_search, ocr_index, asr_index, metadata_store
    global encoder_workers, branch1_encoders, branch1_searcher, branch1_qdrant, branch2_searcher, branch3_asr_searcher, branch3_ocr_searcher, kis_fusion_searcher, qdrant_client, query_cache
    global _health_cache
    started = time.perf_counter()
    qdrant = QdrantHttpClient(QDRANT_URL)
    qdrant_client = qdrant
    ocr_index = OcrFtsIndex(
        DATA_ROOT / "ocr_transcripts",
        STATE_ROOT / "ocr.sqlite3",
        manifest_path=STATE_ROOT / "branch3_ocr_manifest.json",
        canonical_metadata_path=(
            DATA_ROOT / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
        ),
    )
    metadata_store = FrameMetadataStore(DATA_ROOT, ocr_index)
    ocr_index.metadata = metadata_store
    asr_index = AsrFtsIndex(
        DATA_ROOT / "asr_segments",
        STATE_ROOT / "asr.sqlite3",
        metadata_store,
        # ASR has its own atomic preparation command.  Server startup only
        # opens the prepared database and must never rebuild it.
        auto_prepare=False,
    )
    branch3_asr_searcher = Branch3AsrSearch(asr_index, heavy_search_lock)
    branch3_ocr_searcher = Branch3OcrSearch(ocr_index, heavy_search_lock)
    encoder_workers = EncoderWorkerManager()
    encoders = ProcessCpuTextEncoders(encoder_workers)
    searcher = CpuQdrantSearch(
        qdrant,
        encoders,
        ocr_index,
        metadata_store,
        asr_service=branch3_asr_searcher,
        ocr_service=branch3_ocr_searcher,
    )
    workbench_search = IndependentModalitySearch(searcher=searcher, registry=encoders, dam_match_threshold=0.50)
    branch1_qdrant = qdrant
    branch1_encoders = ProcessBranch1Encoders(encoder_workers)
    query_cache = PersistentQueryEmbeddingCache(STATE_ROOT / "query_embeddings.sqlite3")
    branch1_searcher = Branch1Search(
        qdrant,
        branch1_encoders,
        query_cache,
        heavy_search_lock,
    )
    branch2_searcher = Branch2Search(
        qdrant,
        DATA_ROOT,
        STATE_ROOT,
        encoders,
        branch1_encoders,
        query_cache,
        heavy_search_lock,
    )
    kis_fusion_searcher = KisFusionSearch(
        branch1_searcher,
        branch2_searcher,
        branch3_asr_searcher,
        branch3_ocr_searcher,
        data_root=DATA_ROOT,
        state_root=STATE_ROOT,
        search_lock=heavy_search_lock,
    )
    with _health_cache_lock:
        _health_cache = None
    LOGGER.info("CPU workbench server ready in %.1fs", time.perf_counter() - started)
    try:
        yield
    finally:
        if encoder_workers is not None:
            encoder_workers.close_active()
        if query_cache is not None:
            query_cache.close()
        if asr_index is not None:
            asr_index.close()
        if ocr_index is not None:
            ocr_index.close()
        try:
            qdrant.close()
        except AttributeError:
            pass


app = FastAPI(title="AIC-2026 CPU-only retrieval workbench", lifespan=lifespan)


def _qdrant_health() -> dict[str, Any]:
    if qdrant_client is None:
        return {"ready": False, "error": "Qdrant client is not initialized"}
    try:
        value = qdrant_client.collection("aic_frames")
        collection_status = value.get("status")
        return {
            "ready": collection_status == "green",
            "url": QDRANT_URL,
            "collection": "aic_frames",
            "status": collection_status,
            "points_count": value.get("points_count"),
            "fail_closed": collection_status != "green",
        }
    except Exception as error:
        return {"ready": False, "url": QDRANT_URL, "error": str(error)}


async def _safe_health_call(function: Any, *args: Any) -> dict[str, Any]:
    try:
        value = await run_in_threadpool(function, *args)
        return value if isinstance(value, dict) else {"ready": False, "error": "invalid health response"}
    except Exception as error:
        return {"ready": False, "status": "not_ready", "error": str(error), "fail_closed": True}


def _health_cache_signature() -> tuple[tuple[str, int, int], ...]:
    """Return cheap stat fingerprints for health inputs.

    The cache avoids repeating Qdrant and manifest checks for a few seconds,
    but an artifact/model manifest replacement must invalidate it immediately.
    Hashing large matrices is intentionally not part of this hot path; their
    hashes are checked by the preparation gates and their stat is chained here.
    """
    paths = (
        STATE_ROOT / "branch1_data_gate.json",
        STATE_ROOT / "branch1_encoder_compatibility.json",
        STATE_ROOT / "qdrant_ingestion_manifest.json",
        STATE_ROOT / "branch2_dam_manifest.json",
        STATE_ROOT / "branch3_asr_manifest.json",
        STATE_ROOT / "asr.sqlite3",
        STATE_ROOT / "ocr.sqlite3",
        DATA_ROOT / "ocr_transcripts",
        STATE_ROOT / "branch2_dam_bm25.sqlite3",
        STATE_ROOT / "branch3_ocr_manifest.json",
        STATE_ROOT / "resource_qualification.json",
        STATE_ROOT / "runtime_fingerprint.json",
        Path(os.environ.get("AIC_QUERY_MODEL_MANIFEST", "/models/query_models.json")),
        Path(os.environ.get("AIC_BRANCH1_MODEL_ROOT", "/models/branch1")) / "manifest.json",
        DATA_ROOT / "visual_embeddings" / "metaclip2" / "keyframes_visual_vectors.f16.npy",
        DATA_ROOT / "visual_embeddings" / "beit3" / "keyframes_visual_vectors.f16.npy",
        DATA_ROOT / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl",
        DATA_ROOT / "visual_embeddings" / "beit3" / "keyframes_metadata.jsonl",
        *tuple(sorted((DATA_ROOT / "scene_embeddings").glob("*.safetensors"))),
    )
    # Model setup records immutable asset inventories.  Include their cheap
    # stats in the cache key so a changed/deleted weight invalidates readiness
    # without rehashing multi-GB files on every health request.
    manifest_paths = (
        Path(os.environ.get("AIC_QUERY_MODEL_MANIFEST", "/models/query_models.json")),
        Path(os.environ.get("AIC_BRANCH1_MODEL_ROOT", "/models/branch1")) / "manifest.json",
    )
    asset_paths: list[Path] = []
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            groups = [manifest.get("models") or {}, {"branch1": {"files": list((manifest.get("assets") or {}).values())}}]
            for group in groups:
                for model in group.values():
                    for item in model.get("files") or []:
                        if isinstance(item, dict) and item.get("path"):
                            asset_paths.append(Path(str(item["path"])))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    paths = (*paths, *asset_paths)
    signature: list[tuple[str, int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
            signature.append((str(path), int(stat.st_size), int(stat.st_mtime_ns)))
        except OSError:
            signature.append((str(path), -1, -1))
    return tuple(signature)


def _fusion_production_ready(
    fusion_ready: bool,
    branch_states: tuple[dict[str, Any], ...],
    resource_state: dict[str, Any] | None,
) -> bool:
    """Apply the strict KIS production gate after resource validation.

    Operational readiness and production qualification are intentionally
    separate.  A fusion endpoint may be searchable while provenance or RAM
    qualification is pending, but a missing/malformed resource report or a
    branch that omitted an explicit production attestation must never be
    interpreted as production-ready.
    """

    if not fusion_ready or not isinstance(resource_state, dict):
        return False
    if resource_state.get("production_ready") is not True:
        return False
    return all(
        isinstance(state, dict) and state.get("production_ready") is True
        for state in branch_states
    )


async def _dependency_health(*, force: bool = False) -> dict[str, Any]:
    global _health_cache
    now = time.monotonic()
    signature = _health_cache_signature()
    with _health_cache_lock:
        if (
            not force
            and _health_cache is not None
            and now - _health_cache[0] < HEALTH_CACHE_SECONDS
            and _health_cache[1] == signature
        ):
            return _health_cache[2]
    # The aggregate snapshot can be built from the canonical branch services;
    # the legacy CpuQdrantSearch object is not a prerequisite for KIS fusion
    # health.  Treat the process as starting only before any canonical service
    # has been wired by lifespan.
    runtime_initialized = any(
        service is not None
        for service in (
            branch1_searcher,
            branch2_searcher,
            branch3_asr_searcher,
            branch3_ocr_searcher,
            kis_fusion_searcher,
        )
    )
    if not runtime_initialized:
        snapshot = {
            "status": "starting",
            "device": "cpu",
            "ready": False,
            "production_ready": False,
            "ocr": {"status": "starting", "ready": False, "production_ready": False, "required": False, "fail_closed": True},
            "asr": {"status": "starting", "ready": False, "production_ready": False, "required": False, "fail_closed": True},
            "branch3_ocr": {"status": "starting", "ready": False, "production_ready": False, "required": False, "fail_closed": True},
            "kis_fusion": {
                "schema_version": "kis.fusion.health.v1",
                "branch": "final_fusion",
                "task_type": "KIS",
                "status": "starting",
                "ready": False,
                "production_ready": False,
                "required": False,
                "fail_closed": True,
            },
            "qwen_enabled": False,
            "gemini_enabled": False,
            "optional_degraded_components": ["branch3_asr", "branch3_ocr", "kis_fusion"],
            "estimated_peak_total_rss_bytes": 0,
            "api_rss_bytes": current_process_rss_bytes(),
            "peak_worker_rss_bytes": 0,
            "components": {
                "api_process": {"ready": False},
                "qdrant": {"ready": False},
                "metadata": {"ready": False},
                "ocr": {"status": "starting", "ready": False, "production_ready": False, "required": False, "fail_closed": True},
                "asr": {"status": "starting", "ready": False, "production_ready": False, "required": False, "fail_closed": True},
                "image_search": {"ready": False},
                "siglip_text": {"ready": False},
                "branch1": {"ready": False},
                "branch2": {"ready": False},
                "branch3_asr": {"status": "starting", "ready": False, "production_ready": False, "required": False, "fail_closed": True},
                "branch3_ocr": {"status": "starting", "ready": False, "production_ready": False, "required": False, "fail_closed": True},
                "kis_fusion": {
                    "schema_version": "kis.fusion.health.v1",
                    "branch": "final_fusion",
                    "task_type": "KIS",
                    "status": "starting",
                    "ready": False,
                    "production_ready": False,
                    "required": False,
                    "fail_closed": True,
                },
                "resource_qualification": {"ready": False},
            },
        }
    else:
        qdrant_state = await run_in_threadpool(_qdrant_health)
        branch1_state = (
            await _safe_health_call(
                branch1_health, DATA_ROOT, branch1_qdrant, branch1_encoders, STATE_ROOT
            )
            if branch1_qdrant is not None and branch1_encoders is not None
            else {"ready": False, "status": "starting"}
        )
        branch2_state = (
            await _safe_health_call(branch2_searcher.health)
            if branch2_searcher is not None
            else {"ready": False, "status": "starting"}
        )
        branch1_models = branch1_state.get("models")
        branch1_models = branch1_models if isinstance(branch1_models, dict) else {}
        siglip_state = branch1_models.get("siglip2")
        siglip_state = siglip_state if isinstance(siglip_state, dict) else {}
        siglip_sections = tuple(
            section if isinstance(section, dict) else {}
            for section in (
                siglip_state.get("data", {}),
                siglip_state.get("collection", {}),
                siglip_state.get("ingestion", {}),
                siglip_state.get("text_encoder", {}),
            )
        )
        siglip_text_ready = all(
            section.get("ready") is True
            for section in siglip_sections
        )
        siglip_image_ready = siglip_text_ready and (
            siglip_sections[-1].get("image_ready") is True
        )
        canonical_metadata = DATA_ROOT / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
        video_metadata_dir = getattr(metadata_store, "video_metadata_dir", None)
        video_metadata_dir = (
            video_metadata_dir if isinstance(video_metadata_dir, Path) else None
        )
        metadata_ready = bool(
            metadata_store is not None
            and canonical_metadata.is_file()
            and video_metadata_dir is not None
            and video_metadata_dir.is_dir()
            and any(video_metadata_dir.glob("*.jsonl"))
        )
        timeline_ready = bool(
            metadata_store is not None
            and video_metadata_dir is not None
            and video_metadata_dir.is_dir()
            and any(video_metadata_dir.glob("*.jsonl"))
        )
        branch3_ocr_state = (
            await _safe_health_call(branch3_ocr_searcher.health)
            if branch3_ocr_searcher is not None
            else {"ready": False, "status": "starting"}
        )
        branch3_ocr_state = {**branch3_ocr_state, "required": False}
        # The legacy OCR component name remains for compatibility; Branch 3
        # OCR is the canonical readiness source for the bilingual workspace.
        ocr_state = branch3_ocr_state
        branch3_asr_state = (
            await _safe_health_call(branch3_asr_searcher.health)
            if branch3_asr_searcher is not None
            else {"ready": False, "status": "starting"}
        )
        branch3_asr_state = {**branch3_asr_state, "required": False}
        # The legacy ASR component name remains for response compatibility,
        # but Branch 3 is the single source of truth for its readiness.
        asr_state = branch3_asr_state
        raw_branch2_components = branch2_state.get("components")
        # Health adapters are untrusted input at this boundary.  Only the
        # canonical object shape is eligible for capability checks; a list or
        # scalar must fail closed rather than causing an AttributeError.
        branch2_components = (
            raw_branch2_components if isinstance(raw_branch2_components, dict) else {}
        )
        # KIS fusion is an optional capability at the server level, but it is
        # fail-closed internally: all four branch pools must be ready before
        # the endpoint is advertised.  Reuse the branch states already
        # collected in this snapshot instead of issuing nested health scans.
        # The aggregate capability must reflect the same concrete objects the
        # fusion endpoint will use, not merely four green branch summaries.
        # These checks are intentionally lightweight (no corpus scan): the
        # branch health payloads already own collection/index validation.
        fusion_execution_ready = kis_fusion_searcher is not None and all(
            callable(getattr(service, "_execute_locked", None))
            for service in (
                branch1_searcher,
                branch2_searcher,
                branch3_asr_searcher,
                branch3_ocr_searcher,
            )
        )
        fusion_mapping_ready = bool(
            canonical_metadata.is_file()
            and isinstance(raw_branch2_components, dict)
            and isinstance(branch2_components.get("frame_mapping"), dict)
            and branch2_components["frame_mapping"].get("ready") is True
        )
        fusion_beit_ready = bool(
            isinstance(raw_branch2_components, dict)
            and all(
                isinstance(branch2_components.get(name), dict)
                and branch2_components[name].get("ready") is True
                for name in ("beit3_collection", "beit3_ingestion", "beit3_text_encoder")
            )
        )
        cache_ready = query_cache is not None and all(
            callable(getattr(query_cache, method, None))
            for method in ("key", "get", "put")
        )
        fusion_ready = (
            fusion_execution_ready
            and cache_ready
            and fusion_mapping_ready
            and fusion_beit_ready
            and all(
                state.get("ready") is True
                for state in (branch1_state, branch2_state, branch3_asr_state, branch3_ocr_state)
            )
        )
        fusion_state = {
            "schema_version": "kis.fusion.health.v1",
            "branch": "final_fusion",
            "task_type": "KIS",
            "status": "ready" if fusion_ready else "not_ready",
            "ready": fusion_ready,
            "required": False,
            # Filled after resource qualification is read below.  Keeping a
            # false placeholder prevents a partially-built state from ever
            # being observed as production-ready.
            "production_ready": False,
            "fail_closed": True,
            "rrf_k": 60,
            "final_top_k": 150,
            "rerank_top_k": 100,
            "branch_pool_limits": {
                "branch1": 1500,
                "branch2": 500,
                "ocr": 500,
                "asr": 500,
            },
            "beit3_weight": 0.25,
            "rrf_weight": 0.75,
            "branch_weights": dict(DEFAULT_BRANCH_WEIGHTS),
            "branches": {
                "branch1": branch1_state,
                "branch2": branch2_state,
                "asr": branch3_asr_state,
                "ocr": branch3_ocr_state,
            },
            "canonical_frame_mapping": {
                "ready": fusion_mapping_ready,
                "path": str(canonical_metadata),
            },
            "query_cache": {"ready": cache_ready, "persistent": True},
            "execution_contract": {
                "ready": fusion_execution_ready,
                "branches": {
                    name: callable(getattr(service, "_execute_locked", None))
                    for name, service in (
                        ("branch1", branch1_searcher),
                        ("branch2", branch2_searcher),
                        ("asr", branch3_asr_searcher),
                        ("ocr", branch3_ocr_searcher),
                    )
                },
            },
            "beit3": {
                "ready": fusion_beit_ready,
            },
        }
        try:
            resource_state = resource_qualification(STATE_ROOT)
            if not isinstance(resource_state, dict):
                raise ValueError("invalid resource qualification response")
        except Exception as error:
            # Resource qualification is diagnostic, but malformed state must
            # never turn a health/config request into a 500 or a false green
            # production gate.
            resource_state = {
                "ready": False,
                "production_ready": False,
                "fail_closed": True,
                "error": str(error),
            }
        fusion_state["resource_qualification"] = resource_state
        fusion_state["api_rss_bytes"] = current_process_rss_bytes()
        fusion_state["warnings"] = (
            ["KIS fusion requires all four branch pools to be ready"]
            if not fusion_ready
            else []
        )
        resource_ready = bool(resource_state.get("production_ready") is True)
        fusion_production_ready = _fusion_production_ready(
            fusion_ready,
            (branch1_state, branch2_state, branch3_asr_state, branch3_ocr_state),
            resource_state,
        )
        fusion_state["production_ready"] = fusion_production_ready
        components = {
            "api_process": {"ready": True},
            "qdrant": qdrant_state,
            "metadata": {
                "ready": metadata_ready,
                "canonical_metadata": str(canonical_metadata),
            },
            "ocr": ocr_state,
            "asr": asr_state,
            "image_search": {"ready": siglip_image_ready},
            "siglip_text": {"ready": siglip_text_ready},
            "branch1": branch1_state,
            "branch2": branch2_state,
            "branch3_asr": branch3_asr_state,
            "branch3_ocr": branch3_ocr_state,
            "kis_fusion": fusion_state,
            "resource_qualification": {
                **resource_state,
                "estimated_peak_total_rss_bytes": (
                    0
                    if encoder_workers is None
                    else encoder_workers.estimated_peak_total_rss_bytes
                ),
                "api_rss_bytes": current_process_rss_bytes(),
                "peak_worker_rss_bytes": (
                    0 if encoder_workers is None else encoder_workers.peak_worker_rss_bytes
                ),
            },
        }
        core_ready = all(
            components[name].get("ready") is True
            for name in ("api_process", "qdrant", "metadata")
        )
        search_ready = (
            core_ready
            and branch1_state.get("ready") is True
            and branch2_state.get("ready") is True
        )
        # A green required search path is not sufficient for production
        # qualification.  Branch health carries the immutable provenance and
        # model/resource gates; propagate those gates instead of allowing an
        # old resource report to make the overall stack appear production
        # ready while an embedding revision remains unverified.
        provenance_ready = all(
            state.get("production_ready") is True
            for state in (branch1_state, branch2_state)
        )
        production_ready = search_ready and resource_ready and provenance_ready
        optional_degraded_components = [
            name for name in ("branch3_asr", "branch3_ocr")
            if components.get(name, {}).get("ready") is not True
        ]
        if not fusion_ready:
            optional_degraded_components.append("kis_fusion")
        snapshot = {
            "status": "ready" if search_ready else "degraded",
            "device": "cpu",
            "production_ready": production_ready,
            "ocr": ocr_state,
            "asr": asr_state,
            "kis_fusion": fusion_state,
            "estimated_peak_total_rss_bytes": (
                0
                if encoder_workers is None
                else encoder_workers.estimated_peak_total_rss_bytes
            ),
            "api_rss_bytes": current_process_rss_bytes(),
            "peak_worker_rss_bytes": (
                0 if encoder_workers is None else encoder_workers.peak_worker_rss_bytes
            ),
            "qwen_enabled": False,
            "gemini_enabled": False,
            "optional_degraded_components": optional_degraded_components,
            "components": components,
        }
    with _health_cache_lock:
        _health_cache = (time.monotonic(), signature, snapshot)
    return snapshot


async def _require_component(component_name: str, message: str) -> None:
    dependency = await _dependency_health()
    component = (dependency.get("components") or {}).get(component_name) or {}
    if component.get("ready") is not True:
        raise HTTPException(
            status_code=503,
            detail={"message": message, "health": dependency},
        )


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return await _dependency_health()


@app.get("/api/branch1/health")
async def get_branch1_health() -> dict[str, Any]:
    if branch1_qdrant is None or branch1_encoders is None:
        return {"status": "starting", "ready": False, "fail_closed": True}
    return await _safe_health_call(
        branch1_health, DATA_ROOT, branch1_qdrant, branch1_encoders, STATE_ROOT
    )


@app.get("/api/branch2/health")
async def get_branch2_health() -> dict[str, Any]:
    if branch2_searcher is None:
        return {"status": "starting", "ready": False, "fail_closed": True}
    return await _safe_health_call(branch2_searcher.health)


@app.get("/api/branch3/asr/health")
async def get_branch3_asr_health() -> dict[str, Any]:
    if branch3_asr_searcher is None:
        return {"status": "starting", "ready": False, "fail_closed": True}
    return await _safe_health_call(branch3_asr_searcher.health)


@app.get("/api/branch3/ocr/health")
async def get_branch3_ocr_health() -> dict[str, Any]:
    # The dedicated OCR health route may perform the cached, non-blocking raw
    # source audit.  Overall health/config and search use the fast snapshot so
    # they never scan the 873 source JSONL files on their hot path.
    unavailable = {
        "status": "starting",
        "ready": False,
        "production_ready": False,
        "required": False,
        "branch": "branch3",
        "modality": "ocr",
        "fail_closed": True,
        "artifact_summary": {
            "source_total": 0,
            "source_verified": 0,
            "source_failed": 0,
            "hash_recomputed": 0,
        },
    }
    try:
        if branch3_ocr_searcher is None:
            return unavailable
        return await _safe_health_call(branch3_ocr_searcher.health, True)
    except Exception as error:
        return {**unavailable, "status": "not_ready", "error": str(error)}


@app.get("/api/fusion/kis/health")
async def get_kis_fusion_health() -> dict[str, Any]:
    # Reuse the same short-lived dependency snapshot as /api/health and
    # /api/config.  Calling the aggregate service directly here would issue
    # another complete Branch-1/2/Qdrant health pass for every browser poll
    # and could expose a different readiness generation from config.
    if kis_fusion_searcher is None:
        return {
            "schema_version": "kis.fusion.health.v1",
            "branch": "final_fusion",
            "task_type": "KIS",
            "status": "starting",
            "ready": False,
            "required": False,
            "production_ready": False,
            "fail_closed": True,
        }
    snapshot = await _dependency_health()
    fusion_state = snapshot.get("kis_fusion")
    if isinstance(fusion_state, dict):
        result = dict(fusion_state)
        result.setdefault("schema_version", "kis.fusion.health.v1")
        result.setdefault("branch", "final_fusion")
        result.setdefault("task_type", "KIS")
        result.setdefault("required", False)
        result.setdefault("ready", False)
        result.setdefault("production_ready", False)
        result.setdefault("status", "ready" if result["ready"] is True else "not_ready")
        result.setdefault("fail_closed", result["ready"] is not True)
        return result
    return {
        "schema_version": "kis.fusion.health.v1",
        "branch": "final_fusion",
        "task_type": "KIS",
        "status": "not_ready",
        "ready": False,
        "required": False,
        "production_ready": False,
        "fail_closed": True,
        "error": "KIS fusion health snapshot is malformed",
    }


def _kis_runtime_error_code(message: str) -> str:
    """Map canonical fusion phase failures to stable API diagnostics."""

    value = str(message or "")
    known_codes = (
        "KIS_FUSION_SEARCH_BUSY",
        "KIS_FUSION_NOT_READY",
        "KIS_FUSION_BRANCH_FAILED",
        "KIS_FUSION_RRF_FAILED",
        "KIS_FUSION_BEIT3_FAILED",
    )
    for code in known_codes:
        if value == code or value.startswith(f"{code}:"):
            return code
    return "KIS_FUSION_EXECUTION_FAILED"


@app.get("/api/config")
async def config() -> dict[str, Any]:
    dependency = await _dependency_health()
    raw_components = dependency.get("components")
    components = raw_components if isinstance(raw_components, dict) else {}

    def component_ready(name: str) -> bool:
        value = components.get(name)
        return isinstance(value, dict) and value.get("ready") is True

    branch1_ready = component_ready("branch1")
    branch2_ready = component_ready("branch2")
    branch3_asr_ready = component_ready("branch3_asr")
    branch3_ocr_ready = component_ready("branch3_ocr")
    kis_fusion_ready = component_ready("kis_fusion")
    image_search_ready = component_ready("image_search")
    siglip_text_ready = component_ready("siglip_text")
    metadata_ready = component_ready("metadata")
    # Keep config capabilities derived from the same health snapshot rather
    # than re-reading metadata independently after the snapshot was built.
    # This makes a manifest replacement visible consistently to health,
    # config, and the branch endpoints.
    timeline_ready = metadata_ready
    branch2_snapshot = components.get("branch2") or {}
    raw_branch2_components = (
        branch2_snapshot.get("components")
        if isinstance(branch2_snapshot, dict)
        else None
    )
    branch2_components = (
        raw_branch2_components if isinstance(raw_branch2_components, dict) else {}
    )
    # The legacy discovery endpoint still embeds object phrases with BGE-M3,
    # so advertising it with only a green DAM collection would create a
    # capability that immediately fails at request time.  Keep the explicit
    # collection/frame gates and include the text encoder it actually calls.
    dam_ready = isinstance(raw_branch2_components, dict) and all(
        isinstance(branch2_components.get(name), dict)
        and branch2_components[name].get("ready") is True
        for name in ("dam_collection", "bge_text_encoder", "frame_mapping")
    )
    return {
        "keyframes_root": str(KEYFRAMES_ROOT), "experiment_mode": "cpu_qdrant_workbench",
        "task_types": ["KIS"], "modalities": ["siglip", "dam", "ocr", "asr"],
        "default_top_k": 20, "max_top_k": 100, "dam_match_threshold": 0.50,
        "fusion_enabled": kis_fusion_ready,
        "reranking_enabled": kis_fusion_ready, "query_parser": "deterministic_local",
        "parser_engines": ["local", "direct"], "allow_qwen_fallback_default": False,
        "device": "cpu",
        "qwen_enabled": False, "gemini_enabled": False,
        "qwen_model_id": None, "gemini_model_id": None,
        "siglip_model_id": "google/siglip2-base-patch16-224",
        "siglip_revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        "bge_model_id": "BAAI/bge-m3",
        "capabilities": {
            "video_timeline": timeline_ready, "image_search": image_search_ready, "submission_prepare": True,
            "submission_modes": ["KIS", "VQA", "TRAKE"], "ordered_search_expansion": True,
            "discovery_cascade": siglip_text_ready and dam_ready, "media_info": MEDIA_INFO_ROOT.is_dir(),
            "ocr": branch3_ocr_ready,
            "asr": branch3_asr_ready,
            "parser_modes": ["local", "direct"], "qwen_fallback_control": False,
            "branch1_three_model": branch1_ready,
            "branch2_dam_hybrid": branch2_ready,
            "branch3_asr": branch3_asr_ready,
            "branch3_ocr": branch3_ocr_ready,
            "kis_fusion": kis_fusion_ready,
        },
        "image_search": {
            "enabled": image_search_ready, "max_upload_bytes": MAX_IMAGE_UPLOAD_BYTES,
            "max_pixels": MAX_IMAGE_PIXELS, "max_dimension": MAX_IMAGE_DIMENSION,
            "content_types": sorted(ALLOWED_IMAGE_CONTENT_TYPES), "max_top_k": 100,
        },
        "ordered_search": {
            "default_top_k_per_event": 300, "max_top_k_per_event": 1000,
            "default_paths_per_video": 1, "max_paths_per_video": 10,
            "max_sequence_reservoir_size": 500,
        },
        "submission": {
            "modes": ["KIS", "VQA", "TRAKE"], "default_target_rows": 100,
            "max_target_rows": 100, "canonical_validation": True,
            "fabricated_frames_allowed": False, "csv_has_header": False, "csv_encoding": "UTF-8",
            "csv_delimiter": ",", "csv_line_endings": ["CRLF", "LF"],
            "formats": {
                "KIS": ["video_id", "frame_idx"],
                "VQA": ["video_id", "frame_idx", "answer"],
                "TRAKE": ["video_id", "frame_id_1", "...", "frame_id_n"],
            },
            "vqa_answer_max_characters": 100,
            "trake_max_sequence_rows": 100,
            "trake_max_events_per_submission_row": 100,
            "trake_requires_complete_sequences": True,
            "trake_requires_same_video": True,
            "trake_requires_strict_time_order": True,
            "trake_canonical_neighbor_fill": True,
            "minimum_valid_rows": 1,
        },
    }


@app.post("/api/search/branch1")
async def search_branch1(request: Branch1SearchRequest) -> dict[str, Any]:
    if branch1_searcher is None or branch1_qdrant is None or branch1_encoders is None:
        raise HTTPException(status_code=503, detail="Branch-1 services are starting")
    readiness = await _safe_health_call(
        branch1_health, DATA_ROOT, branch1_qdrant, branch1_encoders, STATE_ROOT
    )
    if not readiness["ready"]:
        raise HTTPException(status_code=503, detail={"message": "Branch-1 is not ready", "health": readiness})
    try:
        return await run_in_threadpool(
            branch1_searcher.execute,
            request.query_bundle.model_dump(),
            request.model_weights.normalized(),
            request.per_stream_top_k,
            request.final_top_k,
        )
    except RuntimeError as error:
        if str(error) == "BRANCH1_SEARCH_BUSY":
            raise HTTPException(status_code=429, detail="Another Branch-1 search is already running") from error
        raise HTTPException(status_code=503, detail=f"Branch-1 execution failed: {error}") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Branch-1 execution failed: {error}") from error


@app.post("/api/search/branch2")
async def search_branch2(request: Branch2SearchRequest) -> dict[str, Any]:
    if branch2_searcher is None:
        raise HTTPException(status_code=503, detail="Branch-2 services are starting")
    readiness = await _safe_health_call(branch2_searcher.health)
    if not readiness["ready"]:
        raise HTTPException(status_code=503, detail={"message": "Branch-2 is not ready", "health": readiness})
    try:
        return await run_in_threadpool(
            branch2_searcher.execute,
            request.query_bundle.model_dump(),
            request.hybrid_weights.normalized(),
            request.rerank_weights.normalized(),
            request.per_stream_top_k,
            request.pre_rerank_top_k,
            request.rerank_top_k,
        )
    except RuntimeError as error:
        if str(error) == "BRANCH2_SEARCH_BUSY":
            raise HTTPException(status_code=429, detail="Another heavy retrieval search is already running") from error
        raise HTTPException(status_code=503, detail=f"Branch-2 execution failed: {error}") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Branch-2 execution failed: {error}") from error


@app.post("/api/search/branch3/asr")
async def search_branch3_asr(request: Branch3AsrSearchRequest) -> dict[str, Any]:
    if branch3_asr_searcher is None:
        raise HTTPException(status_code=503, detail="Branch-3 ASR service is starting")
    try:
        return await run_in_threadpool(
            branch3_asr_searcher.execute,
            request.query_bundle.model_dump(),
            request.per_stream_top_k,
            request.final_top_k,
        )
    except RuntimeError as error:
        if str(error) == "BRANCH3_ASR_SEARCH_BUSY":
            raise HTTPException(
                status_code=429,
                detail="Another heavy retrieval search is already running",
            ) from error
        raise HTTPException(status_code=503, detail=f"Branch-3 ASR execution failed: {error}") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Branch-3 ASR execution failed: {error}") from error


@app.post("/api/search/branch3/ocr")
async def search_branch3_ocr(request: Branch3OcrSearchRequest) -> dict[str, Any]:
    if branch3_ocr_searcher is None:
        raise HTTPException(status_code=503, detail="Branch-3 OCR service is starting")
    try:
        return await run_in_threadpool(
            branch3_ocr_searcher.execute,
            request.query_bundle.model_dump(),
            request.per_stream_top_k,
            request.final_top_k,
        )
    except RuntimeError as error:
        if str(error) == "BRANCH3_OCR_SEARCH_BUSY":
            raise HTTPException(
                status_code=429,
                detail="Another heavy retrieval search is already running",
            ) from error
        raise HTTPException(status_code=503, detail=f"Branch-3 OCR execution failed: {error}") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Branch-3 OCR execution failed: {error}") from error


@app.post("/api/search/fusion/kis")
async def search_kis_fusion(request: KisFusionSearchRequest) -> dict[str, Any]:
    """Run the fixed four-voter KIS fusion pipeline in one worker call."""

    if kis_fusion_searcher is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "KIS_FUSION_NOT_READY",
                "message": "KIS fusion service is starting",
            },
        )
    try:
        return await run_in_threadpool(
            kis_fusion_searcher.execute,
            request.query_bundle.model_dump(),
            request.branch_weights.normalized(),
        )
    except RuntimeError as error:
        message = str(error)
        error_code = _kis_runtime_error_code(message)
        if error_code == "KIS_FUSION_SEARCH_BUSY":
            raise HTTPException(
                status_code=429,
                detail={"code": "KIS_FUSION_SEARCH_BUSY", "message": "Another heavy KIS search is already running"},
            ) from error
        raise HTTPException(
            status_code=503,
            detail={"code": error_code, "message": message},
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "KIS_FUSION_EXECUTION_FAILED",
                "message": str(error),
            },
        ) from error


@app.post("/api/parse")
async def parse_query(request: ParseRequest) -> dict[str, Any]:
    if request.task_type != "KIS":
        raise HTTPException(status_code=400, detail="CPU-only mode supports KIS retrieval only")
    started = time.perf_counter()
    try:
        parsed = parser.parse(request.query, task_type="KIS")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {
        "parsed_query": parsed.model_dump(exclude={"weights"}),
        "parser_engine_requested": request.engine,
        "qwen_fallback_allowed": False,
        "external_llm_used": False,
        "execution_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


@app.post("/api/search")
async def search(request: SearchRequest) -> dict[str, Any]:
    if searcher is None:
        raise HTTPException(status_code=503, detail="CPU encoders are not ready")
    await _require_component("qdrant", "Qdrant is not ready")
    if request.parsed_query.task_type != "KIS":
        raise HTTPException(status_code=400, detail="CPU-only mode supports KIS retrieval only")
    if request.parsed_query.ocr_keywords:
        await _require_component("branch3_ocr", "OCR index is not prepared")
    started = time.perf_counter()
    try:
        pools = await run_in_threadpool(_execute_search, request.parsed_query, request.top_k)
    except HTTPException:
        raise
    except ValueError as error:
        # Invalid OCR/ASR lexical input is a contract error, not a missing
        # dependency.  Keep the legacy workbench endpoint aligned with the
        # dedicated Branch-3 routes and return 422 to the caller.
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        if str(error) in {"BRANCH3_ASR_SEARCH_BUSY", "BRANCH3_OCR_SEARCH_BUSY"}:
            raise HTTPException(
                status_code=429,
                detail="Another heavy retrieval search is already running",
            ) from error
        raise HTTPException(status_code=503, detail=f"Search dependencies failed: {error}") from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Search dependencies failed: {error}") from error
    session_id = request.session_id or request.parsed_query.session_id or str(uuid.uuid4())
    return {
        "session_id": session_id, "task_type": "KIS", "experiment_mode": "cpu_qdrant_workbench",
        "fusion_applied": False, "reranking_applied": False,
        "total_candidates": sum(item["result_count"] for item in pools.values()),
        "modality_results": pools, "execution_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


@app.post("/api/search/image")
async def search_image(
    file: Annotated[UploadFile, File()],
    top_k: Annotated[int, Form(ge=1, le=100)] = 50,
    video_id: Annotated[str | None, Form(max_length=64)] = None,
) -> dict[str, Any]:
    if searcher is None or encoders is None:
        raise HTTPException(status_code=503, detail="CPU encoders are not ready")
    await _require_component("image_search", "Image search dependencies are not ready")
    try:
        payload = await file.read(MAX_IMAGE_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    image, image_info = await run_in_threadpool(_decode_image, payload, file.content_type)
    canonical = _canonical_video_id(video_id) if video_id and video_id.strip() else None
    started = time.perf_counter()
    try:
        vector = await run_in_threadpool(encoders.embed_siglip_image, image)
        if canonical:
            evaluated = searcher.get_video_frame_count(canonical)
            if evaluated == 0:
                raise HTTPException(status_code=404, detail=f"Video {canonical} not found")
            results = await run_in_threadpool(searcher.search_visual_in_video, vector, canonical, top_k=top_k)
        else:
            results = await run_in_threadpool(searcher.search_visual, vector, top_k=top_k)
            evaluated = 247_956
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Image search failed: {error}") from error
    finally:
        image.close()
    elapsed = round((time.perf_counter() - started) * 1000.0, 2)
    result_pool = _pool("siglip", "SigLIP2 image similarity (CPU + Qdrant)", "<uploaded image>", "uploaded_image", "cosine", "Raw SigLIP2 image-to-image cosine", lambda: results)
    result_pool["provenance"].update({
        "query_tower": "siglip_image",
        "index_modality": "siglip_visual",
    })
    result_pool["execution_time_ms"] = elapsed
    return {
        "task_type": "KIS", "experiment_mode": "cpu_qdrant_workbench", "operation": "image_query",
        "query_modality": "image", "scope": "video" if canonical else "global", "video_id": canonical,
        "evaluated_frames": evaluated, "image_info": image_info, "fusion_applied": False,
        "reranking_applied": False, "modality_result": result_pool, "execution_time_ms": elapsed,
    }


@app.post("/api/video/{video_id}/search/siglip")
async def search_video(video_id: str, request: VideoVisualSearchRequest) -> dict[str, Any]:
    if searcher is None or workbench_search is None:
        raise HTTPException(status_code=503, detail="CPU encoders are not ready")
    await _require_component("siglip_text", "SigLIP text search dependencies are not ready")
    try:
        canonical = _canonical_video_id(video_id)
        evaluated = searcher.get_video_frame_count(canonical)
        if evaluated == 0:
            raise HTTPException(status_code=404, detail=f"Video {canonical} not found")
        started = time.perf_counter()
        pool = await run_in_threadpool(
            workbench_search.search_visual_in_video,
            request.parsed_query,
            video_id=canonical,
            top_k=request.top_k,
        )
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Video visual search failed: {error}") from error
    return {
        "experiment_mode": "cpu_qdrant_workbench", "operation": "manual_video_drilldown",
        "video_id": canonical, "scope_selected_by_user": True, "evaluated_frames": evaluated,
        "fusion_applied": False, "reranking_applied": False, "modality_result": pool,
        "execution_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


@app.post("/api/discover/dam-to-siglip")
async def discover(request: DiscoveryCascadeRequest) -> dict[str, Any]:
    if workbench_search is None:
        raise HTTPException(status_code=503, detail="CPU encoders are not ready")
    await _require_component("siglip_text", "SigLIP text search dependencies are not ready")
    await _require_component("qdrant", "Qdrant is not ready")
    try:
        return await run_in_threadpool(workbench_search.discover_dam_to_siglip, request.parsed_query, dam_top_frames_per_object=request.dam_top_frames_per_object, siglip_top_frames_per_video=request.siglip_top_frames_per_video)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"DAM discovery failed: {error}") from error


@app.post("/api/search/temporal-intersection")
async def temporal(request: TemporalIntersectionRequest) -> dict[str, Any]:
    if workbench_search is None:
        raise HTTPException(status_code=503, detail="CPU encoders are not ready")
    await _require_component("siglip_text", "SigLIP text search dependencies are not ready")
    await _require_component("qdrant", "Qdrant is not ready")
    try:
        return await run_in_threadpool(
            workbench_search.search_temporal_intersection,
            events=[event.model_dump() for event in request.events], anchor_query=request.anchor_query,
            top_k_per_event=request.top_k_per_event, top_k_sequences=request.top_k_sequences,
            max_gap_seconds=request.max_gap_seconds, anchor_event_order=request.anchor_event_order,
            paths_per_video=request.paths_per_video, sequence_reservoir_size=request.sequence_reservoir_size,
            path_beam_width=request.path_beam_width, path_diversity_min_events=request.path_diversity_min_events,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Temporal search failed: {error}") from error


@app.post("/api/submission/prepare")
async def submission(request: SubmissionPrepareRequest) -> dict[str, Any]:
    if metadata_store is None:
        raise HTTPException(status_code=503, detail="Metadata is not ready")
    try:
        result = await run_in_threadpool(
            prepare_submission, _normalize_submission(request.model_dump()),
            frame_lookup=metadata_store.frame_by_idx, video_frames_lookup=metadata_store.video_frames,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Submission preparation failed: {error}") from error
    result["server_verified"] = True
    return result


@app.get("/api/keyframe/{video_id}/{keyframe_n}")
async def keyframe_detail(video_id: str, keyframe_n: int) -> dict[str, Any]:
    if metadata_store is None or searcher is None:
        raise HTTPException(status_code=503, detail="Metadata is not ready")
    await _require_component("qdrant", "Qdrant is not ready")
    canonical = _canonical_video_id(video_id)
    detail = await run_in_threadpool(metadata_store.detail, canonical, keyframe_n)
    if detail is None:
        raise HTTPException(status_code=404, detail="Keyframe not found")
    try:
        dam_objects = await run_in_threadpool(
            searcher.qdrant.dam_for_frame, canonical, int(detail["frame_idx"])
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"DAM detail is unavailable: {error}") from error
    macro_audio_transcript = ""
    if asr_index is not None:
        macro_audio_transcript = await run_in_threadpool(
            asr_index.audio_span, canonical, int(detail["frame_idx"])
        )
    return {
        "keyframe": detail,
        "dam_objects": dam_objects,
        "macro_audio_transcript": macro_audio_transcript,
    }


@app.get("/api/video/{video_id}/keyframes")
async def video_keyframes(video_id: str) -> dict[str, Any]:
    if metadata_store is None:
        raise HTTPException(status_code=503, detail="Metadata is not ready")
    canonical = _canonical_video_id(video_id)
    frames = await run_in_threadpool(metadata_store.video_frames, canonical)
    if not frames:
        raise HTTPException(status_code=404, detail="Video not found")
    return {"video_id": canonical, "total_keyframes": len(frames), "keyframes": frames}


@app.get("/api/video/{video_id}/timeline")
async def video_timeline(video_id: str) -> dict[str, Any]:
    if metadata_store is None:
        raise HTTPException(status_code=503, detail="Metadata is not ready")
    timeline = await run_in_threadpool(metadata_store.timeline, _canonical_video_id(video_id))
    if timeline is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return timeline


@app.get("/api/video/{video_id}/media-info")
async def media_info(video_id: str) -> dict[str, Any]:
    canonical = _canonical_video_id(video_id)
    path = MEDIA_INFO_ROOT / f"{canonical}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Media info for {canonical} not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=500, detail="Media info could not be read") from error


@app.get("/keyframes/{video_id}/{filename}")
async def keyframe_image(video_id: str, filename: str):
    canonical = _canonical_video_id(video_id)
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid keyframe path")
    target = KEYFRAMES_ROOT / canonical / filename
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Keyframe image not found")
    return FileResponse(target)


if (FRONTEND_DIR / "index.html").is_file():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
