"""FastAPI server for the auditable KIS no-fusion retrieval experiment.

The server keeps SigLIP-2 and BGE-M3 query encoders warm, searches DAM,
SigLIP, OCR, and ASR independently, and returns four isolated result pools.
It intentionally does not initialize or execute fusion, synergy, reranking,
VQA reasoning, or TRAKE sequence solving.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator, model_validator

from online.src.contracts.query import ParsedQuery, TaskType
from online.src.retrieval.embeddings import ModelRegistry
from online.src.retrieval.modality_search import IndependentModalitySearch
from online.src.retrieval.query_parser import QueryParser
from online.src.retrieval.vector_search import FastVectorSearchEngine
from online.src.submission import prepare_submission

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aic_server")

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "server_config.yaml"
MAX_IMAGE_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_IMAGE_DIMENSION = 8192
ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
ALLOWED_IMAGE_FORMATS = frozenset(ALLOWED_IMAGE_CONTENT_TYPES.values())


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()

# Global Warm Singletons
_searcher: FastVectorSearchEngine | None = None
_registry: ModelRegistry | None = None
_modality_search: IndependentModalitySearch | None = None
_parser: QueryParser | None = None

# In-memory video ID -> actual directory path map (supports nested keyframes-1, keyframes-2, etc.)
_video_to_dir_map: dict[str, Path] = {}


def _canonical_video_id(video_id: str) -> str:
    """Normalize a user-facing video ID without accepting path-like values."""
    canonical_id = video_id.strip().upper().replace("-", "_")
    if not re.fullmatch(r"[A-Z0-9_]+", canonical_id):
        raise HTTPException(status_code=400, detail="Invalid video ID")
    return canonical_id


def _decode_uploaded_image(
    payload: bytes,
    content_type: str | None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Validate and decode one bounded, non-animated raster image."""
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    if len(payload) > MAX_IMAGE_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded image exceeds the {MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    declared_type = (content_type or "").split(";", 1)[0].strip().lower()
    if declared_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG, PNG, and WebP images are supported",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as source:
                actual_format = str(source.format or "").upper()
                if actual_format not in ALLOWED_IMAGE_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail="Only JPEG, PNG, and WebP images are supported",
                    )
                if actual_format != ALLOWED_IMAGE_CONTENT_TYPES[declared_type]:
                    raise HTTPException(
                        status_code=415,
                        detail="Uploaded image content does not match its media type",
                    )
                if bool(getattr(source, "is_animated", False)) or int(
                    getattr(source, "n_frames", 1)
                ) != 1:
                    raise HTTPException(
                        status_code=415,
                        detail="Animated images are not supported",
                    )

                width, height = source.size
                if width < 1 or height < 1:
                    raise HTTPException(status_code=400, detail="Uploaded image has invalid dimensions")
                if (
                    width > MAX_IMAGE_DIMENSION
                    or height > MAX_IMAGE_DIMENSION
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Uploaded image dimensions exceed the "
                            f"{MAX_IMAGE_DIMENSION}px / {MAX_IMAGE_PIXELS:,}-pixel limit"
                        ),
                    )

                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB").copy()
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise HTTPException(status_code=413, detail="Uploaded image is too large") from error
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Uploaded image could not be decoded") from error

    return image, {
        "format": actual_format.lower(),
        "width": int(image.width),
        "height": int(image.height),
        "byte_count": len(payload),
    }


def _normalize_submission_frame(value: Any) -> Any:
    """Canonicalize client video IDs while retaining auditable provenance."""
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    if "video_id" in normalized:
        normalized["video_id"] = _canonical_video_id(str(normalized["video_id"]))
    return normalized


def _normalize_submission_sequence(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_submission_frame(frame) for frame in value]
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    for field_name in ("matched_events", "events", "frames"):
        if isinstance(normalized.get(field_name), list):
            normalized[field_name] = [
                _normalize_submission_frame(frame) for frame in normalized[field_name]
            ]
    return normalized


def _normalize_submission_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    for field_name in ("manual_selections", "candidate_reservoir"):
        normalized[field_name] = [
            _normalize_submission_frame(frame) for frame in normalized.get(field_name, [])
        ]
    for field_name in ("manual_sequences", "candidate_sequences"):
        normalized[field_name] = [
            _normalize_submission_sequence(sequence)
            for sequence in normalized.get(field_name, [])
        ]
    return normalized


def _index_keyframe_directories(root_dir: Path):
    """Index video keyframe directories at any nesting depth and casing."""
    if not root_dir.exists():
        logger.warning(f"Keyframes root directory not found: {root_dir}")
        return
    logger.info(f"⚡ Discovering video keyframe directories under {root_dir}...")
    t0 = time.perf_counter()
    try:
        for p in root_dir.rglob("*"):
            if p.is_dir():
                name_upper = p.name.upper()
                if name_upper.startswith("L") and ("_" in name_upper or "-" in name_upper):
                    canon_name = name_upper.replace("-", "_")
                    _video_to_dir_map[canon_name] = p
                    _video_to_dir_map[p.name] = p
    except Exception as e:
        logger.warning(f"Error during keyframe directory indexing: {e}")

    dt = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "✅ Indexed %d video keyframe folders in %.1fms",
        len(_video_to_dir_map),
        dt,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _searcher, _registry, _modality_search, _parser

    logger.info("🚀 Starting AIC Retrieval Engine Server...")
    t0 = time.perf_counter()

    # 1. Discover all keyframe directories (supports keyframes, keyframes-2, keyframes-3, etc.)
    _index_keyframe_directories(KEYFRAMES_DIR)

    # 2. Load memory-mapped vector search matrices & metadata
    idx_path = config["paths"]["unified_index"]
    _searcher = FastVectorSearchEngine(
        unified_index_dir=idx_path,
        block_rows=config["retrieval"].get("search_block_rows", 32_768),
    )

    # 3. Warm only the two query encoders used by the independent pools.
    _registry = ModelRegistry.get_instance(
        siglip_id=config["models"]["siglip"],
        siglip_revision=config["models"]["siglip_revision"],
        bge_id=config["models"]["bge_m3"],
        reranker_id=config["models"]["bge_reranker"],
    )
    logger.info("⚡ Pre-warming PyTorch GPU models...")
    _registry._load_siglip()
    _registry._load_bge()

    # 4. Instantiate the no-fusion modality orchestrator.
    _modality_search = IndependentModalitySearch(
        searcher=_searcher,
        registry=_registry,
        dam_match_threshold=config["retrieval"].get("dam_match_threshold", 0.50),
    )

    # 5. Query parser with Gemini & Ollama Qwen
    _parser = QueryParser(
        gemini_model_id=config["models"].get("gemini_model_id", "gemini-3.6-flash"),
        qwen_model_id=config["models"].get("qwen_model_id", "qwen2.5:7b"),
        ollama_url=config["models"].get("qwen_ollama_url", "http://localhost:11434/api/chat"),
    )

    dt = time.perf_counter() - t0
    logger.info(f"✅ Server fully warmed and ready in {dt:.1f}s!")
    yield
    logger.info("🛑 Shutting down AIC Retrieval Server...")


app = FastAPI(title="AIC-2026 Online Retrieval Server", lifespan=lifespan)

# Enable CORS for local browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ──────────────────────────────────────────────────────────────────────────────
class ParseRequest(BaseModel):
    query: str
    task_type: TaskType | None = "KIS"
    engine: Literal["gemini", "qwen", "rule", "direct"] = "gemini"
    # Preserve the original Gemini -> Qwen -> rule chain unless the UI/user
    # explicitly disables Qwen for auditable direct/local operation.
    allow_qwen_fallback: bool = True


class SearchRequest(BaseModel):
    parsed_query: ParsedQuery
    session_id: str | None = None
    top_k: int = Field(default=20, ge=1, le=100, description="Results per modality")


class VideoVisualSearchRequest(BaseModel):
    parsed_query: ParsedQuery
    top_k: int = Field(default=50, ge=1, le=100, description="SigLIP results in video")


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
    def validate_event_orders(self):
        orders = [event.order for event in self.events]
        if len(set(orders)) != len(orders):
            raise ValueError("event orders must be unique")
        if self.anchor_event_order is not None and self.anchor_event_order not in set(orders):
            raise ValueError("anchor_event_order must identify one of the supplied events")
        if self.anchor_query is not None:
            self.anchor_query = self.anchor_query.strip() or None
        if (
            self.sequence_reservoir_size is not None
            and self.sequence_reservoir_size < self.top_k_sequences
        ):
            raise ValueError("sequence_reservoir_size cannot be smaller than top_k_sequences")
        if self.path_diversity_min_events > len(self.events):
            raise ValueError("path_diversity_min_events cannot exceed the event count")
        return self


class SubmissionPrepareRequest(BaseModel):
    """Compact task-aware draft sent by the persistent submission workspace."""

    task_type: Literal["KIS", "VQA", "QA", "TRAKE"] = "KIS"
    mode: str | None = Field(default=None, max_length=64)
    query_id: str = Field(default="1", min_length=1, max_length=64)
    target_rows: int = Field(default=100, ge=1, le=100)
    manual_selections: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    candidate_reservoir: list[dict[str, Any]] = Field(default_factory=list, max_length=5000)
    vqa_answer: str | None = Field(default=None, max_length=100)
    # This validation/export endpoint is intentionally broader than the live
    # ordered-retrieval endpoint (currently capped at six events for compute).
    event_count: int | None = Field(default=None, ge=2, le=100)
    manual_sequences: list[Any] = Field(default_factory=list, max_length=100)
    candidate_sequences: list[Any] = Field(default_factory=list, max_length=500)

    @field_validator("query_id")
    @classmethod
    def validate_submission_query_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", cleaned):
            raise ValueError("query_id may contain only letters, numbers, underscores, and hyphens")
        return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/config")
async def get_client_config():
    """Return paths and default settings for the frontend."""
    return {
        "keyframes_root": config["paths"]["keyframes_root"],
        "experiment_mode": "nofusion",
        "task_types": ["KIS"],
        "modalities": ["siglip", "dam", "ocr", "asr"],
        "default_top_k": config["retrieval"]["default_top_k"],
        "max_top_k": config["retrieval"]["max_top_k"],
        "dam_match_threshold": config["retrieval"].get("dam_match_threshold", 0.50),
        "fusion_enabled": False,
        "reranking_enabled": False,
        "gemini_model_id": config["models"].get("gemini_model_id", "gemini-3.6-flash"),
        "qwen_model_id": config["models"].get("qwen_model_id", "qwen2.5:7b"),
        "siglip_model_id": config["models"]["siglip"],
        "siglip_revision": config["models"]["siglip_revision"],
        "parser_engines": ["gemini", "qwen", "rule", "direct"],
        "allow_qwen_fallback_default": True,
        "capabilities": {
            "video_timeline": True,
            "image_search": True,
            "submission_prepare": True,
            "submission_modes": ["KIS", "VQA", "TRAKE"],
            "ordered_search_expansion": True,
            "parser_modes": ["gemini", "qwen", "rule", "direct"],
            "qwen_fallback_control": True,
        },
        "image_search": {
            "enabled": True,
            "max_upload_bytes": MAX_IMAGE_UPLOAD_BYTES,
            "max_pixels": MAX_IMAGE_PIXELS,
            "max_dimension": MAX_IMAGE_DIMENSION,
            "content_types": sorted(ALLOWED_IMAGE_CONTENT_TYPES),
            "max_top_k": 100,
        },
        "ordered_search": {
            "default_top_k_per_event": 300,
            "max_top_k_per_event": 1000,
            "default_paths_per_video": 1,
            "max_paths_per_video": 10,
            "max_sequence_reservoir_size": 500,
        },
        "submission": {
            "modes": ["KIS", "VQA", "TRAKE"],
            "default_target_rows": 100,
            "max_target_rows": 100,
            "minimum_valid_rows": 1,
            "canonical_validation": True,
            "fabricated_frames_allowed": False,
            "csv_has_header": False,
            "csv_encoding": "UTF-8",
            "csv_delimiter": ",",
            "csv_line_endings": ["CRLF", "LF"],
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
        },
    }


@app.post("/api/parse")
async def parse_query_endpoint(req: ParseRequest):
    """Parse raw user query using Gemini or local Qwen with graceful fallbacks."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if req.task_type != "KIS":
        raise HTTPException(
            status_code=400,
            detail="The no-fusion experiment supports KIS queries only.",
        )

    t0 = time.perf_counter()
    # Keep the public experiment KIS-only, but let the parser use its TRAKE
    # decomposition prompt when a KIS description contains an ordered sequence.
    # The resulting event captions feed the explicit SigLIP intersection view;
    # no legacy TRAKE fusion or reranker is enabled.
    parser_task_type: TaskType = (
        "TRAKE" if _parser._detect_task_type(req.query) == "TRAKE" else "KIS"
    )
    parsed = _parser.parse(
        req.query,
        task_type=parser_task_type,
        engine=req.engine,
        allow_qwen_fallback=req.allow_qwen_fallback,
    )
    parsed.task_type = "KIS"
    dt_ms = (time.perf_counter() - t0) * 1000.0

    return {
        # Weights are intentionally omitted: no modality can gate or influence another.
        "parsed_query": parsed.model_dump(exclude={"weights"}),
        "parser_engine_requested": req.engine,
        "qwen_fallback_allowed": req.allow_qwen_fallback,
        "execution_time_ms": round(dt_ms, 2),
    }


@app.post("/api/search")
async def search_endpoint(req: SearchRequest):
    """Return four independently ranked result pools for one KIS query."""
    t0 = time.perf_counter()
    parsed = req.parsed_query
    if parsed.task_type != "KIS":
        raise HTTPException(
            status_code=400,
            detail="The no-fusion experiment supports KIS queries only.",
        )
    session_id = req.session_id or parsed.session_id or str(uuid.uuid4())
    parsed.session_id = session_id
    if _modality_search is None:
        raise HTTPException(status_code=503, detail="Search models are not ready yet.")
    pools = _modality_search.search(parsed, top_k=req.top_k)

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "session_id": session_id,
        "task_type": "KIS",
        "experiment_mode": "nofusion",
        "fusion_applied": False,
        "reranking_applied": False,
        "total_candidates": sum(pool["result_count"] for pool in pools.values()),
        "modality_results": pools,
        "execution_time_ms": round(dt_ms, 2),
    }


@app.post("/api/search/image")
async def search_by_uploaded_image(
    file: Annotated[UploadFile, File()],
    top_k: Annotated[int, Form(ge=1, le=100)] = 50,
    video_id: Annotated[str | None, Form(max_length=64)] = None,
):
    """Search the existing SigLIP visual index with an uploaded image."""
    if _searcher is None or _registry is None:
        raise HTTPException(status_code=503, detail="Search models are not ready yet.")

    try:
        payload = await file.read(MAX_IMAGE_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    # Pillow may spend noticeable CPU time decoding a valid high-resolution
    # upload. Keep that work off the async event loop just like model inference
    # and the full-index cosine scan below.
    image, image_info = await asyncio.to_thread(
        _decode_uploaded_image,
        payload,
        file.content_type,
    )

    canonical_id: str | None = None
    if video_id is not None and video_id.strip():
        canonical_id = _canonical_video_id(video_id)
        evaluated_frames = _searcher.get_video_frame_count(canonical_id)
        if evaluated_frames == 0:
            image.close()
            raise HTTPException(status_code=404, detail=f"Video {canonical_id} not found")
    else:
        evaluated_frames = _searcher.get_total_frame_count()

    def run_query() -> tuple[list[dict[str, Any]], float, float]:
        embedding_started = time.perf_counter()
        query_vector = _registry.embed_siglip_image(image)
        embedding_ms = (time.perf_counter() - embedding_started) * 1000.0

        search_started = time.perf_counter()
        if canonical_id is None:
            results = _searcher.search_visual(query_vector, top_k=top_k)
        else:
            results = _searcher.search_visual_in_video(
                query_vector,
                canonical_id,
                top_k=top_k,
            )
        search_ms = (time.perf_counter() - search_started) * 1000.0
        return results, embedding_ms, search_ms

    started = time.perf_counter()
    try:
        # Keep accelerator inference and the memory-mapped cosine scan off the
        # async event loop. The model registry serializes SigLIP text/image
        # inference with one shared lock.
        results, embedding_ms, search_ms = await asyncio.to_thread(run_query)
    except (RuntimeError, ValueError) as error:
        logger.exception("Uploaded-image SigLIP search failed")
        raise HTTPException(status_code=500, detail="Uploaded-image search failed") from error
    finally:
        image.close()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    scope = "video" if canonical_id is not None else "global"
    result_pool = {
        "modality": "siglip",
        "display_name": "SigLIP image similarity",
        "status": "ok",
        "reason": "",
        "query": "<uploaded image>",
        "query_source": "uploaded_image",
        "score_type": "cosine",
        "score_description": (
            "Raw cosine between the uploaded-image SigLIP vector and "
            f"{'frames in the selected video' if canonical_id else 'full-frame images'}"
        ),
        "result_count": len(results),
        "execution_time_ms": round(embedding_ms + search_ms, 2),
        "embedding_time_ms": round(embedding_ms, 2),
        "search_time_ms": round(search_ms, 2),
        "results": results,
        "provenance": {
            "query_modality": "uploaded_image",
            "query_tower": "siglip_image",
            "index_modality": "siglip_visual",
            "embedding_model_id": config["models"]["siglip"],
            "embedding_model_revision": config["models"]["siglip_revision"],
            "fusion_applied": False,
        },
    }
    return {
        "task_type": "KIS",
        "experiment_mode": "nofusion",
        "operation": "image_query",
        "query_modality": "image",
        "scope": scope,
        "video_id": canonical_id,
        "evaluated_frames": evaluated_frames,
        "image_info": image_info,
        "fusion_applied": False,
        "reranking_applied": False,
        "modality_result": result_pool,
        "execution_time_ms": round(elapsed_ms, 2),
    }


@app.post("/api/video/{video_id}/search/siglip")
async def search_siglip_inside_video(video_id: str, req: VideoVisualSearchRequest):
    """Manually scope the unchanged SigLIP cosine ranking to one video."""
    canonical_id = video_id.upper().replace("-", "_")
    if not re.fullmatch(r"[A-Z0-9_]+", canonical_id):
        raise HTTPException(status_code=400, detail="Invalid video ID")
    if req.parsed_query.task_type != "KIS":
        raise HTTPException(
            status_code=400,
            detail="The no-fusion experiment supports KIS queries only.",
        )
    if not req.parsed_query.global_scene_en.strip():
        raise HTTPException(
            status_code=400,
            detail="global_scene_en is required for a SigLIP video drill-down.",
        )
    if _searcher is None or _modality_search is None:
        raise HTTPException(status_code=503, detail="Search models are not ready yet.")

    frame_count = _searcher.get_video_frame_count(canonical_id)
    if frame_count == 0:
        raise HTTPException(status_code=404, detail=f"Video {canonical_id} not found")

    started = time.perf_counter()
    pool = _modality_search.search_visual_in_video(
        req.parsed_query,
        video_id=canonical_id,
        top_k=req.top_k,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "experiment_mode": "nofusion",
        "operation": "manual_video_drilldown",
        "video_id": canonical_id,
        "scope_selected_by_user": True,
        "evaluated_frames": frame_count,
        "fusion_applied": False,
        "reranking_applied": False,
        "modality_result": pool,
        "execution_time_ms": round(elapsed_ms, 2),
    }


@app.post("/api/discover/dam-to-siglip")
async def discover_dam_to_siglip(req: DiscoveryCascadeRequest):
    """Discover candidate videos per DAM object, then rank their frames by SigLIP."""
    parsed = req.parsed_query
    if parsed.task_type != "KIS":
        raise HTTPException(
            status_code=400,
            detail="The no-fusion experiment supports KIS queries only.",
        )
    if not parsed.global_scene_en.strip():
        raise HTTPException(
            status_code=400,
            detail="global_scene_en is required for the discovery cascade.",
        )
    if not any(query.strip() for query in parsed.objects_en):
        raise HTTPException(
            status_code=400,
            detail="At least one objects_en query is required for the discovery cascade.",
        )
    if _modality_search is None:
        raise HTTPException(status_code=503, detail="Search models are not ready yet.")

    return _modality_search.discover_dam_to_siglip(
        parsed,
        dam_top_frames_per_object=req.dam_top_frames_per_object,
        siglip_top_frames_per_video=req.siglip_top_frames_per_video,
    )


@app.post("/api/search/temporal-intersection")
async def search_temporal_intersection(req: TemporalIntersectionRequest):
    """Find same-video SigLIP matches that occur in the requested event order."""
    if _modality_search is None:
        raise HTTPException(status_code=503, detail="Search models are not ready yet.")

    return _modality_search.search_temporal_intersection(
        events=[event.model_dump() for event in req.events],
        anchor_query=req.anchor_query,
        top_k_per_event=req.top_k_per_event,
        top_k_sequences=req.top_k_sequences,
        max_gap_seconds=req.max_gap_seconds,
        anchor_event_order=req.anchor_event_order,
        paths_per_video=req.paths_per_video,
        sequence_reservoir_size=req.sequence_reservoir_size,
        path_beam_width=req.path_beam_width,
        path_diversity_min_events=req.path_diversity_min_events,
    )


@app.post("/api/submission/prepare")
async def prepare_submission_endpoint(req: SubmissionPrepareRequest):
    """Validate, complete, and preview one canonical submission draft."""
    if _searcher is None:
        raise HTTPException(status_code=503, detail="Search index is not ready yet.")

    payload = _normalize_submission_payload(req.model_dump())

    def canonical_frame_lookup(video_id: str, frame_idx: int):
        return _searcher.frame_lookup.get((video_id, frame_idx))

    result = prepare_submission(
        payload,
        frame_lookup=canonical_frame_lookup,
        video_frames_lookup=_searcher.get_video_keyframe_list,
    )
    result["server_verified"] = True
    return result


@app.get("/api/keyframe/{video_id}/{keyframe_n}")
async def get_keyframe_detail(video_id: str, keyframe_n: int):
    """Fetch complete metadata, DAM bounding boxes, and surrounding ASR speech."""
    kf = _searcher.get_keyframe_by_video_and_n(video_id, keyframe_n)
    if not kf:
        raise HTTPException(status_code=404, detail=f"Keyframe {video_id}:{keyframe_n} not found")

    dam_objects = _searcher.get_dam_objects_for_frame(video_id, kf["frame_idx"])
    audio_span = _searcher.get_video_audio_span(
        video_id,
        max(0, kf["frame_idx"] - 450),
        kf["frame_idx"] + 450,
    )

    return {
        "keyframe": kf,
        "dam_objects": dam_objects,
        "macro_audio_transcript": audio_span,
    }


@app.get("/api/video/{video_id}/keyframes")
async def get_video_keyframes(video_id: str):
    """Return all keyframes in a video for filmstrip slider navigation."""
    kfs = _searcher.get_video_keyframe_list(video_id)
    if not kfs:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    return {"video_id": video_id, "total_keyframes": len(kfs), "keyframes": kfs}


@app.get("/api/video/{video_id}/timeline")
async def get_video_timeline(video_id: str):
    """Return a lightweight canonical keyframe timeline for video playback."""
    if _searcher is None:
        raise HTTPException(status_code=503, detail="Search index is not ready yet.")
    canonical_id = _canonical_video_id(video_id)
    timeline = _searcher.get_video_timeline(canonical_id)
    if timeline is None:
        raise HTTPException(status_code=404, detail=f"Video {canonical_id} not found")
    return timeline


@app.get("/api/video/{video_id}/media-info")
async def get_video_media_info(video_id: str):
    """Return the YouTube mapping metadata for a video without loading retrieval models."""
    canonical_id = video_id.upper().replace("-", "_")
    if not re.fullmatch(r"[A-Z0-9_]+", canonical_id):
        raise HTTPException(status_code=400, detail="Invalid video ID")

    media_info_dir = (
        Path(str(config["paths"]["media_info"]).strip().strip('"').strip("'"))
        .expanduser()
        .resolve()
    )
    media_info_path = media_info_dir / f"{canonical_id}.json"
    if not media_info_path.is_file():
        raise HTTPException(status_code=404, detail=f"Media info for {canonical_id} not found")

    try:
        with media_info_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Could not read media info %s: %s", media_info_path, error)
        raise HTTPException(status_code=500, detail="Media info could not be read") from error


# ──────────────────────────────────────────────────────────────────────────────
# Static File & UI Serving
# ──────────────────────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend" / "dist"
KEYFRAMES_DIR = (
    Path(str(config["paths"]["keyframes_root"]).strip().strip('"').strip("'"))
    .expanduser()
    .resolve()
)


@app.get("/keyframes/{video_id}/{filename}")
async def serve_keyframe_image(video_id: str, filename: str):
    """Serve a keyframe across nested folders, casings, and image extensions."""
    canon_vid = video_id.upper().replace("-", "_")
    v_dir = _video_to_dir_map.get(canon_vid) or _video_to_dir_map.get(video_id)

    stem = Path(filename).stem
    candidates_to_try = [
        filename,
        f"{stem}.jpg",
        f"{stem}.jpeg",
        f"{stem}.png",
        f"{stem}.JPG",
        f"{stem}.JPEG",
        f"{stem}.PNG",
    ]
    if stem.isdigit():
        num = int(stem)
        candidates_to_try.extend(
            [
                f"{num:08d}.jpg",
                f"{num:08d}.jpeg",
                f"{num:08d}.png",
                f"{num:03d}.jpg",
                f"{num:04d}.jpg",
                f"{num}.jpg",
                f"{num:03d}.png",
                f"{num}.png",
            ]
        )

    # 1. Look up in indexed directory map
    if v_dir:
        for fname in candidates_to_try:
            target = v_dir / fname
            if target.exists():
                return FileResponse(target)

    # 2. Fallback direct check
    for fname in candidates_to_try:
        direct = KEYFRAMES_DIR / video_id / fname
        if direct.exists():
            return FileResponse(direct)

    # 3. Dynamic glob search fallback
    for fname in candidates_to_try:
        for match in KEYFRAMES_DIR.rglob(f"{video_id}/{fname}"):
            if match.exists():
                _video_to_dir_map[canon_vid] = match.parent
                return FileResponse(match)

    raise HTTPException(status_code=404, detail=f"Keyframe image {video_id}/{filename} not found")


# Serve only Vite's compiled output. The source tree contains TypeScript and
# bare module imports that browsers cannot consume through StaticFiles.
if (FRONTEND_DIR / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning(
        "Compiled frontend not found at %s. Run `npm run build` before starting the UI.",
        FRONTEND_DIR,
    )


def main():
    import uvicorn

    host = os.environ.get("AIC_HOST") or config["server"].get("host", "127.0.0.1")
    port = int(os.environ.get("AIC_PORT") or config["server"].get("port", 8890))
    logger.info(f"Starting server on http://{host}:{port}")
    uvicorn.run("online.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
