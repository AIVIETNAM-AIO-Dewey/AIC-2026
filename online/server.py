"""FastAPI server for the auditable KIS no-fusion retrieval experiment.

The server keeps SigLIP-2 and BGE-M3 query encoders warm, searches DAM,
SigLIP, OCR, and ASR independently, and returns four isolated result pools.
It intentionally does not initialize or execute fusion, synergy, reranking,
VQA reasoning, or TRAKE sequence solving.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from online.src.contracts.query import ParsedQuery, TaskType
from online.src.retrieval.embeddings import ModelRegistry
from online.src.retrieval.modality_search import IndependentModalitySearch
from online.src.retrieval.query_parser import QueryParser
from online.src.retrieval.vector_search import FastVectorSearchEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aic_server")

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "server_config.yaml"


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
    _modality_search = IndependentModalitySearch(searcher=_searcher, registry=_registry)

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
    engine: str = "gemini"  # "gemini" or "qwen"


class SearchRequest(BaseModel):
    parsed_query: ParsedQuery
    session_id: str | None = None
    top_k: int = Field(default=20, ge=1, le=100, description="Results per modality")


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
        "fusion_enabled": False,
        "reranking_enabled": False,
        "gemini_model_id": config["models"].get("gemini_model_id", "gemini-3.6-flash"),
        "qwen_model_id": config["models"].get("qwen_model_id", "qwen2.5:7b"),
        "siglip_model_id": config["models"]["siglip"],
        "siglip_revision": config["models"]["siglip_revision"],
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
    parsed = _parser.parse(req.query, task_type=req.task_type, engine=req.engine)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    return {
        # Weights are intentionally omitted: no modality can gate or influence another.
        "parsed_query": parsed.model_dump(exclude={"weights"}),
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
