"""FastAPI High-Speed GPU Server for AIC Online Retrieval Engine.

Maintains warm GPU models in memory:
- SigLIP-2 Text Encoder (768-d)
- BGE-M3 Dense Text Encoder (1024-d)
- BGE-Reranker-v2-m3 Cross-Encoder
- BLAS Memory-Mapped Vector Search Engine (177k keyframes, 435k DAM objects)

Provides low-latency endpoints for:
1. LLM Query Parsing (Gemini 3.6 / Qwen 2.5 with automatic fallback)
2. 4-Channel Branch Search & Caching
3. Instant (< 5ms) CPU-based RRF Re-Fusion upon Weight Adjustment
4. Precision Stage 2 Cross-Encoder & VQA / TRAKE reasoning
5. Keyframe metadata, DAM bounding box overlays, and filmstrip navigation
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from online.src.contracts.query import ParsedQuery, TaskType
from online.src.retrieval.embeddings import ModelRegistry
from online.src.retrieval.fusion import MultimodalFusionEngine
from online.src.retrieval.query_parser import QueryParser
from online.src.retrieval.stage2_reranker import Stage2Reranker
from online.src.retrieval.vector_search import FastVectorSearchEngine
from online.src.retrieval.vqa_reasoner import VQAReasoner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aic_server")

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "server_config.yaml"


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()

# Global Warm Singletons
_searcher: Optional[FastVectorSearchEngine] = None
_registry: Optional[ModelRegistry] = None
_fusion: Optional[MultimodalFusionEngine] = None
_reranker: Optional[Stage2Reranker] = None
_parser: Optional[QueryParser] = None

# In-memory video ID -> actual directory path map (supports nested keyframes-1, keyframes-2, etc.)
_video_to_dir_map: dict[str, Path] = {}
# In-memory Branch Cache: session_id -> { "parsed_query": ..., "branches": dict, "created_at": float }
_branch_cache: dict[str, dict[str, Any]] = {}


def _index_keyframe_directories(root_dir: Path):
    """Recursively discover and index all video keyframe directories at any nesting depth."""
    if not root_dir.exists():
        return
    logger.info(f"⚡ Discovering video keyframe directories under {root_dir}...")
    t0 = time.perf_counter()
    try:
        for p in root_dir.rglob("L*_*"):
            if p.is_dir() and p.name.startswith("L") and "_" in p.name:
                _video_to_dir_map[p.name] = p
    except Exception as e:
        logger.warning(f"Error during keyframe directory indexing: {e}")

    dt = (time.perf_counter() - t0) * 1000.0
    logger.info(f"✅ Indexed {len(_video_to_dir_map)} video keyframe folders in {dt:.1f}ms (Supports arbitrary nested folders)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _searcher, _registry, _fusion, _reranker, _parser

    logger.info("🚀 Starting AIC Retrieval Engine Server...")
    t0 = time.perf_counter()

    # 1. Discover all keyframe directories (supports keyframes, keyframes-2, keyframes-3, etc.)
    _index_keyframe_directories(KEYFRAMES_DIR)

    # 2. Load memory-mapped vector search matrices & metadata
    idx_path = config["paths"]["unified_index"]
    _searcher = FastVectorSearchEngine(unified_index_dir=idx_path)

    # 3. Warm up GPU embedding & reranker models
    _registry = ModelRegistry.get_instance(
        siglip_id=config["models"]["siglip"],
        bge_id=config["models"]["bge_m3"],
        reranker_id=config["models"]["bge_reranker"],
    )
    logger.info("⚡ Pre-warming PyTorch GPU models...")
    _registry._load_siglip()
    _registry._load_bge()
    _registry._load_reranker()

    # 4. Instantiate fusion & reranker engines
    _fusion = MultimodalFusionEngine(
        searcher=_searcher,
        registry=_registry,
        k_rrf=config["retrieval"].get("k_rrf", 60),
    )
    vqa_reasoner = VQAReasoner(
        gemini_model_id=config["models"].get("gemini_model_id", "gemini-3.6-flash"),
        qwen_model_id=config["models"].get("qwen_model_id", "qwen2.5:7b"),
        ollama_url=config["models"].get("qwen_ollama_url", "http://localhost:11434/api/chat"),
    )
    _reranker = Stage2Reranker(registry=_registry, vqa_reasoner=vqa_reasoner)

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
    task_type: Optional[TaskType] = "KIS"
    engine: str = "gemini"  # "gemini" or "qwen"


class SearchRequest(BaseModel):
    parsed_query: ParsedQuery
    session_id: Optional[str] = None
    top_k_pool: int = Field(default=300, description="Stage 1 candidate pool size")
    top_k_rerank: int = Field(default=50, description="Stage 2 cross-encoder evaluated items")
    final_top_k: int = Field(default=20, description="Final number of top results returned")
    run_stage2: bool = Field(default=True, description="Whether to run Stage 2 cross-encoder")


class CachedReFuseRequest(BaseModel):
    session_id: str
    weights: dict[str, float]
    top_k_pool: int = 300
    top_k_rerank: int = 50
    final_top_k: int = 20
    run_stage2: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/config")
async def get_client_config():
    """Return paths and default settings for the frontend."""
    return {
        "keyframes_root": config["paths"]["keyframes_root"],
        "default_weights": config["retrieval"]["default_weights"],
        "top_k_pool": config["retrieval"]["top_k_pool"],
        "top_k_rerank": config["retrieval"]["top_k_rerank"],
        "gemini_model_id": config["models"].get("gemini_model_id", "gemini-3.6-flash"),
        "qwen_model_id": config["models"].get("qwen_model_id", "qwen2.5:7b"),
    }


@app.post("/api/parse")
async def parse_query_endpoint(req: ParseRequest):
    """Parse raw user query using Gemini or local Qwen with graceful fallbacks."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    t0 = time.perf_counter()
    parsed = _parser.parse(req.query, task_type=req.task_type, engine=req.engine)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "parsed_query": parsed.model_dump(),
        "execution_time_ms": round(dt_ms, 2),
    }


@app.post("/api/search")
async def search_endpoint(req: SearchRequest):
    """Full multimodal search: embed -> search -> cache branch hits -> fuse -> Stage 2 rerank."""
    t0 = time.perf_counter()
    parsed = req.parsed_query
    session_id = req.session_id or parsed.session_id or str(uuid.uuid4())
    parsed.session_id = session_id

    branch_limit = config["retrieval"].get("branch_limit", 500)

    # Branch A: TRAKE Multi-Event Pipeline
    if parsed.task_type == "TRAKE" and parsed.trake_events:
        event_queries = []
        event_pools = []
        event_branches = []
        for ev in parsed.trake_events:
            sub = ParsedQuery(
                task_type="KIS",
                original_query=ev.description,
                global_scene_en=ev.scene_en,
                objects_en=ev.objects_en,
                speech_vi=ev.speech_vi,
                ocr_keywords=ev.ocr_keywords,
                weights=parsed.weights,
            )
            event_queries.append(sub)
            ev_branches = _fusion.retrieve_branches(sub, branch_limit=branch_limit)
            event_branches.append(ev_branches)
            ev_pool = _fusion.fuse_from_branch_hits(
                vis_hits=ev_branches.get("vis", []),
                dam_hits=ev_branches.get("dam", []),
                asr_hits=ev_branches.get("asr", []),
                ocr_hits=ev_branches.get("ocr", []),
                weights=parsed.weights,
                top_k_pool=100,
            )
            event_pools.append(ev_pool)

        _branch_cache[session_id] = {
            "parsed_query": parsed,
            "branches": event_branches[0] if event_branches else {},
            "event_branches": event_branches,
            "event_queries": event_queries,
            "created_at": time.time(),
        }

        raw_sequences = _reranker.solve_trake_video_guided_dp(
            event_queries=event_queries,
            candidate_pools=event_pools,
            searcher=_searcher,
            top_n_videos=max(50, req.final_top_k),
            final_top_k=req.final_top_k,
        )
        event_descs = [ev.description for ev in parsed.trake_events]
        results = _reranker.rerank_trake_sequences(
            event_descriptions=event_descs,
            candidate_sequences=raw_sequences,
            searcher=_searcher,
            final_top_k=req.final_top_k,
        )
        fused_pool = raw_sequences

    # Branch B: KIS / VQA Standard Single-Query Pipeline
    else:
        # 1. Retrieve 4 branches independently
        branches = _fusion.retrieve_branches(parsed, branch_limit=branch_limit)

        # 2. Store branch hits in memory cache for instant weight adjustment re-runs
        _branch_cache[session_id] = {
            "parsed_query": parsed,
            "branches": branches,
            "created_at": time.time(),
        }

        # 3. Fuse branches with Weighted RRF & Synergy
        fused_pool = _fusion.fuse_from_branch_hits(
            vis_hits=branches["vis"],
            dam_hits=branches["dam"],
            asr_hits=branches["asr"],
            ocr_hits=branches["ocr"],
            weights=parsed.weights,
            top_k_pool=req.top_k_pool,
        )

        # 4. Optional Stage 2 Reranking
        results = fused_pool
        if req.run_stage2:
            if parsed.task_type == "KIS":
                results = _reranker.rerank_kis(
                    parsed,
                    fused_pool,
                    final_top_k=req.final_top_k,
                    top_k_rerank=req.top_k_rerank,
                )
            elif parsed.task_type == "VQA":
                results = _reranker.rerank_vqa(
                    parsed,
                    fused_pool,
                    final_top_k=req.final_top_k,
                    top_k_rerank=req.top_k_rerank,
                )

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "session_id": session_id,
        "task_type": parsed.task_type,
        "total_fused_candidates": len(fused_pool),
        "results": results,
        "execution_time_ms": round(dt_ms, 2),
    }


@app.post("/api/search/cached")
async def cached_re_fuse_endpoint(req: CachedReFuseRequest):
    """Instant (< 5ms) CPU-based RRF Re-Fusion using cached branch hits. No re-embedding."""
    t0 = time.perf_counter()
    entry = _branch_cache.get(req.session_id)
    if not entry:
        raise HTTPException(
            status_code=404,
            detail="Session ID not found in branch cache. Please run full search first.",
        )

    parsed: ParsedQuery = entry["parsed_query"]
    branches: dict[str, list[dict]] = entry["branches"]

    # 1. Instant Re-fuse on cached branch hits
    fused_pool = _fusion.fuse_from_branch_hits(
        vis_hits=branches.get("vis", []),
        dam_hits=branches.get("dam", []),
        asr_hits=branches.get("asr", []),
        ocr_hits=branches.get("ocr", []),
        weights=req.weights,
        top_k_pool=req.top_k_pool,
    )

    # 2. Stage 2 Reranking
    results = fused_pool
    if req.run_stage2:
        if parsed.task_type == "KIS":
            results = _reranker.rerank_kis(
                parsed,
                fused_pool,
                final_top_k=req.final_top_k,
                top_k_rerank=req.top_k_rerank,
            )
        elif parsed.task_type == "VQA":
            results = _reranker.rerank_vqa(
                parsed,
                fused_pool,
                final_top_k=req.final_top_k,
                top_k_rerank=req.top_k_rerank,
            )
        elif parsed.task_type == "TRAKE":
            event_branches = entry.get("event_branches", [])
            event_queries = entry.get("event_queries", [])
            if event_branches and event_queries:
                event_pools = []
                for ev_b, ev_q in zip(event_branches, event_queries):
                    ev_pool = _fusion.fuse_from_branch_hits(
                        vis_hits=ev_b.get("vis", []),
                        dam_hits=ev_b.get("dam", []),
                        asr_hits=ev_b.get("asr", []),
                        ocr_hits=ev_b.get("ocr", []),
                        weights=req.weights,
                        top_k_pool=req.top_k_pool,
                    )
                    event_pools.append(ev_pool)

                raw_sequences = _reranker.solve_trake_video_guided_dp(
                    event_queries=event_queries,
                    candidate_pools=event_pools,
                    searcher=_searcher,
                    top_n_videos=max(50, req.final_top_k),
                    final_top_k=req.final_top_k,
                )
                event_descs = [ev.description for ev in parsed.trake_events]
                results = _reranker.rerank_trake_sequences(
                    event_descriptions=event_descs,
                    candidate_sequences=raw_sequences,
                    searcher=_searcher,
                    final_top_k=req.final_top_k,
                )
            else:
                results = fused_pool

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "session_id": req.session_id,
        "is_cached": True,
        "total_fused_candidates": len(fused_pool),
        "results": results,
        "execution_time_ms": round(dt_ms, 2),
    }


@app.get("/api/keyframe/{video_id}/{keyframe_n}")
async def get_keyframe_detail(video_id: str, keyframe_n: int):
    """Fetch complete metadata, DAM bounding boxes, and surrounding ASR speech."""
    kf = _searcher.get_keyframe_by_video_and_n(video_id, keyframe_n)
    if not kf:
        raise HTTPException(status_code=404, detail=f"Keyframe {video_id}:{keyframe_n} not found")

    dam_objects = _searcher.get_dam_objects_for_frame(video_id, kf["frame_idx"])
    audio_span = _searcher.get_video_audio_span(video_id, max(0, kf["frame_idx"] - 450), kf["frame_idx"] + 450)

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


# ──────────────────────────────────────────────────────────────────────────────
# Static File & UI Serving
# ──────────────────────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
KEYFRAMES_DIR = Path(config["paths"]["keyframes_root"])


@app.get("/keyframes/{video_id}/{filename}")
async def serve_keyframe_image(video_id: str, filename: str):
    """Dynamically serve keyframe image across single or multi-batch folders (keyframes-1, keyframes-2, etc.)."""
    # 1. Look up in indexed directory map
    v_dir = _video_to_dir_map.get(video_id)
    if v_dir:
        target = v_dir / filename
        if target.exists():
            return FileResponse(target)

    # 2. Fallback direct check
    direct = KEYFRAMES_DIR / video_id / filename
    if direct.exists():
        return FileResponse(direct)

    # 3. Dynamic glob search fallback
    for match in KEYFRAMES_DIR.rglob(f"{video_id}/{filename}"):
        if match.exists():
            _video_to_dir_map[video_id] = match.parent
            return FileResponse(match)

    raise HTTPException(status_code=404, detail=f"Keyframe image {video_id}/{filename} not found")


# Mount frontend at / with html=True to automatically serve index.html, style.css, and app.js
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")



def main():
    import uvicorn

    host = os.environ.get("AIC_HOST") or config["server"].get("host", "127.0.0.1")
    port = int(os.environ.get("AIC_PORT") or config["server"].get("port", 8890))
    logger.info(f"Starting server on http://{host}:{port}")
    uvicorn.run("online.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
