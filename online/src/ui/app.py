"""FastAPI Web Server for Online Video Retrieval Application."""

from __future__ import annotations

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from online.src.contracts.query import ParsedQuery, SearchResponse, TaskType
from online.src.retrieval.pipeline import VideoRetrievalEngine

app = FastAPI(title="AIC-2026 Multimodal Video Retrieval Engine", version="1.0.0")

# Enable CORS for local web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
STATIC_DIR = Path(__file__).parent / "static"
KEYFRAMES_ROOT = Path("/Users/khoale/Downloads/AIC_Challenger/data/keyframes")
QDRANT_DB_PATH = Path("/Users/khoale/Downloads/AIC_HCM/qdrant_db")

# Global Engine Instance (lazy-loaded or initialized at startup)
engine: VideoRetrievalEngine = None


@app.on_event("startup")
def startup_event():
    global engine
    engine = VideoRetrievalEngine(
        qdrant_db_path=str(QDRANT_DB_PATH),
        keyframes_root=str(KEYFRAMES_ROOT),
    )
    engine.models.warmup()


class ParseRequest(BaseModel):
    query: str
    task_type: TaskType = "KIS"


class SearchRequest(BaseModel):
    parsed_query: ParsedQuery
    top_k: int = 50


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "engine_ready": engine is not None,
        "qdrant_db": str(QDRANT_DB_PATH),
    }


@app.post("/api/parse_query", response_model=ParsedQuery)
def parse_query_endpoint(req: ParseRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return engine.parse_query(req.query, task_type=req.task_type)


@app.post("/api/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return engine.search(query=req.parsed_query, top_k=req.top_k)


# Keyframe image serving with robust fallback
@app.get("/keyframes/{video_id}/{filename}")
def get_keyframe_image(video_id: str, filename: str):
    # Direct match
    img_path = KEYFRAMES_ROOT / video_id / filename
    if img_path.exists():
        return FileResponse(img_path)

    # Alternate formatting (001.jpg, 01.jpg, 1.jpg, 0001.jpg)
    try:
        stem = filename.split(".")[0]
        n = int(stem)
        candidates = [
            KEYFRAMES_ROOT / video_id / f"{n:03d}.jpg",
            KEYFRAMES_ROOT / video_id / f"{n:04d}.jpg",
            KEYFRAMES_ROOT / video_id / f"{n}.jpg",
            KEYFRAMES_ROOT / video_id / f"{n:02d}.jpg",
        ]
        for c in candidates:
            if c.exists():
                return FileResponse(c)
    except Exception:
        pass

    raise HTTPException(status_code=404, detail="Image not on local disk")


# Mount static web UI assets
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
