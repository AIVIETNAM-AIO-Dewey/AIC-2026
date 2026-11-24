"""Live Apple Silicon MPS / CPU subset retrieval workbench for local demos.

Serves the compiled Retrieval Studio UI alongside real M5 Pro MPS GPU neural
inference (SigLIP2, MetaCLIP 2, BEiT-3 COCO) and SQLite FTS5 lexical retrieval
over any locally extracted subset of competition keyframe archives, DAM region
descriptions, SigLIP2 scene shards, and Vietnamese OCR transcripts.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_default_root(env_var: str, primary: Path, fallback: Path) -> Path:
    configured = os.environ.get(env_var)
    if configured:
        return Path(configured)
    return primary if primary.is_dir() else fallback


DATA_ROOT = Path(os.environ.get("AIC_DATA_ROOT", str(REPO_ROOT / "data" / "artifacts")))
METACLIP2_ROOT = _resolve_default_root(
    "AIC_METACLIP2_ROOT",
    DATA_ROOT / "visual_embeddings" / "metaclip2",
    REPO_ROOT / "data" / "metaclip2",
)
BEIT3_ROOT = _resolve_default_root(
    "AIC_BEIT3_ROOT",
    DATA_ROOT / "visual_embeddings" / "beit3",
    REPO_ROOT / "data" / "beit3",
)
STATE_ROOT = Path(os.environ.get("AIC_STATE_ROOT", str(REPO_ROOT / ".local" / "mps_demo")))
MODEL_ROOT = Path(os.environ.get("AIC_MODEL_ROOT", str(REPO_ROOT / "models")))
BRANCH1_MODEL_ROOT = Path(os.environ.get("AIC_BRANCH1_MODEL_ROOT", str(MODEL_ROOT / "branch1")))

# Set model/HuggingFace cache env vars before importing transformers or encoder modules
# because huggingface_hub freezes HF_HUB_CACHE at module import time.
os.environ.setdefault("AIC_MODEL_ROOT", str(MODEL_ROOT))
os.environ.setdefault("AIC_BRANCH1_MODEL_ROOT", str(BRANCH1_MODEL_ROOT))
os.environ.setdefault("AIC_QUERY_MODEL_MANIFEST", str(MODEL_ROOT / "query_models.json"))
os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "huggingface"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image, ImageOps, UnidentifiedImageError  # noqa: E402
from safetensors.numpy import load_file as load_safetensors  # noqa: E402

from online.src.contracts.api import (  # noqa: E402
    Branch1SearchRequest,
    Branch2SearchRequest,
    Branch3AsrSearchRequest,
    Branch3OcrSearchRequest,
    DiscoveryCascadeRequest,
    KisFusionSearchRequest,
    KisQueryPlanRequest,
    KisTemporalSearchRequest,
    ParseRequest,
    RelatedFramesRequest,
    SearchRequest,
    SubmissionPrepareRequest,
    TemporalIntersectionRequest,
    VideoVisualFusionRequest,
)
from online.src.contracts.query import ParsedQuery  # noqa: E402
from online.src.retrieval.branches.branch1.contracts import (  # noqa: E402
    MODEL_SPECS,
    QUERY_ROLES,
    normalize_model_weights,
)
from online.src.retrieval.branches.branch1.fusion import aggregate_model_streams  # noqa: E402
from online.src.retrieval.branches.branch2.contracts import (  # noqa: E402
    DEFAULT_PER_STREAM_TOP_K,
    DEFAULT_PRE_RERANK_TOP_K,
    DEFAULT_RERANK_TOP_K,
)
from online.src.retrieval.branches.branch2.fusion import fuse_dense_sparse  # noqa: E402
from online.src.retrieval.branches.branch3.contracts import (  # noqa: E402
    ASR_RESULT_SCHEMA_VERSION,
)
from online.src.retrieval.branches.branch3.contracts import (  # noqa: E402
    DEFAULT_FINAL_TOP_K as BRANCH3_FINAL_TOP_K,
)
from online.src.retrieval.branches.branch3.ocr import OCR_RESULT_SCHEMA_VERSION  # noqa: E402
from online.src.retrieval.branches.final_fusion.contracts import (  # noqa: E402
    DEFAULT_BRANCH_WEIGHTS,
)
from online.src.retrieval.branches.final_fusion.query_plan import (  # noqa: E402
    build_kis_query_plan,
)
from online.src.retrieval.branches.final_fusion.service import KisFusionSearch  # noqa: E402
from online.src.retrieval.branches.final_fusion.temporal import (  # noqa: E402
    KisTemporalFusionSearch,
    focus_event_query_bundle,
)
from online.src.retrieval.branches.rerankers.beit3_cosine import (  # noqa: E402
    Beit3CosineReranker,
)
from online.src.retrieval.encoders.device import (  # noqa: E402
    clear_accelerator_cache,
    device_contract,
    move_tensors,
    resolve_device,
    transformers_dtype_kwargs,
)
from online.src.retrieval.infrastructure.persistent_cache import (  # noqa: E402
    PersistentQueryEmbeddingCache,
)
from online.src.retrieval.infrastructure.qdrant import base_frame  # noqa: E402
from online.src.retrieval.infrastructure.query_parser import LocalQueryParser  # noqa: E402
from online.src.retrieval.infrastructure.resources import current_process_rss_bytes  # noqa: E402
from online.src.retrieval.infrastructure.scoring import normalize_weights  # noqa: E402
from online.src.retrieval.modalities.lexical import (  # noqa: E402
    _folded_tokens,
    _ordered_lexical_bigrams,
    _stream_identity,
    _token_bigrams,
    fold_text,
    ordered_lexical_tokens,
    query_tokens,
    sigmoid_zscore,
)
from online.src.retrieval.modalities.ocr import repair_mojibake  # noqa: E402
from online.src.retrieval.modalities.workbench import IndependentModalitySearch  # noqa: E402
from online.src.submission import (  # noqa: E402
    RelatedFrameSearch,
    SourceFrameIndex,
    prepare_submission,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("aic_mps_demo")

KEYFRAMES_ROOT = DATA_ROOT / "keyframes"


def _resolve_media_info_root(data_root: Path, configured: str | None = None) -> Path:
    if configured:
        return Path(configured)
    canonical = data_root / "media-info"
    legacy = data_root / "media_info"
    return legacy if not canonical.is_dir() and legacy.is_dir() else canonical


MEDIA_INFO_ROOT = _resolve_media_info_root(DATA_ROOT, os.environ.get("AIC_MEDIA_INFO_ROOT"))
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend" / "dist"
MAX_IMAGE_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MAX_IMAGE_DIMENSION = 8192
ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
DEMO_CORPUS_VERSION = 1

parser = LocalQueryParser()
heavy_search_lock = threading.Lock()


class MpsEncoderPool:
    """Keep pinned visual/text encoders resident on MPS (or CPU fallback)."""

    def __init__(self, model_root: Path, *, device: str | None = None) -> None:
        from online.src.retrieval.encoders.sequential_manager import (
            BEIT3_CHECKPOINT_NAME,
            METACLIP2_REVISION,
            SIGLIP_REVISION,
            UNILM_REVISION,
        )

        self.model_root = Path(model_root)
        self.requested_device = str(device or os.environ.get("AIC_DEVICE", "auto")).strip().lower()
        self.device = resolve_device(self.requested_device, allow_fallback=True)
        self.revisions = {
            "siglip2": SIGLIP_REVISION,
            "metaclip2": METACLIP2_REVISION,
            "beit3": f"{BEIT3_CHECKPOINT_NAME}@{UNILM_REVISION}",
        }
        self._lock = threading.RLock()
        self._encoders: dict[str, Any] = {}
        self._siglip_processor: Any = None
        self._siglip_image_model: Any = None

    def cache_device_for_model(self, _model_name: str) -> str:
        return self.device

    @property
    def cache_device(self) -> str:
        return self.device

    def _get_encoder(self, model_name: str) -> Any:
        from online.src.retrieval.encoders.sequential_manager import SequentialBranch1Encoders

        with self._lock:
            encoder = self._encoders.get(model_name)
            if encoder is None:
                encoder = SequentialBranch1Encoders(
                    model_root=self.model_root,
                    device=self.device,
                    allow_cpu_fallback=True,
                )
                self._encoders[model_name] = encoder
            return encoder

    def encode(self, model_name: str, texts: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        encoder = self._get_encoder(model_name)
        return encoder.encode(model_name, texts)

    @torch.inference_mode()
    def embed_siglip_image(self, image: Image.Image) -> np.ndarray:
        from transformers import AutoModel, AutoProcessor

        from online.src.retrieval.encoders.sequential_manager import SIGLIP_ID, SIGLIP_REVISION

        with self._lock:
            if self._siglip_processor is None:
                self._siglip_processor = AutoProcessor.from_pretrained(
                    SIGLIP_ID,
                    revision=SIGLIP_REVISION,
                    trust_remote_code=False,
                    local_files_only=True,
                    use_fast=False,
                )
            if self._siglip_image_model is None:
                siglip_enc = self._get_encoder("siglip2")
                siglip_enc._load("siglip2")
                self._siglip_image_model = siglip_enc._model
            if self._siglip_image_model is None:
                self._siglip_image_model = (
                    AutoModel.from_pretrained(
                        SIGLIP_ID,
                        revision=SIGLIP_REVISION,
                        trust_remote_code=False,
                        low_cpu_mem_usage=True,
                        local_files_only=True,
                        **transformers_dtype_kwargs(torch.float32),
                    )
                    .to(self.device)
                    .eval()
                )
            inputs = self._siglip_processor(images=image.convert("RGB"), return_tensors="pt")
            inputs = move_tensors(inputs, self.device)
            output = self._siglip_image_model.get_image_features(**inputs)
            if isinstance(output, torch.Tensor):
                features = output
            elif getattr(output, "pooler_output", None) is not None:
                features = output.pooler_output
            else:
                features = output[0]
            vector = F.normalize(features.float(), p=2, dim=-1)[0]
            return vector.cpu().numpy().astype(np.float32)

    def embed_siglip_text(self, text: str) -> np.ndarray:
        vectors, _ = self.encode("siglip2", [text] * 12)
        return vectors[0]

    def siglip_text_diagnostics(self, text: str) -> dict[str, Any]:
        encoder = self._get_encoder("siglip2")
        encoder._load("siglip2")
        assert encoder._tokenizer is not None
        token_ids = encoder._tokenizer(text, add_special_tokens=True)["input_ids"]
        return {
            "token_count": len(token_ids),
            "max_tokens": 64,
            "truncated": len(token_ids) > 64,
            "effective_query": text,
        }

    def health(self) -> dict[str, Any]:
        contract = device_contract(self.requested_device, allow_fallback=True)
        return {
            "siglip2": {"ready": True, "execution_device": self.device, "device": contract},
            "metaclip2": {"ready": True, "execution_device": self.device, "device": contract},
            "beit3": {"ready": True, "execution_device": self.device, "device": contract},
        }

    def unload(self) -> None:
        # Keep models resident in unified MPS memory for interactive query latency.
        return

    def close_active(self) -> None:
        with self._lock:
            for encoder in self._encoders.values():
                encoder.unload()
            self._encoders.clear()
            self._siglip_processor = None
            self._siglip_image_model = None
        clear_accelerator_cache(self.device)


def _canonical_video_id(value: str) -> str:
    canonical = value.strip().upper().replace("-", "_")
    if not re.fullmatch(r"[A-Z0-9]+_V\d+", canonical):
        raise HTTPException(status_code=400, detail="Invalid video ID")
    return canonical


class MpsSubsetCorpus:
    """In-memory MPS tensor index + SQLite FTS5 store for extracted keyframes."""

    def __init__(
        self,
        *,
        data_root: Path,
        metaclip2_root: Path,
        beit3_root: Path,
        state_root: Path,
        device: str,
    ) -> None:
        self.data_root = Path(data_root)
        self.metaclip2_root = Path(metaclip2_root)
        self.beit3_root = Path(beit3_root)
        self.state_root = Path(state_root)
        self.device = torch.device(device)
        self.keyframes_root = self.data_root / "keyframes"
        self.db_path = self.state_root / "demo_text.sqlite3"
        self._db_lock = threading.RLock()
        self._db: sqlite3.Connection | None = None

        self.frames: list[dict[str, Any]] = []
        self.frame_point_ids: dict[str, int] = {}
        self.frame_by_uid: dict[str, dict[str, Any]] = {}
        self.frame_by_point_id: dict[int, dict[str, Any]] = {}
        self.point_id_to_row: dict[int, int] = {}
        self.video_frames_map: dict[str, tuple[dict[str, Any], ...]] = {}
        self.video_row_indices: dict[str, torch.Tensor] = {}

        self.metaclip2_tensor: torch.Tensor | None = None
        self.beit3_tensor: torch.Tensor | None = None
        self.siglip2_tensor: torch.Tensor | None = None
        self.siglip2_rows_to_point_ids: list[int] = []
        self.siglip2_point_id_to_row: dict[int, int] = {}
        self.siglip2_video_rows: dict[str, torch.Tensor] = {}
        self.siglip2_videos: set[str] = set()
        self.ocr_videos: set[str] = set()
        self.dam_region_count = 0
        self.ocr_frame_count = 0

        self._load()

    def _discover_keyframes(self) -> tuple[set[str], set[tuple[str, str]]]:
        if not self.keyframes_root.is_dir():
            raise RuntimeError(f"Keyframes directory not found: {self.keyframes_root}")
        videos: set[str] = set()
        files: set[tuple[str, str]] = set()
        for video_dir in sorted(self.keyframes_root.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name.upper().replace("-", "_")
            jpgs = list(video_dir.glob("*.jpg"))
            if not jpgs:
                continue
            videos.add(video_id)
            for img in jpgs:
                files.add((video_id, img.name))
        if not files:
            raise RuntimeError(f"No .jpg keyframes found in {self.keyframes_root}")
        return videos, files

    @staticmethod
    def _normalize_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
        tensor = torch.from_numpy(array.astype(np.float32, copy=False))
        tensor = F.normalize(tensor, p=2, dim=-1)
        return tensor.to(device)

    def _load(self) -> None:
        started = time.perf_counter()
        self.state_root.mkdir(parents=True, exist_ok=True)
        available_videos, available_files = self._discover_keyframes()

        meta_path = self.metaclip2_root / "keyframes_metadata.jsonl"
        metaclip_npy = self.metaclip2_root / "keyframes_visual_vectors.f16.npy"
        beit3_npy = self.beit3_root / "keyframes_visual_vectors.f16.npy"
        if not meta_path.is_file() or not metaclip_npy.is_file() or not beit3_npy.is_file():
            raise RuntimeError(
                f"Required visual embedding files missing under {self.metaclip2_root} / {self.beit3_root}"
            )

        selected_rows: list[int] = []
        by_video: dict[str, list[dict[str, Any]]] = {}
        video_indices: dict[str, list[int]] = {}
        with meta_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                video_id = str(item["video_id"]).upper().replace("-", "_")
                if video_id not in available_videos:
                    continue
                filename = str(
                    item.get("filename") or Path(str(item.get("image_relpath", ""))).name
                )
                if (video_id, filename) not in available_files:
                    continue
                frame_idx = int(item["frame_idx"])
                point_id = int(item["point_id"])
                frame_uid = f"{video_id}:{frame_idx}"
                frame = {
                    "point_id": point_id,
                    "global_idx": point_id,
                    "frame_uid": frame_uid,
                    "video_id": video_id,
                    "keyframe_n": int(item["keyframe_n"]),
                    "frame_idx": frame_idx,
                    "pts_time_s": float(item["pts_time_s"]),
                    "fps": float(item["fps"]),
                    "image_relpath": f"keyframes/{video_id}/{filename}",
                    "filename": filename,
                    "submission_string": f"{video_id}, {frame_idx}",
                }
                demo_row = len(self.frames)
                self.frames.append(frame)
                self.frame_point_ids[frame_uid] = point_id
                self.frame_by_uid[frame_uid] = frame
                self.frame_by_point_id[point_id] = frame
                self.point_id_to_row[point_id] = demo_row
                selected_rows.append(int(item["global_vector_row"]))
                by_video.setdefault(video_id, []).append(frame)
                video_indices.setdefault(video_id, []).append(demo_row)

        if not selected_rows:
            raise RuntimeError("No canonical frames matched the extracted keyframe images")

        for video_id, frames in by_video.items():
            frames.sort(key=lambda row: int(row["frame_idx"]))
            self.video_frames_map[video_id] = tuple(frames)
            self.video_row_indices[video_id] = torch.tensor(
                video_indices[video_id], dtype=torch.long, device=self.device
            )

        idx_arr = np.asarray(selected_rows, dtype=np.int64)
        mc_mmap = np.load(metaclip_npy, mmap_mode="r", allow_pickle=False)
        bt_mmap = np.load(beit3_npy, mmap_mode="r", allow_pickle=False)
        self.metaclip2_tensor = self._normalize_tensor(np.asarray(mc_mmap[idx_arr]), self.device)
        self.beit3_tensor = self._normalize_tensor(np.asarray(bt_mmap[idx_arr]), self.device)

        self._load_siglip_subset()
        self._prepare_sqlite_indexes()
        LOGGER.info(
            "Loaded MPS subset corpus: %d videos, %d keyframes, %d SigLIP2 frames, "
            "%d DAM regions, %d OCR frames on %s in %.2fs",
            len(self.video_frames_map),
            len(self.frames),
            len(self.siglip2_rows_to_point_ids),
            self.dam_region_count,
            self.ocr_frame_count,
            self.device,
            time.perf_counter() - started,
        )

    def _load_siglip_subset(self) -> None:
        scene_dir = self.data_root / "scene_embeddings"
        if not scene_dir.is_dir():
            return
        chunks: list[np.ndarray] = []
        siglip_vid_rows: dict[str, list[int]] = {}
        for jsonl_path in sorted(scene_dir.glob("*.jsonl")):
            video_id = jsonl_path.stem.upper().replace("-", "_")
            if video_id not in self.video_frames_map:
                continue
            tensor_path = scene_dir / f"{jsonl_path.stem}.safetensors"
            if not tensor_path.is_file():
                continue
            matrix = load_safetensors(tensor_path).get("embeddings")
            if matrix is None or matrix.ndim != 2 or matrix.shape[1] != 768:
                continue
            row_map: list[int] = []
            point_ids: list[int] = []
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    uid = str(rec.get("frame_uid") or f"{video_id}:{int(rec['frame_idx'])}")
                    frame = self.frame_by_uid.get(uid)
                    if frame is None:
                        continue
                    local_row = int(rec["row"])
                    if 0 <= local_row < matrix.shape[0]:
                        row_map.append(local_row)
                        point_ids.append(int(frame["point_id"]))
            if not row_map:
                continue
            sliced = np.asarray(matrix[np.asarray(row_map, dtype=np.int64)], dtype=np.float32)
            base_idx = len(self.siglip2_rows_to_point_ids)
            chunks.append(sliced)
            self.siglip2_videos.add(video_id)
            vid_indices: list[int] = []
            for offset, pid in enumerate(point_ids):
                global_sig_row = base_idx + offset
                self.siglip2_rows_to_point_ids.append(pid)
                self.siglip2_point_id_to_row[pid] = global_sig_row
                vid_indices.append(global_sig_row)
            siglip_vid_rows[video_id] = vid_indices

        if chunks:
            stacked = np.concatenate(chunks, axis=0)
            self.siglip2_tensor = self._normalize_tensor(stacked, self.device)
            for video_id, indices in siglip_vid_rows.items():
                self.siglip2_video_rows[video_id] = torch.tensor(
                    indices, dtype=torch.long, device=self.device
                )

    def _source_signature(self) -> str:
        dam_path = self.data_root / "dense_text_embeddings" / "dam_metadata.jsonl"
        dam_stat = dam_path.stat() if dam_path.is_file() else None
        ocr_dir = self.data_root / "ocr_transcripts"
        ocr_files = (
            sorted(
                p.name
                for p in ocr_dir.glob("*.jsonl")
                if p.stem.upper().replace("-", "_") in self.video_frames_map
            )
            if ocr_dir.is_dir()
            else []
        )
        return json.dumps(
            {
                "version": DEMO_CORPUS_VERSION,
                "videos": len(self.video_frames_map),
                "frames": len(self.frames),
                "dam_mtime": int(dam_stat.st_mtime_ns) if dam_stat else 0,
                "dam_size": int(dam_stat.st_size) if dam_stat else 0,
                "ocr_files": ocr_files,
            },
            sort_keys=True,
        )

    def _prepare_sqlite_indexes(self) -> None:
        signature = self._source_signature()
        rebuild = True
        if self.db_path.is_file():
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT value FROM demo_meta WHERE key = 'signature'").fetchone()
                if row and row["value"] == signature:
                    self._db = conn
                    self.dam_region_count = int(
                        conn.execute("SELECT COUNT(*) FROM dam_regions").fetchone()[0]
                    )
                    self.ocr_frame_count = int(
                        conn.execute("SELECT COUNT(*) FROM ocr_frames").fetchone()[0]
                    )
                    ocr_vids = conn.execute("SELECT DISTINCT video_id FROM ocr_frames").fetchall()
                    self.ocr_videos = {str(r["video_id"]) for r in ocr_vids}
                    rebuild = False
                else:
                    conn.close()
            except sqlite3.Error:
                rebuild = True

        if not rebuild:
            return

        staging = self.db_path.with_suffix(".staging.sqlite3")
        if staging.exists():
            staging.unlink()
        db = sqlite3.connect(staging, check_same_thread=False)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode = OFF")
            db.execute("PRAGMA synchronous = OFF")
            db.execute(
                """CREATE TABLE dam_regions (
                    id INTEGER PRIMARY KEY,
                    region_id TEXT UNIQUE NOT NULL,
                    point_id INTEGER NOT NULL,
                    frame_uid TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    keyframe_n INTEGER NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    pts_time_s REAL NOT NULL,
                    fps REAL NOT NULL,
                    image_relpath TEXT NOT NULL,
                    class_entity TEXT NOT NULL,
                    bbox_json TEXT NOT NULL,
                    description_en TEXT NOT NULL
                )"""
            )
            db.execute(
                "CREATE VIRTUAL TABLE dam_fts USING fts5("
                "content_text, content='', tokenize='unicode61')"
            )
            db.execute(
                """CREATE TABLE ocr_frames (
                    id INTEGER PRIMARY KEY,
                    frame_uid TEXT UNIQUE NOT NULL,
                    point_id INTEGER NOT NULL,
                    video_id TEXT NOT NULL,
                    keyframe_n INTEGER NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    pts_time_s REAL NOT NULL,
                    fps REAL NOT NULL,
                    image_relpath TEXT NOT NULL,
                    full_text TEXT NOT NULL,
                    folded_text TEXT NOT NULL
                )"""
            )
            db.execute(
                "CREATE VIRTUAL TABLE ocr_fts USING fts5("
                "folded_text, content='', tokenize='unicode61')"
            )
            db.execute("CREATE TABLE demo_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

            dam_path = self.data_root / "dense_text_embeddings" / "dam_metadata.jsonl"
            if dam_path.is_file():
                dam_rows: list[tuple[Any, ...]] = []
                dam_fts_rows: list[tuple[int, str]] = []
                row_id = 0
                with dam_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        video_id = str(item.get("video_id") or "").upper().replace("-", "_")
                        if video_id not in self.video_frames_map:
                            continue
                        frame_uid = f"{video_id}:{int(item['frame_idx'])}"
                        frame = self.frame_by_uid.get(frame_uid)
                        if frame is None:
                            continue
                        row_id += 1
                        class_entity = str(item.get("class_entity") or "object")
                        desc = str(item.get("description_en") or "")
                        dam_rows.append(
                            (
                                row_id,
                                str(item["region_id"]),
                                int(frame["point_id"]),
                                frame_uid,
                                video_id,
                                int(frame["keyframe_n"]),
                                int(frame["frame_idx"]),
                                float(frame["pts_time_s"]),
                                float(frame["fps"]),
                                str(frame["image_relpath"]),
                                class_entity,
                                json.dumps(item.get("bbox") or []),
                                desc,
                            )
                        )
                        dam_fts_rows.append((row_id, f"{class_entity} {desc}".strip()))
                        if len(dam_rows) >= 5_000:
                            db.executemany(
                                "INSERT INTO dam_regions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                dam_rows,
                            )
                            db.executemany(
                                "INSERT INTO dam_fts(rowid, content_text) VALUES (?,?)",
                                dam_fts_rows,
                            )
                            dam_rows.clear()
                            dam_fts_rows.clear()
                if dam_rows:
                    db.executemany(
                        "INSERT INTO dam_regions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        dam_rows,
                    )
                    db.executemany(
                        "INSERT INTO dam_fts(rowid, content_text) VALUES (?,?)",
                        dam_fts_rows,
                    )
                self.dam_region_count = row_id

            db.execute("CREATE INDEX dam_regions_frame_idx ON dam_regions(frame_uid)")

            ocr_dir = self.data_root / "ocr_transcripts"
            if ocr_dir.is_dir():
                ocr_rows: list[tuple[Any, ...]] = []
                ocr_fts_rows: list[tuple[int, str]] = []
                ocr_id = 0
                for jsonl_path in sorted(ocr_dir.glob("*.jsonl")):
                    video_id = jsonl_path.stem.upper().replace("-", "_")
                    if video_id not in self.video_frames_map:
                        continue
                    with jsonl_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            item = json.loads(line)
                            frame_idx = int(item["frame_idx"])
                            frame_uid = f"{video_id}:{frame_idx}"
                            frame = self.frame_by_uid.get(frame_uid)
                            if frame is None:
                                continue
                            raw_text = repair_mojibake(str(item.get("full_text") or "").strip())
                            folded = fold_text(raw_text)
                            ocr_id += 1
                            self.ocr_videos.add(video_id)
                            ocr_rows.append(
                                (
                                    ocr_id,
                                    frame_uid,
                                    int(frame["point_id"]),
                                    video_id,
                                    int(frame["keyframe_n"]),
                                    int(frame["frame_idx"]),
                                    float(frame["pts_time_s"]),
                                    float(frame["fps"]),
                                    str(frame["image_relpath"]),
                                    raw_text,
                                    folded,
                                )
                            )
                            ocr_fts_rows.append((ocr_id, folded))
                if ocr_rows:
                    db.executemany(
                        "INSERT INTO ocr_frames VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        ocr_rows,
                    )
                    db.executemany(
                        "INSERT INTO ocr_fts(rowid, folded_text) VALUES (?,?)",
                        ocr_fts_rows,
                    )
                self.ocr_frame_count = ocr_id

            db.execute("INSERT INTO demo_meta(key, value) VALUES ('signature', ?)", (signature,))
            db.execute("PRAGMA optimize")
            db.commit()
        except BaseException:
            db.close()
            if staging.exists():
                staging.unlink()
            raise
        db.close()
        os.replace(staging, self.db_path)
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row

    def dam_for_frame(self, video_id: str, frame_idx: int) -> list[dict[str, Any]]:
        if self._db is None:
            return []
        uid = f"{video_id}:{int(frame_idx)}"
        with self._db_lock:
            rows = self._db.execute(
                "SELECT region_id, class_entity, bbox_json, description_en "
                "FROM dam_regions WHERE frame_uid = ? ORDER BY region_id",
                (uid,),
            ).fetchall()
        return [
            {
                "region_id": str(row["region_id"]),
                "class_entity": str(row["class_entity"]),
                "bbox": json.loads(row["bbox_json"]),
                "description_en": str(row["description_en"]),
            }
            for row in rows
        ]

    def search_dam_bm25(self, queries: list[str], top_k: int) -> dict[str, dict[str, Any]]:
        if self._db is None:
            return {}
        hits: dict[str, dict[str, Any]] = {}
        for role, query in zip(QUERY_ROLES, queries, strict=True):
            tokens = list(dict.fromkeys(re.findall(r"[\w]+", query.casefold(), flags=re.UNICODE)))[
                :48
            ]
            if not tokens:
                continue
            expression = " OR ".join(f'"{tok.replace(chr(34), "")}"' for tok in tokens)
            with self._db_lock:
                rows = self._db.execute(
                    "SELECT r.*, bm25(dam_fts) AS bm25_raw FROM dam_fts "
                    "JOIN dam_regions r ON r.id = dam_fts.rowid "
                    "WHERE dam_fts MATCH ? ORDER BY bm25_raw LIMIT ?",
                    (expression, int(top_k)),
                ).fetchall()
            for rank, row in enumerate(rows, 1):
                frame_uid = str(row["frame_uid"])
                score = 1.0 / (60.0 + rank)
                bm25_raw = float(row["bm25_raw"])
                evidence = {
                    "rank": rank,
                    "rank_score": score,
                    "bm25_raw": bm25_raw,
                    "region_id": str(row["region_id"]),
                    "language": "en",
                    "observed": True,
                }
                current = hits.get(frame_uid)
                if current is None:
                    hits[frame_uid] = {
                        "point_id": int(row["point_id"]),
                        "global_idx": int(row["point_id"]),
                        "frame_uid": frame_uid,
                        "video_id": str(row["video_id"]),
                        "keyframe_n": int(row["keyframe_n"]),
                        "frame_idx": int(row["frame_idx"]),
                        "pts_time_s": float(row["pts_time_s"]),
                        "fps": float(row["fps"]),
                        "image_relpath": str(row["image_relpath"]),
                        "submission_string": f"{row['video_id']}, {int(row['frame_idx'])}",
                        "sparse_raw": score,
                        "sparse_bm25_raw": bm25_raw,
                        "sparse_best_query_role": role,
                        "sparse_best_query_language": "en",
                        "sparse_query_scores": {role: evidence},
                        "sparse_winner": {
                            "point_id": int(row["point_id"]),
                            "region_id": str(row["region_id"]),
                            "class_entity": str(row["class_entity"]),
                            "bbox": json.loads(row["bbox_json"]),
                            "description_en": str(row["description_en"]),
                            "query_role": role,
                            "query_language": "en",
                            "rank": rank,
                            "bm25_raw": bm25_raw,
                        },
                    }
                else:
                    role_scores = current.setdefault("sparse_query_scores", {})
                    prev = role_scores.get(role)
                    if prev is None or score > float(prev["rank_score"]):
                        role_scores[role] = evidence
                    if score > float(current["sparse_raw"]):
                        current["sparse_raw"] = score
                        current["sparse_bm25_raw"] = bm25_raw
                        current["sparse_best_query_role"] = role
                        current["sparse_winner"] = {
                            "point_id": int(row["point_id"]),
                            "region_id": str(row["region_id"]),
                            "class_entity": str(row["class_entity"]),
                            "bbox": json.loads(row["bbox_json"]),
                            "description_en": str(row["description_en"]),
                            "query_role": role,
                            "query_language": "en",
                            "rank": rank,
                            "bm25_raw": bm25_raw,
                        }
        ordered = sorted(
            hits.values(),
            key=lambda item: (-float(item["sparse_raw"]), str(item["frame_uid"])),
        )
        for rank, item in enumerate(ordered, 1):
            item["sparse_rank"] = rank
        return hits

    def lookup_ocr_many(self, frame_uids: Any) -> dict[str, str]:
        if self._db is None:
            return {}
        uids = [str(uid) for uid in frame_uids if uid]
        if not uids:
            return {}
        placeholders = ",".join("?" for _ in uids)
        with self._db_lock:
            rows = self._db.execute(
                f"SELECT frame_uid, full_text FROM ocr_frames WHERE frame_uid IN ({placeholders})",
                uids,
            ).fetchall()
        return {str(row["frame_uid"]): str(row["full_text"]) for row in rows}

    def close(self) -> None:
        with self._db_lock:
            if self._db is not None:
                self._db.close()
                self._db = None


class MpsSubsetQdrant:
    """Qdrant-compatible cosine retriever running directly on Apple Silicon MPS tensors."""

    def __init__(self, corpus: MpsSubsetCorpus) -> None:
        self.corpus = corpus
        self.device = corpus.device

    def collection(self, name: str) -> dict[str, Any]:
        if name == "aic_frames":
            return {"status": "green", "points_count": len(self.corpus.frames)}
        if name == "aic_beit3_frames":
            return {"status": "green", "points_count": len(self.corpus.frames)}
        if name == "aic_dam_regions":
            return {"status": "green", "points_count": self.corpus.dam_region_count}
        return {"status": "green", "points_count": 0}

    def _space(self, collection: str, vector_name: str) -> tuple[torch.Tensor | None, str]:
        if collection == "aic_frames" and vector_name == "siglip2":
            return self.corpus.siglip2_tensor, "siglip2"
        if collection == "aic_frames" and vector_name == "metaclip2":
            return self.corpus.metaclip2_tensor, "metaclip2"
        if collection == "aic_beit3_frames" and vector_name == "beit3":
            return self.corpus.beit3_tensor, "beit3"
        return None, vector_name

    @torch.inference_mode()
    def query(
        self,
        collection: str,
        vector_name: str,
        query_vector: np.ndarray,
        limit: int,
        query_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        matrix, space = self._space(collection, vector_name)
        if matrix is None or matrix.shape[0] == 0:
            return []
        q_np = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        q_tensor = F.normalize(torch.from_numpy(q_np).to(self.device), p=2, dim=-1)

        has_ids: list[int] | None = None
        video_id_filter: str | None = None
        if isinstance(query_filter, dict):
            for clause in query_filter.get("must") or []:
                if not isinstance(clause, dict):
                    continue
                if "has_id" in clause:
                    has_ids = [int(pid) for pid in clause["has_id"]]
                elif clause.get("key") == "video_id":
                    match = clause.get("match") or {}
                    if "value" in match:
                        video_id_filter = str(match["value"]).upper().replace("-", "_")

        if has_ids is not None:
            row_map = (
                self.corpus.siglip2_point_id_to_row
                if space == "siglip2"
                else self.corpus.point_id_to_row
            )
            rows = [row_map[pid] for pid in has_ids if pid in row_map]
            if not rows:
                return []
            row_tensor = torch.tensor(rows, dtype=torch.long, device=self.device)
            sub_matrix = matrix.index_select(0, row_tensor)
            scores = torch.matmul(sub_matrix, q_tensor)
            k = min(int(limit), len(rows))
            top_scores, top_pos = torch.topk(scores, k=k, sorted=True)
            row_ids = [rows[idx] for idx in top_pos.cpu().tolist()]
            score_vals = top_scores.cpu().tolist()
        elif video_id_filter is not None:
            idx_tensor = (
                self.corpus.siglip2_video_rows.get(video_id_filter)
                if space == "siglip2"
                else self.corpus.video_row_indices.get(video_id_filter)
            )
            if idx_tensor is None or idx_tensor.numel() == 0:
                return []
            sub_matrix = matrix.index_select(0, idx_tensor)
            scores = torch.matmul(sub_matrix, q_tensor)
            k = min(int(limit), int(idx_tensor.numel()))
            top_scores, top_pos = torch.topk(scores, k=k, sorted=True)
            row_ids = idx_tensor.index_select(0, top_pos).cpu().tolist()
            score_vals = top_scores.cpu().tolist()
        else:
            scores = torch.matmul(matrix, q_tensor)
            k = min(int(limit), int(matrix.shape[0]))
            top_scores, top_pos = torch.topk(scores, k=k, sorted=True)
            row_ids = top_pos.cpu().tolist()
            score_vals = top_scores.cpu().tolist()

        results: list[dict[str, Any]] = []
        for row_idx, score in zip(row_ids, score_vals, strict=True):
            if space == "siglip2":
                point_id = self.corpus.siglip2_rows_to_point_ids[row_idx]
                payload = self.corpus.frame_by_point_id[point_id]
            else:
                payload = self.corpus.frames[row_idx]
                point_id = int(payload["point_id"])
            results.append({"id": point_id, "score": float(score), "payload": payload})
        return results

    def find_frame_point(
        self, _collection: str, video_id: str, frame_idx: int
    ) -> dict[str, Any] | None:
        uid = f"{video_id.upper().replace('-', '_')}:{int(frame_idx)}"
        frame = self.corpus.frame_by_uid.get(uid)
        if frame is None:
            return None
        return {"id": int(frame["point_id"]), "payload": frame}

    def query_by_id(
        self, collection: str, vector_name: str, point_id: int, limit: int
    ) -> list[dict[str, Any]]:
        matrix, space = self._space(collection, vector_name)
        if matrix is None:
            return []
        row_map = (
            self.corpus.siglip2_point_id_to_row
            if space == "siglip2"
            else self.corpus.point_id_to_row
        )
        row_idx = row_map.get(int(point_id))
        if row_idx is None:
            return []
        seed_vec = matrix[row_idx].detach().cpu().numpy()
        return self.query(collection, vector_name, seed_vec, limit=limit)

    def dam_for_frame(self, video_id: str, frame_idx: int) -> list[dict[str, Any]]:
        return self.corpus.dam_for_frame(video_id, frame_idx)

    def close(self) -> None:
        return


class MpsMetadataStore:
    """Metadata adapter backed by the locally extracted 55-video subset."""

    def __init__(self, corpus: MpsSubsetCorpus) -> None:
        self.corpus = corpus

    def video_frames(self, video_id: str) -> tuple[dict[str, Any], ...]:
        canonical = video_id.upper().replace("-", "_")
        return self.corpus.video_frames_map.get(canonical, ())

    def detail(self, video_id: str, keyframe_n: int) -> dict[str, Any] | None:
        canonical = video_id.upper().replace("-", "_")
        for frame in self.video_frames(canonical):
            if int(frame["keyframe_n"]) == int(keyframe_n):
                item = dict(frame)
                ocr_map = self.corpus.lookup_ocr_many([item["frame_uid"]])
                item["ocr_text"] = ocr_map.get(item["frame_uid"], "")
                return item
        return None

    def frame_by_idx(self, video_id: str, frame_idx: int) -> dict[str, Any] | None:
        canonical = video_id.upper().replace("-", "_")
        frame = self.corpus.frame_by_uid.get(f"{canonical}:{int(frame_idx)}")
        if frame is None:
            return None
        item = dict(frame)
        ocr_map = self.corpus.lookup_ocr_many([item["frame_uid"]])
        item["ocr_text"] = ocr_map.get(item["frame_uid"], "")
        return item

    def lookup_many(self, frame_uids: Any) -> dict[str, str]:
        return self.corpus.lookup_ocr_many(frame_uids)


class MpsBranch1Search:
    """Three-model MPS visual search with fair per-video weight normalization."""

    def __init__(
        self,
        qdrant: MpsSubsetQdrant,
        encoder: MpsEncoderPool,
        cache: PersistentQueryEmbeddingCache,
        search_lock: threading.Lock,
    ) -> None:
        self.qdrant = qdrant
        self.encoder = encoder
        self.cache = cache
        self._search_lock = search_lock

    def execute(
        self,
        query_bundle: dict[str, Any],
        weights: dict[str, float],
        per_stream_top_k: int,
        final_top_k: int,
        *,
        _lock_already_held: bool = False,
        _video_id: str | None = None,
    ) -> dict[str, Any]:
        weights = normalize_model_weights(weights)
        scoped_video_id = (
            str(_video_id).strip().upper().replace("-", "_") if _video_id is not None else None
        )
        acquired = False
        if not _lock_already_held:
            if not self._search_lock.acquire(blocking=False):
                raise RuntimeError("BRANCH1_SEARCH_BUSY")
            acquired = True
        started = time.perf_counter()
        try:
            by_role = {str(item["role"]): item for item in query_bundle["queries"]}
            stream_contracts = {
                model_name: [
                    {"role": role, "language": language}
                    for role in QUERY_ROLES
                    for language in spec["languages"]
                ]
                for model_name, spec in MODEL_SPECS.items()
            }
            model_candidates: dict[str, dict[str, dict[str, Any]]] = {}
            diagnostics: dict[str, Any] = {}
            timing: dict[str, Any] = {"models": {}}
            for model_name, spec in MODEL_SPECS.items():
                model_started = time.perf_counter()
                descriptors = stream_contracts[model_name]
                texts = [str(by_role[d["role"]][d["language"]]) for d in descriptors]
                revision = self.encoder.revisions[model_name]
                tokenizer_config = (
                    "languages=vi,en;max_tokens=64;normalization=l2"
                    if model_name == "siglip2"
                    else "languages=vi,en;max_tokens=77;normalization=l2"
                    if model_name == "metaclip2"
                    else "languages=en;max_tokens=64;output=language_head;normalization=l2"
                )
                stream_contract = [
                    {"role": d["role"], "language": d["language"], "text": text}
                    for d, text in zip(descriptors, texts, strict=True)
                ]
                cache_key = self.cache.key(
                    model_name,
                    revision,
                    texts,
                    tokenizer_config=tokenizer_config,
                    stream_contract=stream_contract,
                    device=self.encoder.device,
                )
                cached = self.cache.get(cache_key)
                encode_started = time.perf_counter()
                if cached is None:
                    vectors, model_diag = self.encoder.encode(model_name, texts)
                    model_diag = [
                        {
                            **dict(diag),
                            "role": d["role"],
                            "language": d["language"],
                            "stream": f"{d['role']}:{d['language']}",
                        }
                        for diag, d in zip(model_diag, descriptors, strict=True)
                    ]
                    self.cache.put(cache_key, model_name, vectors, model_diag)
                    cache_hit = False
                else:
                    vectors, model_diag = cached
                    cache_hit = True
                encode_ms = (time.perf_counter() - encode_started) * 1000.0
                q_started = time.perf_counter()
                query_filter = (
                    {"must": [{"key": "video_id", "match": {"value": scoped_video_id}}]}
                    if scoped_video_id
                    else None
                )
                streams = [
                    self.qdrant.query(
                        str(spec["collection"]),
                        str(spec["vector"]),
                        vectors[idx],
                        per_stream_top_k,
                        query_filter,
                    )
                    for idx in range(len(descriptors))
                ]
                q_ms = (time.perf_counter() - q_started) * 1000.0
                stream_keys = tuple(f"{d['role']}:{d['language']}" for d in descriptors)
                model_candidates[model_name] = aggregate_model_streams(stream_keys, streams)
                diagnostics[model_name] = model_diag
                timing["models"][model_name] = {
                    "cache_hit": cache_hit,
                    "encoding_ms": round(encode_ms, 2),
                    "qdrant_search_ms": round(q_ms, 2),
                    "candidate_count": len(model_candidates[model_name]),
                    "total_ms": round((time.perf_counter() - model_started) * 1000.0, 2),
                }

            results = self._fuse_subset_candidates(model_candidates, weights, int(final_top_k))
            timing["total_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            return {
                "schema_version": "branch1.result.v1",
                "fusion_applied": True,
                "reranking_applied": False,
                "weights": weights,
                "per_stream_top_k": per_stream_top_k,
                "final_top_k": final_top_k,
                "candidate_union_count": len(
                    set().union(*(set(v) for v in model_candidates.values()))
                ),
                "candidate_count_before_gate": len(
                    set().union(*(set(v) for v in model_candidates.values()))
                ),
                "gate_top_k": int(final_top_k),
                "future_fusion_eligible": scoped_video_id is None,
                "scope": "video" if scoped_video_id else "global",
                "video_id": scoped_video_id,
                "query_streams": {
                    name: [{**d, "stream": f"{d['role']}:{d['language']}"} for d in descriptors]
                    for name, descriptors in stream_contracts.items()
                },
                "stream_count": sum(len(d) for d in stream_contracts.values()),
                "result_count": len(results),
                "tokenizer_diagnostics": diagnostics,
                "timing": timing,
                "results": results,
            }
        finally:
            if acquired:
                self._search_lock.release()

    def _fuse_subset_candidates(
        self,
        model_candidates: dict[str, dict[str, dict[str, Any]]],
        weights: dict[str, float],
        final_top_k: int,
    ) -> list[dict[str, Any]]:
        frame_uids = set().union(*(set(v) for v in model_candidates.values()))
        siglip_videos = self.qdrant.corpus.siglip2_videos
        results: list[dict[str, Any]] = []
        for frame_uid in frame_uids:
            payload: dict[str, Any] = {}
            provenance: dict[str, Any] = {}
            best_rank = math.inf
            best_order = math.inf
            best_model = None
            best_role = None
            best_lang = None
            video_id = frame_uid.split(":", 1)[0]
            available_models = [
                name for name in MODEL_SPECS if name != "siglip2" or video_id in siglip_videos
            ]
            weight_denom = sum(weights[name] for name in available_models) or 1.0
            final_score = 0.0
            for order, model_name in enumerate(MODEL_SPECS):
                item = model_candidates.get(model_name, {}).get(frame_uid)
                if item is None:
                    provenance[model_name] = {
                        "observed": False,
                        "raw_cosine": None,
                        "normalized_score": 0.0,
                        "best_query_role": None,
                        "best_query_language": None,
                        "best_query_rank": None,
                        "query_scores": {},
                    }
                    continue
                if not payload:
                    payload = dict(item["payload"])
                cand_rank = int(item["best_query_rank"])
                if (cand_rank, order) < (best_rank, best_order):
                    best_rank = cand_rank
                    best_order = order
                    best_model = model_name
                    best_role = item.get("best_query_role")
                    best_lang = item.get("best_query_language")
                norm_score = float(item["normalized_score"])
                eff_weight = weights[model_name] / weight_denom
                final_score += eff_weight * norm_score
                provenance[model_name] = {
                    "observed": True,
                    "raw_cosine": float(item["raw_score"]),
                    "normalized_score": norm_score,
                    "best_query_role": item["best_query_role"],
                    "best_query_language": item.get("best_query_language"),
                    "best_query_rank": cand_rank,
                    "query_scores": item["query_scores"],
                    "normalization_mean": float(item["normalization_mean"]),
                    "normalization_std": float(item["normalization_std"]),
                }
            results.append(
                {
                    **payload,
                    "frame_uid": frame_uid,
                    "global_idx": int(payload["point_id"]),
                    "final_score": final_score,
                    "score": final_score,
                    "score_type": "weighted_zsigmoid_fusion",
                    "best_stream_rank": None if math.isinf(best_rank) else int(best_rank),
                    "best_model": best_model,
                    "best_query_role": best_role,
                    "best_query_language": best_lang,
                    "model_provenance": provenance,
                }
            )
        results.sort(
            key=lambda item: (
                -item["final_score"],
                item["best_stream_rank"] if item["best_stream_rank"] is not None else math.inf,
                item["frame_uid"],
            )
        )
        selected = results[:final_top_k]
        for rank, item in enumerate(selected, 1):
            item["rank"] = rank
            item["final_score"] = round(float(item["final_score"]), 8)
            item["score"] = item["final_score"]
        return selected

    def _execute_locked(
        self,
        query_bundle: dict[str, Any],
        weights: dict[str, float],
        per_stream_top_k: int,
        final_top_k: int,
    ) -> dict[str, Any]:
        return self.execute(
            query_bundle, weights, per_stream_top_k, final_top_k, _lock_already_held=True
        )

    def execute_in_video(
        self, query_bundle: dict[str, Any], video_id: str, top_k: int
    ) -> dict[str, Any]:
        return self.execute(
            query_bundle,
            {"siglip2": 0.45, "metaclip2": 0.30, "beit3": 0.25},
            int(top_k),
            int(top_k),
            _video_id=video_id,
        )

    def health(self) -> dict[str, Any]:
        models_health = self.encoder.health()
        return {
            "status": "ready",
            "ready": True,
            "production_ready": False,
            "fail_closed": True,
            "execution_device": self.encoder.device,
            "models": {
                name: {
                    "data": {"ready": True},
                    "collection": {"ready": True},
                    "text_encoder": models_health[name],
                }
                for name in ("siglip2", "metaclip2", "beit3")
            },
        }


class MpsBranch2Search:
    """Branch 2 subset service: SQLite DAM BM25 + live MPS BEiT-3 COCO rerank."""

    def __init__(
        self,
        corpus: MpsSubsetCorpus,
        qdrant: MpsSubsetQdrant,
        beit_encoders: MpsEncoderPool,
        cache: PersistentQueryEmbeddingCache,
        search_lock: threading.Lock,
    ) -> None:
        self.corpus = corpus
        self.qdrant = qdrant
        self.beit_encoders = beit_encoders
        self.cache = cache
        self.search_lock = search_lock
        self._reranker = Beit3CosineReranker(qdrant, beit_encoders, corpus.frame_point_ids)

    def health(self) -> dict[str, Any]:
        contract = device_contract(self.beit_encoders.requested_device, allow_fallback=True)
        return {
            "status": "ready",
            "ready": True,
            "production_ready": False,
            "fail_closed": True,
            "components": {
                "dam_data": {"ready": True, "documents": self.corpus.dam_region_count},
                "dam_collection": {"ready": False, "subset_mode": "bm25_plus_beit3_mps"},
                "bm25": {"ready": True, "documents": self.corpus.dam_region_count},
                "bge_text_encoder": {"ready": False},
                "beit3_collection": {"ready": True},
                "beit3_ingestion": {"ready": True},
                "beit3_text_encoder": {
                    "ready": True,
                    "execution_device": self.beit_encoders.device,
                    "device": contract,
                },
                "frame_mapping": {"ready": True, "frames": len(self.corpus.frames)},
            },
        }

    def execute(
        self,
        query_bundle: dict[str, Any],
        hybrid_weights: dict[str, float],
        rerank_weights: dict[str, float],
        per_stream_top_k: int = DEFAULT_PER_STREAM_TOP_K,
        pre_rerank_top_k: int = DEFAULT_PRE_RERANK_TOP_K,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        *,
        _lock_already_held: bool = False,
    ) -> dict[str, Any]:
        acquired = False
        if not _lock_already_held:
            if not self.search_lock.acquire(blocking=False):
                raise RuntimeError("BRANCH2_SEARCH_BUSY")
            acquired = True
        started = time.perf_counter()
        try:
            by_role = {str(item["role"]): item for item in query_bundle["queries"]}
            texts = [str(by_role[role]["en"]) for role in QUERY_ROLES]
            sparse_started = time.perf_counter()
            sparse_hits = self.corpus.search_dam_bm25(texts, per_stream_top_k)
            sparse_ms = round((time.perf_counter() - sparse_started) * 1000.0, 2)

            hybrid = fuse_dense_sparse(
                {},
                sparse_hits,
                {"dense": 0.0, "sparse": 1.0},
                pre_rerank_top_k,
            )
            norm_rerank = normalize_weights(rerank_weights, ("beit3", "previous"))
            rerank_started = time.perf_counter()
            beit_diagnostics: list[dict[str, Any]] = []
            if not hybrid:
                reranked: list[dict[str, Any]] = []
                rerank_info: dict[str, Any] = {
                    "candidate_count": 0,
                    "rerank_count": 0,
                    "weights": norm_rerank,
                    "scoring": "cosine",
                    "checkpoint_task": "BEiT-3 COCO Retrieval",
                }
                beit_cache_hit = False
            else:
                stream_contract = [
                    {"role": role, "language": "en", "text": text}
                    for role, text in zip(QUERY_ROLES, texts, strict=True)
                ]
                ckey = self.cache.key(
                    "beit3",
                    self.beit_encoders.revisions["beit3"],
                    texts,
                    tokenizer_config=(
                        "languages=en;max_tokens=64;output=language_head;normalization=l2"
                    ),
                    stream_contract=stream_contract,
                    device=self.beit_encoders.device,
                )
                cached = self.cache.get(ckey)
                if cached is None:
                    beit_vectors, beit_diagnostics = self.beit_encoders.encode("beit3", texts)
                    beit_diagnostics = [
                        {**dict(d), "role": role, "language": "en", "stream": f"{role}:en"}
                        for d, role in zip(beit_diagnostics, QUERY_ROLES, strict=True)
                    ]
                    self.cache.put(ckey, "beit3", beit_vectors, beit_diagnostics)
                    beit_cache_hit = False
                else:
                    beit_vectors, beit_diagnostics = cached
                    beit_cache_hit = True
                reranked, rerank_info = self._reranker.rerank(
                    hybrid,
                    texts,
                    top_k=rerank_top_k,
                    weights=norm_rerank,
                    text_vectors=beit_vectors,
                    tokenizer_diagnostics=beit_diagnostics,
                )
            reranked = list(reranked)[:DEFAULT_PRE_RERANK_TOP_K]
            rerank_ms = round((time.perf_counter() - rerank_started) * 1000.0, 2)
            return {
                "schema_version": "branch2.result.v1",
                "fusion_applied": True,
                "reranking_applied": bool(reranked),
                "hybrid_weights": normalize_weights(hybrid_weights, ("dense", "sparse")),
                "rerank_weights": norm_rerank,
                "per_stream_top_k": per_stream_top_k,
                "pre_rerank_top_k": pre_rerank_top_k,
                "rerank_top_k": rerank_top_k,
                "result_count": len(reranked),
                "candidate_count_before_gate": len(sparse_hits),
                "gate_top_k": DEFAULT_PRE_RERANK_TOP_K,
                "future_fusion_eligible": True,
                "stream_count": 12,
                "dense_candidate_count": 0,
                "sparse_candidate_count": len(sparse_hits),
                "timing": {
                    "sparse_ms": sparse_ms,
                    "rerank_ms": rerank_ms,
                    "beit_cache_hit": beit_cache_hit,
                    "total_ms": round((time.perf_counter() - started) * 1000.0, 2),
                },
                "rerank": rerank_info,
                "results": reranked,
            }
        finally:
            if acquired:
                self.search_lock.release()

    def _execute_locked(
        self,
        query_bundle: dict[str, Any],
        hybrid_weights: dict[str, float],
        rerank_weights: dict[str, float],
        per_stream_top_k: int = DEFAULT_PER_STREAM_TOP_K,
        pre_rerank_top_k: int = DEFAULT_PRE_RERANK_TOP_K,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    ) -> dict[str, Any]:
        return self.execute(
            query_bundle,
            hybrid_weights,
            rerank_weights,
            per_stream_top_k,
            pre_rerank_top_k,
            rerank_top_k,
            _lock_already_held=True,
        )


class MpsBranch3OcrSearch:
    """Branch-3 OCR SQLite FTS5 search over locally available OCR transcripts."""

    def __init__(self, corpus: MpsSubsetCorpus, search_lock: threading.Lock) -> None:
        self.corpus = corpus
        self._search_lock = search_lock

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "ready": True,
            "production_ready": False,
            "required": False,
            "branch": "branch3",
            "modality": "ocr",
            "fail_closed": False,
            "frames": self.corpus.ocr_frame_count,
            "fts_frames": self.corpus.ocr_frame_count,
            "mapped_frames": self.corpus.ocr_frame_count,
            "videos": len(self.corpus.ocr_videos),
        }

    def _query_stream(self, query: str, limit: int) -> tuple[list[sqlite3.Row], list[str]]:
        tokens = query_tokens(query)
        if not tokens or self.corpus._db is None:
            return [], tokens
        expr = " OR ".join(f'"{tok.replace(chr(34), " ")}"' for tok in tokens)
        with self.corpus._db_lock:
            rows = self.corpus._db.execute(
                """
                SELECT f.id, f.frame_uid, f.point_id, f.video_id, f.keyframe_n,
                       f.frame_idx, f.pts_time_s, f.fps, f.image_relpath,
                       f.full_text, bm25(ocr_fts) AS bm25_score
                FROM ocr_fts
                JOIN ocr_frames AS f ON f.id = ocr_fts.rowid
                WHERE ocr_fts MATCH ?
                ORDER BY bm25_score, f.id
                LIMIT ?
                """,
                (expr, int(limit)),
            ).fetchall()
        return rows, tokens

    def execute_single(self, query: str, top_k: int) -> list[dict[str, Any]]:
        payload = self._search_streams({"legacy": query}, top_k, top_k)
        return payload["results"]

    def _search_streams(
        self, streams: dict[str, str], per_stream_top_k: int, final_top_k: int
    ) -> dict[str, Any]:
        started = time.perf_counter()
        frame_hits: dict[str, dict[str, Any]] = {}
        stream_counts: dict[str, int] = {}
        stream_order = list(streams)
        for stream, query in streams.items():
            role, language = _stream_identity(stream)
            rows, tokens = self._query_stream(query, per_stream_top_k)
            stream_counts[stream] = len(rows)
            max_rel = max((max(0.0, -float(r["bm25_score"])) for r in rows), default=0.0) or 1.0
            ordered_toks = ordered_lexical_tokens(query)
            query_bigrams = _ordered_lexical_bigrams(query)
            tok_set = set(tokens)
            occ = sum(1 for tok in ordered_toks if tok in tok_set)
            for rank, row in enumerate(rows, 1):
                uid = str(row["frame_uid"])
                full_text = str(row["full_text"] or "")
                t_tokens = _folded_tokens(full_text)
                t_set = set(t_tokens)
                matched = [tok for tok in tokens if tok in t_set]
                tok_cov = len(matched) / len(tokens) if tokens else 0.0
                t_bigrams = set(_token_bigrams(t_tokens))
                if query_bigrams:
                    ngram_cov = sum(1 for p in query_bigrams if p in t_bigrams) / len(query_bigrams)
                elif occ == 1:
                    ngram_cov = tok_cov
                else:
                    ngram_cov = 0.0
                bm25_raw = float(row["bm25_score"])
                bm25_rel = max(0.0, -bm25_raw) / max_rel
                combined = 0.50 * bm25_rel + 0.30 * tok_cov + 0.20 * ngram_cov
                ev = {
                    "role": role,
                    "language": language,
                    "stream": stream,
                    "rank": rank,
                    "bm25_raw": round(bm25_raw, 8),
                    "bm25_relevance": round(bm25_rel, 8),
                    "token_coverage": round(tok_cov, 8),
                    "ngram_coverage": round(ngram_cov, 8),
                    "combined_score": round(combined, 8),
                    "matched_terms": matched,
                }
                cand = frame_hits.setdefault(
                    uid,
                    {
                        "frame_uid": uid,
                        "point_id": int(row["point_id"]),
                        "video_id": str(row["video_id"]),
                        "keyframe_n": int(row["keyframe_n"]),
                        "frame_idx": int(row["frame_idx"]),
                        "pts_time_s": float(row["pts_time_s"]),
                        "fps": float(row["fps"]),
                        "image_relpath": str(row["image_relpath"]),
                        "full_text": full_text,
                        "query_scores": {},
                        "stream_provenance": {},
                    },
                )
                cand["query_scores"][stream] = combined
                cand["stream_provenance"][stream] = ev

        if not frame_hits:
            return {
                "results": [],
                "candidate_frame_count": 0,
                "stream_counts": stream_counts,
                "timing": {"total_ms": round((time.perf_counter() - started) * 1000.0, 2)},
            }
        values = [
            max(float(v) for v in item["query_scores"].values()) for item in frame_hits.values()
        ]
        normalized = sigmoid_zscore(values)
        ranked: list[dict[str, Any]] = []
        for idx, cand in enumerate(frame_hits.values()):
            best_stream, raw_score = max(
                cand["query_scores"].items(),
                key=lambda p: (
                    float(p[1]),
                    -int(cand["stream_provenance"][p[0]]["rank"]),
                    -stream_order.index(p[0]),
                ),
            )
            role, lang = _stream_identity(best_stream)
            winner = cand["stream_provenance"][best_stream]
            frame = base_frame(
                cand, score=float(normalized[idx]), rank=1, score_type="ocr_bm25_ngram"
            )
            frame.update(
                {
                    "frame_uid": cand["frame_uid"],
                    "point_id": cand["point_id"],
                    "global_idx": cand["point_id"],
                    "ocr_text": cand["full_text"],
                    "full_text": cand["full_text"],
                    "ocr_raw_score": round(float(raw_score), 8),
                    "ocr_combined_score": round(float(raw_score), 8),
                    "ocr_normalized_score": round(float(normalized[idx]), 8),
                    "raw_score": round(float(raw_score), 8),
                    "normalized_score": round(float(normalized[idx]), 8),
                    "best_query_role": role,
                    "best_query_language": lang,
                    "best_query_stream": best_stream,
                    "ocr_best_query_role": role,
                    "ocr_best_query_language": lang,
                    "ocr_best_query_stream": best_stream,
                    "ocr_best_rank": int(winner["rank"]),
                    "matched_keywords": winner["matched_terms"],
                    "matched_terms": winner["matched_terms"],
                    "bm25_raw": winner["bm25_raw"],
                    "ocr_bm25_raw": winner["bm25_raw"],
                    "bm25_relevance": winner["bm25_relevance"],
                    "ocr_bm25_relevance": winner["bm25_relevance"],
                    "token_coverage": winner["token_coverage"],
                    "ocr_token_coverage": winner["token_coverage"],
                    "ngram_coverage": winner["ngram_coverage"],
                    "adjacent_bigram_coverage": winner["ngram_coverage"],
                    "ocr_stream_provenance": cand["stream_provenance"],
                }
            )
            ranked.append(frame)
        ranked.sort(
            key=lambda item: (
                -float(item["ocr_normalized_score"]),
                int(item.get("ocr_best_rank", 2_000_001)),
                str(item["frame_uid"]),
            )
        )
        selected = ranked[:final_top_k]
        for rank, frame in enumerate(selected, 1):
            frame["rank"] = rank
        return {
            "results": selected,
            "candidate_frame_count": len(frame_hits),
            "stream_counts": stream_counts,
            "timing": {"total_ms": round((time.perf_counter() - started) * 1000.0, 2)},
        }

    def execute(
        self,
        query_bundle: dict[str, Any],
        per_stream_top_k: int,
        final_top_k: int,
        *,
        _lock_already_held: bool = False,
    ) -> dict[str, Any]:
        acquired = False
        if not _lock_already_held:
            if not self._search_lock.acquire(blocking=False):
                raise RuntimeError("BRANCH3_OCR_SEARCH_BUSY")
            acquired = True
        try:
            by_role = {str(item["role"]): item for item in query_bundle["queries"]}
            streams = {
                f"{role}:{lang}": str(by_role[role][lang]).strip()
                for role in QUERY_ROLES
                for lang in ("vi", "en")
            }
            payload = self._search_streams(streams, per_stream_top_k, final_top_k)
            return {
                "schema_version": OCR_RESULT_SCHEMA_VERSION,
                "fusion_applied": False,
                "reranking_applied": False,
                "query_roles": list(QUERY_ROLES),
                "query_languages": ["vi", "en"],
                "stream_count": len(streams),
                "per_stream_top_k": per_stream_top_k,
                "final_top_k": final_top_k,
                "candidate_frame_count": payload["candidate_frame_count"],
                "stream_counts": payload["stream_counts"],
                "result_count": len(payload["results"]),
                "candidate_count_before_gate": payload["candidate_frame_count"],
                "gate_top_k": BRANCH3_FINAL_TOP_K,
                "future_fusion_eligible": True,
                "timing": payload["timing"],
                "results": payload["results"],
            }
        finally:
            if acquired:
                self._search_lock.release()

    def _execute_locked(
        self, query_bundle: dict[str, Any], per_stream_top_k: int, final_top_k: int
    ) -> dict[str, Any]:
        return self.execute(query_bundle, per_stream_top_k, final_top_k, _lock_already_held=True)


class MpsBranch3AsrSearch:
    """Empty fail-safe ASR adapter when asr_segments/ is absent in a local subset."""

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "ready": True,
            "production_ready": False,
            "required": False,
            "branch": "branch3",
            "modality": "asr",
            "segments": 0,
            "mapped_segments": 0,
            "warnings": ["ASR segments are absent in the local 55-video subset"],
        }

    def execute(
        self,
        _query_bundle: dict[str, Any],
        per_stream_top_k: int,
        final_top_k: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": ASR_RESULT_SCHEMA_VERSION,
            "fusion_applied": False,
            "reranking_applied": False,
            "query_roles": list(QUERY_ROLES),
            "query_languages": ["vi", "en"],
            "stream_count": 12,
            "per_stream_top_k": per_stream_top_k,
            "final_top_k": final_top_k,
            "candidate_segment_count": 0,
            "candidate_frame_count": 0,
            "stream_counts": {},
            "result_count": 0,
            "candidate_count_before_gate": 0,
            "gate_top_k": BRANCH3_FINAL_TOP_K,
            "future_fusion_eligible": True,
            "timing": {"total_ms": 0.0},
            "results": [],
        }

    def _execute_locked(
        self, query_bundle: dict[str, Any], per_stream_top_k: int, final_top_k: int
    ) -> dict[str, Any]:
        return self.execute(query_bundle, per_stream_top_k, final_top_k)


class MpsKisFusionSearch(KisFusionSearch):
    """Canonical KIS weighted RRF + MPS BEiT-3 COCO rerank over the extracted subset."""

    def __init__(self, *args: Any, corpus: MpsSubsetCorpus, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.corpus = corpus
        self._frame_point_ids = dict(corpus.frame_point_ids)

    def _load_frame_point_ids(self) -> dict[str, int]:
        return self._frame_point_ids

    def health(self) -> dict[str, Any]:
        return {
            "schema_version": "kis.fusion.health.v1",
            "branch": "final_fusion",
            "task_type": "KIS",
            "status": "ready",
            "ready": True,
            "required": False,
            "production_ready": False,
            "fail_closed": True,
            "rrf_k": 60,
            "final_top_k": 150,
            "rerank_top_k": 100,
            "beit3_weight": 0.25,
            "rrf_weight": 0.75,
            "branch_weights": dict(DEFAULT_BRANCH_WEIGHTS),
            "subset_frames": len(self.corpus.frames),
            "subset_videos": len(self.corpus.video_frames_map),
            "api_rss_bytes": current_process_rss_bytes(),
        }


class _WorkbenchCompatAdapter:
    """Adapter exposing visual/DAM/OCR search to legacy workbench endpoints."""

    def __init__(
        self,
        qdrant: MpsSubsetQdrant,
        encoders: MpsEncoderPool,
        corpus: MpsSubsetCorpus,
        ocr_service: MpsBranch3OcrSearch,
    ) -> None:
        self.qdrant = qdrant
        self.encoders = encoders
        self.corpus = corpus
        self.ocr_service = ocr_service

    def get_video_frame_count(self, video_id: str) -> int:
        return len(self.corpus.video_frames_map.get(video_id.upper().replace("-", "_"), ()))

    def search_visual(self, vector: np.ndarray, *, top_k: int) -> list[dict[str, Any]]:
        points = self.qdrant.query("aic_frames", "siglip2", vector, top_k)
        return self._attach_ocr(points)

    def search_visual_in_video(
        self, vector: np.ndarray, video_id: str, *, top_k: int
    ) -> list[dict[str, Any]]:
        canonical = video_id.upper().replace("-", "_")
        if canonical in self.corpus.siglip2_videos:
            points = self.qdrant.query(
                "aic_frames",
                "siglip2",
                vector,
                top_k,
                {"must": [{"key": "video_id", "match": {"value": canonical}}]},
            )
            return self._attach_ocr(points)
        return []

    def _attach_ocr(self, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = [
            {
                **base_frame(pt["payload"], score=pt["score"], rank=idx, score_type="cosine"),
                "global_idx": int(pt["id"]),
            }
            for idx, pt in enumerate(points, 1)
        ]
        ocr_map = self.corpus.lookup_ocr_many(r["frame_uid"] for r in results)
        for row in results:
            row["ocr_text"] = ocr_map.get(row["frame_uid"], "")
        return results

    def search_dam_queries(self, queries: list[str], top_k: int) -> list[dict[str, Any]]:
        if not queries:
            return []
        padded = (queries + [queries[0]] * 6)[:6]
        hits = self.corpus.search_dam_bm25(padded, top_k)
        ordered = sorted(hits.values(), key=lambda h: (-float(h["sparse_raw"]), h["frame_uid"]))[
            :top_k
        ]
        for rank, item in enumerate(ordered, 1):
            item["rank"] = rank
            item["score"] = item["sparse_raw"]
            item["score_type"] = "bm25"
        return ordered

    def search_speech(self, _query: str, *, top_k: int) -> list[dict[str, Any]]:
        return []


corpus: MpsSubsetCorpus | None = None
mps_qdrant: MpsSubsetQdrant | None = None
mps_encoders: MpsEncoderPool | None = None
metadata_store: MpsMetadataStore | None = None
source_frame_index: SourceFrameIndex | None = None
related_frame_searcher: RelatedFrameSearch | None = None
query_cache: PersistentQueryEmbeddingCache | None = None
branch1_searcher: MpsBranch1Search | None = None
branch2_searcher: MpsBranch2Search | None = None
branch3_ocr_searcher: MpsBranch3OcrSearch | None = None
branch3_asr_searcher: MpsBranch3AsrSearch | None = None
kis_fusion_searcher: MpsKisFusionSearch | None = None
compat_searcher: _WorkbenchCompatAdapter | None = None
workbench_search: IndependentModalitySearch | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global corpus, mps_qdrant, mps_encoders, metadata_store, source_frame_index
    global related_frame_searcher, query_cache, branch1_searcher, branch2_searcher
    global branch3_ocr_searcher, branch3_asr_searcher, kis_fusion_searcher
    global compat_searcher, workbench_search

    started = time.perf_counter()
    device = resolve_device(os.environ.get("AIC_DEVICE", "auto"), allow_fallback=True)
    corpus = MpsSubsetCorpus(
        data_root=DATA_ROOT,
        metaclip2_root=METACLIP2_ROOT,
        beit3_root=BEIT3_ROOT,
        state_root=STATE_ROOT,
        device=device,
    )
    mps_qdrant = MpsSubsetQdrant(corpus)
    mps_encoders = MpsEncoderPool(BRANCH1_MODEL_ROOT, device=device)
    metadata_store = MpsMetadataStore(corpus)
    source_frame_index = SourceFrameIndex(metadata_store, MEDIA_INFO_ROOT)
    related_frame_searcher = RelatedFrameSearch(mps_qdrant, metadata_store)
    query_cache = PersistentQueryEmbeddingCache(STATE_ROOT / "query_embeddings.sqlite3")

    branch1_searcher = MpsBranch1Search(mps_qdrant, mps_encoders, query_cache, heavy_search_lock)
    branch2_searcher = MpsBranch2Search(
        corpus, mps_qdrant, mps_encoders, query_cache, heavy_search_lock
    )
    branch3_ocr_searcher = MpsBranch3OcrSearch(corpus, heavy_search_lock)
    branch3_asr_searcher = MpsBranch3AsrSearch()
    kis_fusion_searcher = MpsKisFusionSearch(
        branch1_searcher,
        branch2_searcher,
        branch3_asr_searcher,
        branch3_ocr_searcher,
        data_root=DATA_ROOT,
        state_root=STATE_ROOT,
        search_lock=heavy_search_lock,
        corpus=corpus,
    )
    compat_searcher = _WorkbenchCompatAdapter(
        corpus=corpus, qdrant=mps_qdrant, encoders=mps_encoders, ocr_service=branch3_ocr_searcher
    )
    workbench_search = IndependentModalitySearch(
        searcher=compat_searcher, registry=mps_encoders, dam_match_threshold=0.50
    )
    LOGGER.info(
        "MPS subset demo ready on %s (%d videos, %d frames) in %.2fs",
        device,
        len(corpus.video_frames_map),
        len(corpus.frames),
        time.perf_counter() - started,
    )
    try:
        yield
    finally:
        if mps_encoders is not None:
            mps_encoders.close_active()
        if query_cache is not None:
            query_cache.close()
        if corpus is not None:
            corpus.close()


app = FastAPI(title="AIC-2026 MPS Subset Retrieval Workbench", lifespan=lifespan)


def _execution_device_health() -> dict[str, Any]:
    contract = device_contract(os.environ.get("AIC_DEVICE", "auto"), allow_fallback=True)
    return {
        "preferred": contract["preferred"],
        "neural_encoders": contract,
        "vector_cosine_search": {
            "actual": mps_encoders.device if mps_encoders else contract["preferred"],
            "backend": "pytorch-mps-tensor" if contract["preferred"] == "mps" else "pytorch-cpu",
        },
        "lexical": {"actual": "cpu", "backend": "sqlite-fts5"},
    }


@app.get("/api/health")
async def health(compact: bool = False) -> dict[str, Any]:
    ready = kis_fusion_searcher is not None and corpus is not None
    devices = _execution_device_health()
    b1 = branch1_searcher.health() if branch1_searcher else {"ready": False}
    b2 = branch2_searcher.health() if branch2_searcher else {"ready": False}
    b3_ocr = branch3_ocr_searcher.health() if branch3_ocr_searcher else {"ready": False}
    b3_asr = branch3_asr_searcher.health() if branch3_asr_searcher else {"ready": False}
    kf = kis_fusion_searcher.health() if kis_fusion_searcher else {"ready": False}
    payload = {
        "status": "ready" if ready else "starting",
        "device": devices["preferred"],
        "execution_devices": devices,
        "ready": ready,
        "production_ready": False,
        "mode": "mps_subset_demo",
        "indexed_videos": len(corpus.video_frames_map) if corpus else 0,
        "indexed_frames": len(corpus.frames) if corpus else 0,
        "components": {
            "api_process": {"ready": True},
            "qdrant": {"ready": ready, "backend": "mps_subset_tensor_store"},
            "metadata": {"ready": ready},
            "ocr": b3_ocr,
            "asr": b3_asr,
            "image_search": {"ready": bool(corpus and corpus.siglip2_tensor is not None)},
            "siglip_text": {"ready": bool(corpus and corpus.siglip2_tensor is not None)},
            "branch1": b1,
            "branch2": b2,
            "branch3_asr": b3_asr,
            "branch3_ocr": b3_ocr,
            "kis_fusion": kf,
        },
    }
    return (
        payload
        if not compact
        else {
            "status": payload["status"],
            "device": payload["device"],
            "ready": payload["ready"],
            "production_ready": False,
            "components": {
                k: {"ready": v.get("ready", False)} for k, v in payload["components"].items()
            },
        }
    )


@app.get("/api/branch1/health")
async def get_branch1_health() -> dict[str, Any]:
    return branch1_searcher.health() if branch1_searcher else {"status": "starting", "ready": False}


@app.get("/api/branch2/health")
async def get_branch2_health() -> dict[str, Any]:
    return branch2_searcher.health() if branch2_searcher else {"status": "starting", "ready": False}


@app.get("/api/branch3/asr/health")
async def get_branch3_asr_health() -> dict[str, Any]:
    return branch3_asr_searcher.health() if branch3_asr_searcher else {"ready": False}


@app.get("/api/branch3/ocr/health")
async def get_branch3_ocr_health() -> dict[str, Any]:
    return branch3_ocr_searcher.health() if branch3_ocr_searcher else {"ready": False}


@app.get("/api/fusion/kis/health")
async def get_kis_fusion_health() -> dict[str, Any]:
    return (
        kis_fusion_searcher.health()
        if kis_fusion_searcher
        else {"schema_version": "kis.fusion.health.v1", "ready": False}
    )


@app.get("/api/config")
async def config() -> dict[str, Any]:
    devices = _execution_device_health()
    has_siglip = bool(corpus and corpus.siglip2_tensor is not None)
    return {
        "keyframes_root": str(KEYFRAMES_ROOT),
        "experiment_mode": "mps_subset_demo",
        "task_types": ["KIS", "VQA", "TRAKE"],
        "modalities": ["siglip", "dam", "ocr", "asr"],
        "default_top_k": 20,
        "max_top_k": 100,
        "fusion_enabled": True,
        "reranking_enabled": True,
        "query_parser": "deterministic_local",
        "parser_engines": ["local", "direct"],
        "device": devices["preferred"],
        "execution_devices": devices,
        "capabilities": {
            "video_timeline": True,
            "video_visual_fusion": True,
            "image_search": has_siglip,
            "submission_prepare": True,
            "related_frame_fill": True,
            "submission_modes": ["KIS", "VQA", "TRAKE"],
            "ordered_search_expansion": True,
            "discovery_cascade": False,
            "media_info": MEDIA_INFO_ROOT.is_dir(),
            "ocr": True,
            "asr": True,
            "parser_modes": ["local", "direct"],
            "qwen_fallback_control": False,
            "branch1_three_model": True,
            "branch2_dam_hybrid": True,
            "branch3_asr": True,
            "branch3_ocr": True,
            "kis_fusion": True,
            "kis_query_planning": True,
            "kis_ordered_events": True,
        },
        "image_search": {
            "enabled": has_siglip,
            "max_upload_bytes": MAX_IMAGE_UPLOAD_BYTES,
            "max_pixels": MAX_IMAGE_PIXELS,
            "max_dimension": MAX_IMAGE_DIMENSION,
            "content_types": sorted(ALLOWED_IMAGE_CONTENT_TYPES),
            "max_top_k": 100,
        },
    }


@app.post("/api/query/kis/plan")
async def plan_kis_query(request: KisQueryPlanRequest) -> dict[str, Any]:
    try:
        return build_kis_query_plan(
            query=request.query,
            task_type=request.task_type,
            parser=parser,
            query_bundle=(request.query_bundle.model_dump() if request.query_bundle else None),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/parse")
async def parse_query(request: ParseRequest) -> dict[str, Any]:
    if request.task_type != "KIS":
        raise HTTPException(status_code=400, detail="Local mode supports KIS retrieval only")
    started = time.perf_counter()
    parsed = parser.parse(request.query, task_type="KIS")
    return {
        "parsed_query": parsed.model_dump(exclude={"weights"}),
        "parser_engine_requested": request.engine,
        "qwen_fallback_allowed": False,
        "external_llm_used": False,
        "execution_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


async def _run_search(
    fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(fn, *args, **kwargs)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        message = str(error)
        if "BUSY" in message:
            raise HTTPException(status_code=409, detail=message) from error
        LOGGER.exception("Search execution failed")
        raise HTTPException(status_code=503, detail=message) from error
    except Exception as error:
        LOGGER.exception("Unexpected search failure")
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/search/branch1")
async def search_branch1(request: Branch1SearchRequest) -> dict[str, Any]:
    assert branch1_searcher is not None
    return await _run_search(
        branch1_searcher.execute,
        request.query_bundle.model_dump(),
        request.model_weights.normalized(),
        request.per_stream_top_k,
        request.final_top_k,
    )


@app.post("/api/search/branch2")
async def search_branch2(request: Branch2SearchRequest) -> dict[str, Any]:
    assert branch2_searcher is not None
    return await _run_search(
        branch2_searcher.execute,
        request.query_bundle.model_dump(),
        request.hybrid_weights.normalized(),
        request.rerank_weights.normalized(),
        request.per_stream_top_k,
        request.pre_rerank_top_k,
        request.rerank_top_k,
    )


@app.post("/api/search/branch3/asr")
async def search_branch3_asr(request: Branch3AsrSearchRequest) -> dict[str, Any]:
    assert branch3_asr_searcher is not None
    return await _run_search(
        branch3_asr_searcher.execute,
        request.query_bundle.model_dump(),
        request.per_stream_top_k,
        request.final_top_k,
    )


@app.post("/api/search/branch3/ocr")
async def search_branch3_ocr(request: Branch3OcrSearchRequest) -> dict[str, Any]:
    assert branch3_ocr_searcher is not None
    return await _run_search(
        branch3_ocr_searcher.execute,
        request.query_bundle.model_dump(),
        request.per_stream_top_k,
        request.final_top_k,
    )


@app.post("/api/search/fusion/kis")
async def search_kis_fusion(request: KisFusionSearchRequest) -> dict[str, Any]:
    assert kis_fusion_searcher is not None
    return await _run_search(
        kis_fusion_searcher._execute_prechecked,
        request.query_bundle.model_dump(),
        request.branch_weights.normalized(),
    )


@app.post("/api/search/fusion/kis/temporal")
async def search_kis_temporal(request: KisTemporalSearchRequest) -> dict[str, Any]:
    assert kis_fusion_searcher is not None
    temporal_searcher = KisTemporalFusionSearch(kis_fusion_searcher)
    return await _run_search(
        temporal_searcher.execute,
        query_bundle=request.query_bundle.model_dump(),
        events=[event.model_dump() for event in request.events],
        branch_weights=request.branch_weights.normalized(),
        top_k_sequences=request.top_k_sequences,
        max_gap_seconds=request.max_gap_seconds,
        task_type=request.task_type,
        health_already_checked=True,
    )


@app.post("/api/search")
async def search(request: SearchRequest) -> dict[str, Any]:
    assert workbench_search is not None
    started = time.perf_counter()
    parsed: ParsedQuery = request.parsed_query
    top_k = request.top_k
    pools = await run_in_threadpool(
        lambda: {
            "siglip": workbench_search._run_pool(
                modality="siglip",
                display_name="Visual search (MPS)",
                query=parsed.global_scene_en or parsed.original_query,
                query_source="global_scene_en",
                score_type="cosine",
                score_description="SigLIP2 MPS cosine",
                search=lambda: compat_searcher.search_visual(
                    mps_encoders.embed_siglip_text(parsed.global_scene_en or parsed.original_query),
                    top_k=top_k,
                ),
            )
            if (parsed.global_scene_en or parsed.original_query)
            and compat_searcher
            and mps_encoders
            else {},
            "dam": workbench_search._run_pool(
                modality="dam",
                display_name="DAM BM25",
                query=parsed.objects_en or [parsed.original_query],
                query_source="objects_en",
                score_type="bm25",
                score_description="DAM region BM25",
                search=lambda: compat_searcher.search_dam_queries(
                    parsed.objects_en or [parsed.original_query], top_k
                ),
            )
            if compat_searcher
            else {},
            "ocr": workbench_search._run_pool(
                modality="ocr",
                display_name="OCR FTS5",
                query=parsed.ocr_keywords,
                query_source="ocr_keywords",
                score_type="bm25",
                score_description="OCR FTS5",
                search=lambda: branch3_ocr_searcher.execute_single(
                    " ".join(parsed.ocr_keywords), top_k
                ),
            )
            if parsed.ocr_keywords and branch3_ocr_searcher
            else {},
        }
    )
    return {
        "session_id": request.session_id or parsed.session_id or str(uuid.uuid4()),
        "task_type": "KIS",
        "experiment_mode": "mps_subset_demo",
        "fusion_applied": False,
        "reranking_applied": False,
        "total_candidates": sum(p.get("result_count", 0) for p in pools.values()),
        "modality_results": {k: v for k, v in pools.items() if v},
        "execution_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


@app.post("/api/search/image")
async def search_image(
    file: Annotated[UploadFile, File()],
    top_k: Annotated[int, Form(ge=1, le=100)] = 50,
    video_id: Annotated[str | None, Form(max_length=64)] = None,
) -> dict[str, Any]:
    assert mps_encoders is not None and compat_searcher is not None and corpus is not None
    try:
        payload = await file.read(MAX_IMAGE_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if len(payload) > MAX_IMAGE_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image is too large")
    if file.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP are supported")
    try:
        with Image.open(io.BytesIO(payload)) as src:
            src.load()
            image = ImageOps.exif_transpose(src).convert("RGB").copy()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise HTTPException(
            status_code=400, detail="Uploaded image could not be decoded"
        ) from error

    canonical = _canonical_video_id(video_id) if video_id and video_id.strip() else None
    started = time.perf_counter()
    try:
        vector = await run_in_threadpool(mps_encoders.embed_siglip_image, image)
        if canonical:
            results = await run_in_threadpool(
                compat_searcher.search_visual_in_video, vector, canonical, top_k=top_k
            )
            evaluated = compat_searcher.get_video_frame_count(canonical)
        else:
            results = await run_in_threadpool(compat_searcher.search_visual, vector, top_k=top_k)
            evaluated = len(corpus.siglip2_rows_to_point_ids)
    finally:
        image.close()
    elapsed = round((time.perf_counter() - started) * 1000.0, 2)
    return {
        "task_type": "KIS",
        "experiment_mode": "mps_subset_demo",
        "operation": "image_query",
        "query_modality": "image",
        "scope": "video" if canonical else "global",
        "video_id": canonical,
        "evaluated_frames": evaluated,
        "fusion_applied": False,
        "reranking_applied": False,
        "modality_result": {
            "modality": "siglip",
            "display_name": "SigLIP2 image similarity (MPS)",
            "status": "ok",
            "result_count": len(results),
            "execution_time_ms": elapsed,
            "results": results,
        },
        "execution_time_ms": elapsed,
    }


@app.post("/api/video/{video_id}/search/visual-fusion")
async def search_video_visual_fusion(
    video_id: str, request: VideoVisualFusionRequest
) -> dict[str, Any]:
    assert branch1_searcher is not None and corpus is not None
    canonical = _canonical_video_id(video_id)
    evaluated = len(corpus.video_frames_map.get(canonical, ()))
    if evaluated == 0:
        raise HTTPException(status_code=404, detail=f"Video {canonical} not found")
    started = time.perf_counter()
    if request.query_bundle is not None:
        query_bundle = request.query_bundle.model_dump()
    else:
        query_bundle = focus_event_query_bundle(
            build_kis_query_plan(query=request.query, task_type="KIS", parser=parser)[
                "query_bundle"
            ],
            {"order": 1, "description": request.query, "vi": request.query, "en": request.query},
            shared_context_only=False,
        )
    result = await run_in_threadpool(
        branch1_searcher.execute_in_video, query_bundle, canonical, request.top_k
    )
    pool = {
        **result,
        "modality": "visual_fusion",
        "display_name": f"Visual fusion inside {canonical}",
        "status": "ok",
        "reason": "",
        "query": request.query,
        "query_source": "scoped_visual_query",
        "score_type": "weighted_zsigmoid_fusion",
        "evaluated_frames": evaluated,
    }
    return {
        "schema_version": "video.visual-fusion.result.v1",
        "experiment_mode": "mps_subset_demo",
        "operation": "visual_video_drilldown",
        "video_id": canonical,
        "scope_selected_by_user": True,
        "evaluated_frames": evaluated,
        "models": ["siglip2", "metaclip2", "beit3"],
        "model_weights": result.get("weights"),
        "fusion_applied": True,
        "cross_modal_fusion_applied": False,
        "reranking_applied": False,
        "modality_result": pool,
        "execution_time_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


@app.post("/api/discover/dam-to-siglip")
async def discover(_request: DiscoveryCascadeRequest) -> dict[str, Any]:
    raise HTTPException(status_code=501, detail="Discovery cascade is disabled in subset demo")


@app.post("/api/search/temporal-intersection")
async def temporal(_request: TemporalIntersectionRequest) -> dict[str, Any]:
    raise HTTPException(
        status_code=501, detail="Use Ordered KIS Events (/api/search/fusion/kis/temporal)"
    )


@app.post("/api/submission/prepare")
async def submission(request: SubmissionPrepareRequest) -> dict[str, Any]:
    assert metadata_store is not None and source_frame_index is not None
    payload = request.model_dump()
    result = await run_in_threadpool(
        prepare_submission,
        payload,
        frame_lookup=source_frame_index.resolve,
        video_frames_lookup=metadata_store.video_frames,
    )
    result["server_verified"] = True
    return result


@app.post("/api/submission/related-frames")
async def related_frames(request: RelatedFramesRequest) -> dict[str, Any]:
    assert related_frame_searcher is not None and source_frame_index is not None
    canonical = _canonical_video_id(request.video_id)
    seed = await run_in_threadpool(source_frame_index.resolve, canonical, request.frame_idx)
    if seed is None:
        raise HTTPException(status_code=404, detail="Source frame outside verified timeline")
    embedding_seed_idx = int(seed["related_seed_frame_idx"])
    result = await run_in_threadpool(
        related_frame_searcher.execute, canonical, embedding_seed_idx, request.limit
    )
    result["requested_seed"] = seed
    result["embedding_seed"] = {
        "video_id": canonical,
        "frame_idx": embedding_seed_idx,
        "reason": "exact indexed embedding"
        if bool(seed["indexed_keyframe"])
        else "nearest indexed embedding to the exact source frame",
    }
    return result


@app.get("/api/frame/{video_id}/{frame_idx}")
async def exact_frame(video_id: str, frame_idx: int) -> dict[str, Any]:
    assert metadata_store is not None
    canonical = _canonical_video_id(video_id)
    frame = await run_in_threadpool(metadata_store.frame_by_idx, canonical, frame_idx)
    if frame is None:
        raise HTTPException(status_code=404, detail="Exact indexed frame not found")
    return {
        "schema_version": "frame.identity.v1",
        "exact_match": True,
        "keyframe": {**frame, "validation": "canonical"},
    }


@app.get("/api/video/{video_id}/source-frame/{frame_idx}")
async def source_frame(video_id: str, frame_idx: int) -> dict[str, Any]:
    assert source_frame_index is not None
    canonical = _canonical_video_id(video_id)
    frame = await run_in_threadpool(source_frame_index.resolve, canonical, frame_idx)
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame outside video timeline")
    return {
        "schema_version": SourceFrameIndex.frame_schema_version,
        "exact_match": True,
        "source_frame": frame,
    }


@app.get("/api/keyframe/{video_id}/{keyframe_n}")
async def keyframe_detail(video_id: str, keyframe_n: int) -> dict[str, Any]:
    assert metadata_store is not None and corpus is not None
    canonical = _canonical_video_id(video_id)
    detail = await run_in_threadpool(metadata_store.detail, canonical, keyframe_n)
    if detail is None:
        raise HTTPException(status_code=404, detail="Keyframe not found")
    dam_objects = await run_in_threadpool(corpus.dam_for_frame, canonical, int(detail["frame_idx"]))
    return {
        "keyframe": detail,
        "dam_objects": dam_objects,
        "macro_audio_transcript": "",
    }


@app.get("/api/video/{video_id}/keyframes")
async def video_keyframes(video_id: str) -> dict[str, Any]:
    assert metadata_store is not None
    canonical = _canonical_video_id(video_id)
    frames = await run_in_threadpool(metadata_store.video_frames, canonical)
    if not frames:
        raise HTTPException(status_code=404, detail="Video not found")
    return {"video_id": canonical, "total_keyframes": len(frames), "keyframes": frames}


@app.get("/api/video/{video_id}/timeline")
async def video_timeline(video_id: str) -> dict[str, Any]:
    assert source_frame_index is not None
    canonical = _canonical_video_id(video_id)
    timeline = await run_in_threadpool(source_frame_index.timeline, canonical)
    if timeline is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return timeline


@app.get("/api/video/{video_id}/media-info")
async def media_info(video_id: str) -> dict[str, Any]:
    canonical = _canonical_video_id(video_id)
    path = MEDIA_INFO_ROOT / f"{canonical}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Media info for {canonical} not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/keyframes/{video_id}/{filename}")
async def keyframe_image(video_id: str, filename: str) -> FileResponse:
    canonical = _canonical_video_id(video_id)
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid keyframe path")
    target = KEYFRAMES_ROOT / canonical / filename
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Keyframe image not found")
    return FileResponse(target)


if (FRONTEND_DIR / "index.html").is_file():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
