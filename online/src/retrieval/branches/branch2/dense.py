"""DAM dense retrieval with six English-query LSE and max region-to-frame pooling."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ...infrastructure.qdrant import QdrantHttpClient, base_frame
from ..branch1.contracts import EXPECTED_FRAMES
from .contracts import EXPECTED_DAM_REGIONS


def normalized_lse(scores: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("LSE scores must be a non-empty 2D matrix")
    if not np.isfinite(values).all():
        raise ValueError("LSE scores must be finite")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("LSE temperature must be positive and finite")
    scaled = values / float(temperature)
    maximum = scaled.max(axis=1)
    result = float(temperature) * (
        maximum + np.log(np.exp(scaled - maximum[:, None]).sum(axis=1))
    )
    return result - float(temperature) * math.log(values.shape[1])


class DamDenseRetriever:
    MANIFEST_SCHEMA_VERSION = "branch2.dam.v2"
    def __init__(self, qdrant: QdrantHttpClient, data_root: Path, state_root: Path, *, temperature: float = 0.05) -> None:
        if not math.isfinite(float(temperature)) or float(temperature) <= 0:
            raise ValueError("DAM LSE temperature must be positive and finite")
        self.qdrant = qdrant
        self.temperature = float(temperature)
        dense_dir = data_root / "dense_text_embeddings"
        self.matrix_path = dense_dir / "dam_vectors.f16.npy"
        self.metadata_path = dense_dir / "dam_metadata.jsonl"
        self.manifest_path = state_root / "branch2_dam_manifest.json"
        self._matrix: np.memmap | None = None
        self._lock = threading.RLock()
        self.last_timing: dict[str, float] = {}

    def _open_matrix(self) -> np.memmap:
        with self._lock:
            if self._matrix is None:
                matrix = np.load(self.matrix_path, mmap_mode="r", allow_pickle=False)
                if matrix.shape != (EXPECTED_DAM_REGIONS, 1024) or matrix.dtype != np.float16:
                    raise ValueError(f"Invalid DAM matrix: shape={matrix.shape}, dtype={matrix.dtype}")
                self._matrix = matrix
            return self._matrix

    def health(self) -> dict[str, Any]:
        warnings: list[str] = []
        try:
            matrix = np.load(self.matrix_path, mmap_mode="r", allow_pickle=False)
            shape = tuple(matrix.shape)
            dtype = matrix.dtype
        except (OSError, ValueError) as error:
            return {"ready": False, "production_ready": False, "error": str(error)}
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            manifest = {}
        if not isinstance(manifest, dict):
            manifest = {}
        identity = manifest.get("offline_identity") or {}
        identity_recorded = (
            isinstance(identity, dict)
            and identity.get("revision_verified") is not None
            and bool(identity.get("evidence"))
        )
        revision_verified = identity_recorded and identity.get("revision_verified") is True
        if not revision_verified:
            warnings.append("DAM BGE-M3 offline revision is not verified")
        try:
            metadata_stat = self.metadata_path.stat()
            matrix_stat = self.matrix_path.stat()
            files_match_manifest = (
                metadata_stat.st_size == int(manifest.get("metadata_size", -1))
                and metadata_stat.st_mtime_ns == int(manifest.get("metadata_mtime_ns", -1))
                and matrix_stat.st_size == int(manifest.get("matrix_size", -1))
                and matrix_stat.st_mtime_ns == int(manifest.get("matrix_mtime_ns", -1))
            )
        except (OSError, TypeError, ValueError):
            files_match_manifest = False
        ready = (
            manifest.get("passed") is True
            and manifest.get("status") == "ready"
            and manifest.get("schema_version") == self.MANIFEST_SCHEMA_VERSION
            and shape == (EXPECTED_DAM_REGIONS, 1024)
            and dtype == np.float16
            and int(manifest.get("vector_count", 0)) == EXPECTED_DAM_REGIONS
            and int(manifest.get("metadata_count", 0)) == EXPECTED_DAM_REGIONS
            and manifest.get("model_id") == "BAAI/bge-m3"
            and manifest.get("pooling") == "cls"
            and manifest.get("normalization") == "l2"
            and manifest.get("l2_normalized") is True
            and int(manifest.get("dimension", 0)) == 1024
            and manifest.get("dtype") == "float16"
            and manifest.get("finite_verified") is True
            and manifest.get("frame_mapping_verified") is True
            and manifest.get("region_identity_verified") is True
            and identity_recorded
            and files_match_manifest
            and bool(manifest.get("metadata_sha256"))
            and bool(manifest.get("matrix_sha256"))
            and bool(manifest.get("frame_metadata_sha256"))
        )
        return {
            "ready": ready,
            "production_ready": ready and revision_verified,
            "regions": int(manifest.get("metadata_count", 0)),
            "dimension": int(shape[1]) if len(shape) == 2 else 0,
            "dtype": str(dtype),
            "finite": manifest.get("finite_verified") is True,
            "temperature": self.temperature,
            "revision_verified": revision_verified,
            "offline_identity": identity if isinstance(identity, dict) else {},
            "online_revision": manifest.get("online_revision"),
            "frame_metadata_count": int(manifest.get("frame_metadata_count", 0)),
            "frame_metadata_size": int(manifest.get("frame_metadata_size", -1)),
            "frame_metadata_mtime_ns": int(manifest.get("frame_metadata_mtime_ns", -1)),
            "frame_metadata_identity_verified": manifest.get(
                "frame_metadata_identity_verified"
            ) is True,
            # Branch-2 health uses these source fingerprints to bind the DAM
            # preparation manifest to the Qdrant ingestion manifest.
            "matrix_sha256": manifest.get("matrix_sha256"),
            "metadata_sha256": manifest.get("metadata_sha256"),
            "frame_metadata_sha256": manifest.get("frame_metadata_sha256"),
            "manifest": str(self.manifest_path),
            "warnings": warnings,
            "files_match_manifest": files_match_manifest,
        }

    def search(self, query_vectors: np.ndarray, query_roles: tuple[str, ...], top_k: int) -> dict[str, dict[str, Any]]:
        if len(query_roles) != 6 or not 1 <= int(top_k) <= 2_000:
            raise ValueError("DAM dense retrieval requires six roles and top_k in 1..2000")
        vectors = np.asarray(query_vectors, dtype=np.float32)
        if vectors.shape != (len(query_roles), 1024):
            raise ValueError(f"Expected ({len(query_roles)}, 1024) BGE query matrix, got {vectors.shape}")
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("BGE query vectors must be finite and non-zero")
        # Qdrant scores cosine, so the exact mmap path must use the same
        # normalized geometry even if a caller supplies an unnormalized
        # encoder output.
        vectors = vectors / np.maximum(
            np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12
        )
        streams: list[list[dict[str, Any]]] = []
        qdrant_started = time.perf_counter()
        for vector in vectors:
            streams.append(self.qdrant.query("aic_dam_regions", "dam", vector, top_k))
        qdrant_ms = (time.perf_counter() - qdrant_started) * 1000.0
        exact_started = time.perf_counter()
        region_ids = {int(point["id"]) for stream in streams for point in stream}
        if not region_ids:
            self.last_timing = {"qdrant_ms": round(qdrant_ms, 2), "exact_scoring_ms": 0.0}
            return {}
        ordered_ids = sorted(region_ids)
        if ordered_ids[0] < 1 or ordered_ids[-1] > EXPECTED_DAM_REGIONS:
            raise ValueError("Qdrant DAM point IDs are not aligned with the offline DAM matrix")
        rows = np.asarray(self._open_matrix()[np.asarray(ordered_ids, dtype=np.int64) - 1], dtype=np.float32)
        if not np.isfinite(rows).all():
            raise ValueError("Candidate DAM vectors contain non-finite values")
        rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
        scores = rows @ vectors.T
        lse = normalized_lse(scores, self.temperature)
        region_points = {
            int(point["id"]): point
            for stream in streams
            for point in stream
        }
        frames: dict[int, dict[str, Any]] = {}
        for index, point_id in enumerate(ordered_ids):
            point = region_points[point_id]
            payload = dict(point.get("payload") or {})
            try:
                parent_id = int(payload["parent_point_id"])
                video_id = str(payload["video_id"])
                frame_idx = int(payload["frame_idx"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"DAM point {point_id} is missing its canonical frame mapping"
                ) from error
            if not 1 <= parent_id <= EXPECTED_FRAMES:
                raise RuntimeError(
                    f"DAM point {point_id} has an invalid parent_point_id {parent_id}"
                )
            if f"{video_id}:{frame_idx}" != str(
                payload.get("frame_uid") or f"{video_id}:{frame_idx}"
            ):
                raise RuntimeError(f"DAM point {point_id} has inconsistent frame identity")
            query_scores = {role: float(scores[index, role_index]) for role_index, role in enumerate(query_roles)}
            best_query = max(query_scores, key=query_scores.get)
            candidate = frames.setdefault(parent_id, {"region_count": 0, "regions": []})
            candidate["region_count"] += 1
            candidate["regions"].append({
                "point_id": point_id,
                "region_id": payload.get("region_id"),
                "class_entity": payload.get("class_entity", ""),
                "bbox": payload.get("bbox", []),
                "description_en": payload.get("description_en", ""),
                "query_scores": query_scores,
                "best_query_role": best_query,
                "best_query_language": "en",
                "lse_score": float(lse[index]),
            })
        selected = sorted(
            ((max(region["lse_score"] for region in value["regions"]), parent_id, value) for parent_id, value in frames.items()),
            key=lambda item: (-item[0], item[1]),
        )[:top_k]
        payloads = self.qdrant.retrieve("aic_frames", [parent_id for _, parent_id, _ in selected])
        expected_parent_ids = {parent_id for _, parent_id, _ in selected}
        if set(payloads) != expected_parent_ids:
            missing = sorted(expected_parent_ids - set(payloads))
            raise RuntimeError(
                f"DAM dense retrieval is missing {len(missing)} parent frame payloads"
            )
        output: dict[str, dict[str, Any]] = {}
        for rank, (score, parent_id, value) in enumerate(selected, 1):
            payload = payloads.get(parent_id)
            if payload is None:  # defensive; the set check above should catch it
                raise RuntimeError(f"Missing parent frame payload for point {parent_id}")
            payload_uid = str(payload.get("frame_uid") or "")
            derived_uid = f"{payload.get('video_id')}:{int(payload.get('frame_idx', -1))}"
            if not payload_uid or payload_uid != derived_uid:
                raise RuntimeError(
                    f"Parent frame {parent_id} has inconsistent frame_uid {payload_uid!r}"
                )
            winner = max(value["regions"], key=lambda item: item["lse_score"])
            frame = base_frame(payload, score=score, rank=rank, score_type="dam_lse_max")
            frame.update({
                "frame_uid": str(payload["frame_uid"]),
                "global_idx": parent_id,
                "dense_raw": float(score),
                "dense_observed": True,
                "dam_region_count": value["region_count"],
                "dam_winner": winner,
                "dense_query_scores": winner["query_scores"],
                "dense_best_query_role": winner["best_query_role"],
                "dense_best_query_language": winner["best_query_language"],
                "dense_rank": rank,
            })
            output[frame["frame_uid"]] = frame
        self.last_timing = {
            "qdrant_ms": round(qdrant_ms, 2),
            "exact_scoring_ms": round((time.perf_counter() - exact_started) * 1000.0, 2),
        }
        return output
