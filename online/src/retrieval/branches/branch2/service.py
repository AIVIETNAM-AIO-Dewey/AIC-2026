"""Branch 2 service: DAM dense + BM25 sparse + BEiT-3 cosine rerank."""

from __future__ import annotations

import gc
import os
import threading
import time
from pathlib import Path
from typing import Any

from ..branch1.contracts import EXPECTED_FRAMES, QUERY_ROLES
from ...infrastructure.persistent_cache import PersistentQueryEmbeddingCache
from ...encoders.cpu import CpuTextEncoders
from ...infrastructure.qdrant import QdrantHttpClient
from .contracts import DEFAULT_PER_STREAM_TOP_K, DEFAULT_PRE_RERANK_TOP_K, DEFAULT_RERANK_TOP_K, normalize_weights
from .dense import DamDenseRetriever
from .fusion import fuse_dense_sparse
from ..rerankers.beit3_cosine import Beit3CosineReranker
from .sparse import DamBm25Index


class Branch2Search:
    def __init__(self, qdrant: QdrantHttpClient, data_root: Path, state_root: Path, bge_encoders: CpuTextEncoders, beit_encoders: Any, cache: PersistentQueryEmbeddingCache, search_lock: threading.Lock | None = None) -> None:
        self.qdrant = qdrant
        self.bge_encoders = bge_encoders
        self.beit_encoders = beit_encoders
        self.cache = cache
        self.data_root = data_root
        self.search_lock = search_lock or threading.Lock()
        self.dense = DamDenseRetriever(qdrant, data_root, state_root)
        self.sparse = DamBm25Index(data_root, state_root)
        self._frame_point_ids: dict[str, int] | None = None
        self._reranker: Beit3CosineReranker | None = None
        self._resource_lock = threading.RLock()

    @staticmethod
    def _load_frame_ids(data_root: Path) -> dict[str, int]:
        import json
        path = data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
        mapping: dict[str, int] = {}
        minimum = EXPECTED_FRAMES + 1
        maximum = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                point_id = int(item["point_id"])
                frame_uid = str(item["frame_uid"])
                derived_uid = f"{item.get('video_id')}:{int(item.get('frame_idx', -1))}"
                if frame_uid != derived_uid:
                    raise ValueError(f"Frame identity mismatch for point {point_id}: {frame_uid!r}")
                if frame_uid in mapping:
                    raise ValueError(f"Duplicate frame_uid in canonical metadata: {frame_uid}")
                mapping[frame_uid] = point_id
                minimum = min(minimum, point_id)
                maximum = max(maximum, point_id)
        if (
            len(mapping) != EXPECTED_FRAMES
            or minimum != 1
            or maximum != EXPECTED_FRAMES
            or len(set(mapping.values())) != EXPECTED_FRAMES
        ):
            raise ValueError("Frame identity map is incomplete")
        return mapping

    def _get_reranker(self) -> Beit3CosineReranker:
        with self._resource_lock:
            if self._reranker is None:
                self._frame_point_ids = self._load_frame_ids(self.data_root)
                self._reranker = Beit3CosineReranker(
                    self.qdrant, self.beit_encoders, self._frame_point_ids
                )
            return self._reranker

    def health(self) -> dict[str, Any]:
        from .health import branch2_health
        return branch2_health(
            self.data_root,
            self.qdrant,
            self.dense,
            self.sparse,
            self.bge_encoders,
            self.beit_encoders,
            self.sparse.state_root,
        )

    def execute(self, query_bundle: dict[str, Any], hybrid_weights: dict[str, float], rerank_weights: dict[str, float], per_stream_top_k: int = DEFAULT_PER_STREAM_TOP_K, pre_rerank_top_k: int = DEFAULT_PRE_RERANK_TOP_K, rerank_top_k: int = DEFAULT_RERANK_TOP_K, *, _lock_already_held: bool = False) -> dict[str, Any]:
        if not 1 <= int(per_stream_top_k) <= DEFAULT_PER_STREAM_TOP_K:
            raise ValueError(
                f"Branch-2 per_stream_top_k must be between 1 and {DEFAULT_PER_STREAM_TOP_K}"
            )
        if int(pre_rerank_top_k) != DEFAULT_PRE_RERANK_TOP_K:
            raise ValueError(
                f"Branch-2 pre_rerank_top_k is fixed at {DEFAULT_PRE_RERANK_TOP_K}"
            )
        if not 1 <= int(rerank_top_k) <= DEFAULT_RERANK_TOP_K:
            raise ValueError(
                f"Branch-2 BEiT-3 rerank_top_k must be in 1..{DEFAULT_RERANK_TOP_K}"
            )
        if int(rerank_top_k) > int(pre_rerank_top_k):
            raise ValueError("rerank_top_k cannot exceed pre_rerank_top_k")
        acquired = False
        if not _lock_already_held:
            if not self.search_lock.acquire(blocking=False):
                raise RuntimeError("BRANCH2_SEARCH_BUSY")
            acquired = True
        started = time.perf_counter()
        timings: dict[str, Any] = {}
        completed = False
        try:
            if not isinstance(query_bundle, dict):
                raise ValueError("Branch-2 query_bundle must be an object")
            if query_bundle.get("schema_version") != "branch1.query.v1":
                raise ValueError("Branch-2 query_bundle.schema_version must be branch1.query.v1")
            queries = query_bundle.get("queries")
            if not isinstance(queries, list) or len(queries) != len(QUERY_ROLES):
                raise ValueError("Branch-2 requires exactly six query variants")
            by_role: dict[str, dict[str, Any]] = {}
            for item in queries:
                if not isinstance(item, dict):
                    raise ValueError("Branch-2 query variants must be objects")
                role = str(item.get("role") or "")
                if role in by_role:
                    raise ValueError("Branch-2 query roles must be unique")
                vi = str(item.get("vi") or "").strip()
                en = str(item.get("en") or "").strip()
                if not vi or not en:
                    raise ValueError("Branch-2 Vietnamese and English query variants must not be empty")
                by_role[role] = {"role": role, "vi": vi, "en": en}
            if set(by_role) != set(QUERY_ROLES):
                raise ValueError("Branch-2 query roles must contain each role exactly once")
            texts = [str(by_role[role]["en"]) for role in QUERY_ROLES]
            bge_revision = str(getattr(self.bge_encoders, "bge_revision", None) or os.environ.get("AIC_BGE_REVISION", "local-cache"))
            cache_key = self.cache.key(
                "bge_m3",
                f"{getattr(self.bge_encoders, 'bge_id', 'BAAI/bge-m3')}@{bge_revision}",
                texts,
                tokenizer_config="max_tokens=512;pooling=cls;normalization=l2",
                stream_contract=[
                    {"role": role, "language": "en", "text": text}
                    for role, text in zip(QUERY_ROLES, texts, strict=True)
                ],
            )
            encode_started = time.perf_counter()
            cached = self.cache.get(cache_key)
            if cached is None:
                if hasattr(self.bge_encoders, "encode_bge_text"):
                    vectors, diagnostics = self.bge_encoders.encode_bge_text(texts)
                else:
                    vectors = self.bge_encoders.embed_bge_text(texts)
                    diagnostics = [{"token_count": None, "max_tokens": 512, "truncated": False} for _ in texts]
                diagnostics = [
                    {
                        **dict(diagnostic),
                        "role": role,
                        "language": "en",
                        "stream": f"{role}:en",
                    }
                    for diagnostic, role in zip(diagnostics, QUERY_ROLES, strict=True)
                ]
                self.cache.put(cache_key, "bge_m3", vectors, diagnostics)
                cache_hit = False
            else:
                vectors, diagnostics = cached
                cache_hit = True
                diagnostics = [
                    {
                        **dict(diagnostic),
                        "role": role,
                        "language": "en",
                        "stream": f"{role}:en",
                    }
                    for diagnostic, role in zip(diagnostics, QUERY_ROLES, strict=True)
                ]
            timings["bge_encoding_ms"] = round((time.perf_counter() - encode_started) * 1000.0, 2)
            bge_worker_timing = dict(getattr(getattr(self.bge_encoders, "manager", None), "last_timing", {})) if not cache_hit else {}
            timings["bge_model_loading_ms"] = bge_worker_timing.get("model_loading_ms", 0.0)
            timings["bge_model_inference_ms"] = bge_worker_timing.get("inference_ms", 0.0)
            timings["bge_worker_reused"] = bool(bge_worker_timing.get("worker_reused", False))
            timings["bge_worker_spawned"] = bool(bge_worker_timing.get("worker_spawned", not cache_hit))
            timings["bge_worker_pid"] = bge_worker_timing.get("worker_pid")
            timings["bge_worker_load_count"] = bge_worker_timing.get("worker_load_count", 0)
            dense_started = time.perf_counter()
            dense = self.dense.search(vectors, QUERY_ROLES, per_stream_top_k)
            timings["dense_ms"] = round((time.perf_counter() - dense_started) * 1000.0, 2)
            timings["dam_qdrant_ms"] = self.dense.last_timing.get("qdrant_ms", 0.0)
            timings["dam_exact_scoring_ms"] = self.dense.last_timing.get("exact_scoring_ms", 0.0)
            sparse_started = time.perf_counter()
            sparse = self.sparse.search(texts, per_stream_top_k)
            timings["sparse_ms"] = round((time.perf_counter() - sparse_started) * 1000.0, 2)
            fusion_started = time.perf_counter()
            candidate_count_before_gate = len(set(dense) | set(sparse))
            hybrid = fuse_dense_sparse(dense, sparse, normalize_weights(hybrid_weights, ("dense", "sparse")), pre_rerank_top_k)
            timings["fusion_ms"] = round((time.perf_counter() - fusion_started) * 1000.0, 2)
            rerank_started = time.perf_counter()
            normalized_rerank_weights = normalize_weights(
                rerank_weights, ("beit3", "previous")
            )
            # There is no BEiT-3 work to perform when DAM dense and BM25
            # produce no candidate.  Apart from saving CPU/RAM, this keeps an
            # empty standalone pool from loading an encoder during a KIS
            # fusion request whose other voters also have no hits.
            beit_cache_hit = False
            beit_diagnostics: list[dict[str, Any]] = []
            reranked = list(hybrid)
            rerank_info: dict[str, Any] = {
                "candidate_count": len(hybrid),
                "rerank_count": 0,
                "weights": normalized_rerank_weights,
                "text_encoder_output": "language_head",
                "query_language": "en",
                "checkpoint_task": "BEiT-3 COCO Retrieval",
                "scoring": "cosine",
                "previous_score_field": "hybrid_score",
                "tokenizer_diagnostics": [],
                "qdrant_ms": 0.0,
                "scoring_ms": 0.0,
            }
            beit_worker_timing: dict[str, Any] = {}
            if hybrid:
                cache_key = self.cache.key(
                    "beit3",
                    self.beit_encoders.revisions["beit3"],
                    texts,
                    # Keep the cache namespace identical to Branch 1 and
                    # final KIS fusion.  The language contract is part of
                    # the encoder identity even though all six Branch-2
                    # streams are English.
                    tokenizer_config="languages=en;max_tokens=64;output=language_head;normalization=l2",
                    stream_contract=[
                        {"role": role, "language": "en", "text": text}
                        for role, text in zip(QUERY_ROLES, texts, strict=True)
                    ],
                )
                cached_beit = self.cache.get(cache_key)
                if cached_beit is None:
                    beit_vectors, beit_diagnostics = self.beit_encoders.encode("beit3", texts)
                    beit_diagnostics = [
                        {
                            **dict(diagnostic),
                            "role": role,
                            "language": "en",
                            "stream": f"{role}:en",
                        }
                        for diagnostic, role in zip(beit_diagnostics, QUERY_ROLES, strict=True)
                    ]
                    self.cache.put(cache_key, "beit3", beit_vectors, beit_diagnostics)
                    beit_cache_hit = False
                else:
                    beit_vectors, beit_diagnostics = cached_beit
                    beit_cache_hit = True
                    beit_diagnostics = [
                        {
                            **dict(diagnostic),
                            "role": role,
                            "language": "en",
                            "stream": f"{role}:en",
                        }
                        for diagnostic, role in zip(beit_diagnostics, QUERY_ROLES, strict=True)
                    ]
                beit_worker_timing = dict(
                    getattr(getattr(self.beit_encoders, "manager", None), "last_timing", {})
                ) if not beit_cache_hit else {}
                # Reranker accepts the text list and intentionally uses the
                # same COCO retrieval checkpoint.
                reranked, rerank_info = self._get_reranker().rerank(
                    hybrid,
                    texts,
                    top_k=rerank_top_k,
                    weights=normalized_rerank_weights,
                    text_vectors=beit_vectors,
                    tokenizer_diagnostics=beit_diagnostics,
                )
            timings["beit_model_loading_ms"] = beit_worker_timing.get("model_loading_ms", 0.0)
            timings["beit_model_inference_ms"] = beit_worker_timing.get("inference_ms", 0.0)
            timings["beit_worker_reused"] = bool(beit_worker_timing.get("worker_reused", False))
            timings["beit_worker_spawned"] = bool(beit_worker_timing.get("worker_spawned", not beit_cache_hit)) if hybrid else False
            timings["beit_worker_pid"] = beit_worker_timing.get("worker_pid")
            timings["beit_worker_load_count"] = beit_worker_timing.get("worker_load_count", 0)
            # ``pre_rerank_top_k`` is a public output gate.  Keep this
            # boundary defensive for injected/custom rerankers as well as
            # the canonical implementation (which already returns <=500).
            reranked = list(reranked)[:DEFAULT_PRE_RERANK_TOP_K]
            timings["rerank_ms"] = round((time.perf_counter() - rerank_started) * 1000.0, 2)
            timings["beit_cache_hit"] = beit_cache_hit
            if hybrid:
                self.beit_encoders.unload()
            gc.collect()
            timings["total_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            completed = True
            return {
                "schema_version": "branch2.result.v1",
                "fusion_applied": True,
                "reranking_applied": bool(reranked),
                "hybrid_weights": normalize_weights(hybrid_weights, ("dense", "sparse")),
                "rerank_weights": normalized_rerank_weights,
                "per_stream_top_k": per_stream_top_k,
                "pre_rerank_top_k": pre_rerank_top_k,
                "rerank_top_k": rerank_top_k,
                "result_count": len(reranked),
                "candidate_count_before_gate": candidate_count_before_gate,
                "gate_top_k": DEFAULT_PRE_RERANK_TOP_K,
                "future_fusion_eligible": True,
                "query_streams": {
                    "dam_dense": [
                        {"role": role, "language": "en", "stream": f"{role}:en"}
                        for role in QUERY_ROLES
                    ],
                    "bm25_sparse": [
                        {"role": role, "language": "en", "stream": f"{role}:en"}
                        for role in QUERY_ROLES
                    ],
                    "beit3_rerank": [
                        {"role": role, "language": "en", "stream": f"{role}:en"}
                        for role in QUERY_ROLES
                    ],
                },
                "stream_count": 18,
                "dense_candidate_count": len(dense),
                "sparse_candidate_count": len(sparse),
                "timing": {**timings, "bge_cache_hit": cache_hit, "beit_cache_hit": beit_cache_hit},
                "tokenizer_diagnostics": {"bge_m3": diagnostics, "beit3": beit_diagnostics},
                "bge_tokenizer_diagnostics": diagnostics,
                "beit3_tokenizer_diagnostics": beit_diagnostics,
                "rerank": rerank_info,
                "results": reranked,
            }
        finally:
            if not completed:
                self.bge_encoders.unload_all()
                self.beit_encoders.unload()
            gc.collect()
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
        """Run Branch 2 while the shared fusion lock is already held."""

        return self.execute(
            query_bundle,
            hybrid_weights,
            rerank_weights,
            per_stream_top_k,
            pre_rerank_top_k,
            rerank_top_k,
            _lock_already_held=True,
        )
