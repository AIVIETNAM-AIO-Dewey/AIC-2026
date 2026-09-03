"""Three-model Branch-1 retrieval, normalization, and auditable fusion."""

from __future__ import annotations

import gc
import threading
import time
from typing import Any, Protocol

import numpy as np

from ...infrastructure.qdrant import QdrantHttpClient
from ...infrastructure.persistent_cache import PersistentQueryEmbeddingCache
from .contracts import (
    BRANCH1_FINAL_TOP_K,
    MODEL_SPECS,
    QUERY_ROLES,
    normalize_model_weights,
)
from .fusion import aggregate_model_streams, fuse_model_candidates

class Branch1Encoder(Protocol):
    revisions: dict[str, str]

    def encode(self, model_name: str, texts: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]: ...

    def unload(self) -> None: ...


class Branch1Search:
    def __init__(
        self,
        qdrant: QdrantHttpClient,
        encoder: Branch1Encoder,
        cache: PersistentQueryEmbeddingCache,
        search_lock: threading.Lock | None = None,
    ) -> None:
        self.qdrant = qdrant
        self.encoder = encoder
        self.cache = cache
        self._search_lock = search_lock or threading.Lock()

    def execute(
        self,
        query_bundle: dict[str, Any],
        weights: dict[str, float],
        per_stream_top_k: int,
        final_top_k: int,
        *,
        _lock_already_held: bool = False,
    ) -> dict[str, Any]:
        if not 1 <= int(per_stream_top_k) <= 2_000:
            raise ValueError("Branch-1 per_stream_top_k must be between 1 and 2000")
        if int(final_top_k) != BRANCH1_FINAL_TOP_K:
            raise ValueError(
                f"Branch-1 final_top_k is fixed at {BRANCH1_FINAL_TOP_K}"
            )
        weights = normalize_model_weights(weights)
        acquired = False
        if not _lock_already_held:
            if not self._search_lock.acquire(blocking=False):
                raise RuntimeError("BRANCH1_SEARCH_BUSY")
            acquired = True
        started = time.perf_counter()
        model_candidates: dict[str, dict[str, dict[str, Any]]] = {}
        timing: dict[str, Any] = {"models": {}}
        diagnostics: dict[str, Any] = {}
        completed = False
        try:
            if not isinstance(query_bundle, dict):
                raise ValueError("Branch-1 query_bundle must be an object")
            if query_bundle.get("schema_version") != "branch1.query.v1":
                raise ValueError("Branch-1 query_bundle.schema_version must be branch1.query.v1")
            queries = query_bundle.get("queries")
            if not isinstance(queries, list) or len(queries) != len(QUERY_ROLES):
                raise ValueError("Branch-1 requires exactly six query variants")
            by_role: dict[str, dict[str, Any]] = {}
            for item in queries:
                if not isinstance(item, dict):
                    raise ValueError("Branch-1 query variants must be objects")
                role = str(item.get("role") or "")
                if role in by_role:
                    raise ValueError("Branch-1 query roles must be unique")
                vi = str(item.get("vi") or "").strip()
                en = str(item.get("en") or "").strip()
                if not vi or not en:
                    raise ValueError("Branch-1 Vietnamese and English query variants must not be empty")
                by_role[role] = {"role": role, "vi": vi, "en": en}
            if set(by_role) != set(QUERY_ROLES):
                raise ValueError("Branch-1 query roles must contain each role exactly once")
            stream_contracts = {
                model_name: [
                    {"role": role, "language": language}
                    for role in QUERY_ROLES
                    for language in spec["languages"]
                ]
                for model_name, spec in MODEL_SPECS.items()
            }
            for model_name, spec in MODEL_SPECS.items():
                model_started = time.perf_counter()
                stream_descriptors = stream_contracts[model_name]
                texts = [
                    str(by_role[item["role"]][item["language"]])
                    for item in stream_descriptors
                ]
                revision = self.encoder.revisions[model_name]
                tokenizer_config = (
                    "languages=vi,en;max_tokens=64;normalization=l2"
                    if model_name == "siglip2"
                    else "languages=vi,en;max_tokens=77;normalization=l2"
                    if model_name == "metaclip2"
                    else "languages=en;max_tokens=64;output=language_head;normalization=l2"
                )
                cache_key = self.cache.key(
                    model_name,
                    revision,
                    texts,
                    tokenizer_config=tokenizer_config,
                    stream_contract=[
                        {
                            "role": descriptor["role"],
                            "language": descriptor["language"],
                            "text": text,
                        }
                        for descriptor, text in zip(stream_descriptors, texts, strict=True)
                    ],
                )
                cached = self.cache.get(cache_key)
                encode_started = time.perf_counter()
                if cached is None:
                    vectors, model_diagnostics = self.encoder.encode(model_name, texts)
                    expected_rows = len(stream_descriptors)
                    if vectors.shape != (expected_rows, int(spec["dimension"])):
                        raise ValueError(
                            f"{model_name} encoder returned {vectors.shape}; "
                            f"expected ({expected_rows}, {spec['dimension']})"
                        )
                    model_diagnostics = [
                        {
                            **dict(diagnostic),
                            "role": descriptor["role"],
                            "language": descriptor["language"],
                            "stream": f"{descriptor['role']}:{descriptor['language']}",
                        }
                        for diagnostic, descriptor in zip(
                            model_diagnostics, stream_descriptors, strict=True
                        )
                    ]
                    self.cache.put(cache_key, model_name, vectors, model_diagnostics)
                    cache_hit = False
                else:
                    vectors, model_diagnostics = cached
                    cache_hit = True
                vectors = np.asarray(vectors, dtype=np.float32)
                expected_shape = (len(stream_descriptors), int(spec["dimension"]))
                if vectors.shape != expected_shape or not np.isfinite(vectors).all():
                    raise ValueError(
                        f"{model_name} cached encoder output is invalid: "
                        f"shape={vectors.shape}, expected={expected_shape}"
                    )
                norms = np.linalg.norm(vectors, axis=1)
                if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-8):
                    raise ValueError(
                        f"{model_name} encoder output contains a zero or non-finite vector"
                    )
                if len(model_diagnostics) != len(stream_descriptors):
                    raise ValueError(
                        f"{model_name} tokenizer diagnostics returned "
                        f"{len(model_diagnostics)} rows; expected {len(stream_descriptors)}"
                    )
                # Cache entries created by the bilingual contract are
                # enriched with the same stream identity as fresh inference.
                # Keep old diagnostic payloads from being silently associated
                # with the wrong language if a stale cache database survives.
                model_diagnostics = [
                    {
                        **dict(diagnostic),
                        "role": descriptor["role"],
                        "language": descriptor["language"],
                        "stream": f"{descriptor['role']}:{descriptor['language']}",
                    }
                    for diagnostic, descriptor in zip(
                        model_diagnostics, stream_descriptors, strict=True
                    )
                ]
                encode_ms = (time.perf_counter() - encode_started) * 1000.0
                worker_timing = dict(getattr(getattr(self.encoder, "manager", None), "last_timing", {})) if not cache_hit else {}
                qdrant_started = time.perf_counter()
                streams = [
                    self.qdrant.query(
                        str(spec["collection"]),
                        str(spec["vector"]),
                        vectors[index],
                        per_stream_top_k,
                    )
                    for index in range(len(stream_descriptors))
                ]
                qdrant_ms = (time.perf_counter() - qdrant_started) * 1000.0
                normalize_started = time.perf_counter()
                stream_keys = tuple(
                    f"{descriptor['role']}:{descriptor['language']}"
                    for descriptor in stream_descriptors
                )
                model_candidates[model_name] = aggregate_model_streams(
                    stream_keys, streams
                )
                normalize_ms = (time.perf_counter() - normalize_started) * 1000.0
                diagnostics[model_name] = model_diagnostics
                self.encoder.unload()
                gc.collect()
                timing["models"][model_name] = {
                    "cache_hit": cache_hit,
                    "encoding_ms": round(encode_ms, 2),
                    "model_loading_ms": worker_timing.get("model_loading_ms", 0.0),
                    "model_inference_ms": worker_timing.get("inference_ms", 0.0),
                    "worker_reused": bool(worker_timing.get("worker_reused", False)),
                    "worker_spawned": bool(worker_timing.get("worker_spawned", not cache_hit)),
                    "worker_pid": worker_timing.get("worker_pid"),
                    "worker_load_count": worker_timing.get("worker_load_count", 0),
                    "qdrant_search_ms": round(qdrant_ms, 2),
                    "normalization_ms": round(normalize_ms, 2),
                    "candidate_count": len(model_candidates[model_name]),
                    "total_ms": round((time.perf_counter() - model_started) * 1000.0, 2),
                }
            fusion_started = time.perf_counter()
            # Keep the public pool bounded even if a custom fusion adapter
            # accidentally returns more rows than the fixed Branch-1 gate.
            results = list(fuse_model_candidates(model_candidates, weights, final_top_k))[
                :BRANCH1_FINAL_TOP_K
            ]
            timing["fusion_ms"] = round((time.perf_counter() - fusion_started) * 1000.0, 2)
            timing["total_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            response = {
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
                "gate_top_k": BRANCH1_FINAL_TOP_K,
                "future_fusion_eligible": True,
                "query_streams": {
                    model_name: [
                        {
                            **descriptor,
                            "stream": f"{descriptor['role']}:{descriptor['language']}",
                        }
                        for descriptor in descriptors
                    ]
                    for model_name, descriptors in stream_contracts.items()
                },
                "stream_count": sum(len(descriptors) for descriptors in stream_contracts.values()),
                "result_count": len(results),
                "tokenizer_diagnostics": diagnostics,
                "timing": timing,
                "results": results,
            }
            completed = True
            return response
        finally:
            if not completed:
                # A Qdrant/normalization failure is not an inference request
                # that should keep a large model resident for the idle grace
                # period.  The worker manager owns the process and can close
                # it without affecting the cache.
                manager = getattr(self.encoder, "manager", None)
                close_active = getattr(manager, "close_active", None)
                if callable(close_active):
                    close_active()
            self.encoder.unload()
            if acquired:
                self._search_lock.release()

    def _execute_locked(
        self,
        query_bundle: dict[str, Any],
        weights: dict[str, float],
        per_stream_top_k: int,
        final_top_k: int,
    ) -> dict[str, Any]:
        """Run the canonical branch while the shared fusion lock is held."""

        return self.execute(
            query_bundle,
            weights,
            per_stream_top_k,
            final_top_k,
            _lock_already_held=True,
        )
