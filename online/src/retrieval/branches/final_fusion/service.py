"""Canonical KIS cross-branch fusion orchestration.

The service deliberately calls the four standalone branches in-process.  It
holds the shared heavy-search lock for the complete operation, so a fusion
request cannot deadlock by acquiring one of the branch locks again and a
standalone request cannot interleave with a partially-built fusion pool.
"""

from __future__ import annotations

import gc
import json
import math
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from ...infrastructure.resources import current_process_rss_bytes, resource_qualification
from ...infrastructure.scoring import normalize_scores
from ..branch1.contracts import BRANCH1_FINAL_TOP_K, QUERY_ROLES
from ..branch1.health import branch1_health
from ..rerankers.beit3_cosine import Beit3CosineReranker
from .contracts import (
    BRANCH2_PER_STREAM_TOP_K,
    BRANCH2_PRE_RERANK_TOP_K,
    BRANCH2_RERANK_TOP_K,
    BRANCH_POOL_LIMITS,
    DEFAULT_BRANCH_WEIGHTS,
    FINAL_FUSION_RESULT_SCHEMA_VERSION,
    FINAL_RERANK_TOP_K,
    FINAL_TOP_K,
    RRF_K,
    normalize_branch_weights,
)
from .provenance import materialize_fusion_candidate
from .rrf import fuse_branch_pools


class KisFusionSearch:
    """Run Branch 1/2 and Branch 3 ASR/OCR as one KIS search."""

    def __init__(
        self,
        branch1: Any,
        branch2: Any,
        asr: Any,
        ocr: Any,
        *,
        data_root: Path,
        state_root: Path,
        search_lock: threading.Lock | None = None,
        branch1_health_fn: Callable[..., dict[str, Any]] = branch1_health,
    ) -> None:
        self.branch1 = branch1
        self.branch2 = branch2
        self.asr = asr
        self.ocr = ocr
        self.data_root = data_root
        self.state_root = state_root
        self.search_lock = search_lock or threading.Lock()
        self._branch1_health = branch1_health_fn
        self._reranker: Beit3CosineReranker | None = None
        self._frame_point_ids: dict[str, int] | None = None
        self._resource_lock = threading.RLock()

    @staticmethod
    def _safe_health(service: Any) -> dict[str, Any]:
        if service is None:
            return {
                "ready": False,
                "production_ready": False,
                "status": "not_ready",
                "fail_closed": True,
                "error": "service is not initialized",
            }
        try:
            value = service.health()
        except Exception as error:
            return {
                "ready": False,
                "production_ready": False,
                "status": "not_ready",
                "fail_closed": True,
                "error": str(error),
            }
        if not isinstance(value, dict):
            return {
                "ready": False,
                "production_ready": False,
                "status": "not_ready",
                "fail_closed": True,
                "error": "invalid health response",
            }
        payload = dict(value)
        payload.setdefault("ready", False)
        payload.setdefault("production_ready", False)
        payload.setdefault("status", "ready" if payload.get("ready") is True else "not_ready")
        payload.setdefault("fail_closed", payload.get("ready") is not True)
        return payload

    def health(self) -> dict[str, Any]:
        """Return a fail-closed aggregate without executing any branch search."""

        try:
            # ``Branch1Search`` intentionally keeps its readiness policy in
            # the shared ``branch1_health`` function, while lightweight test
            # adapters may expose a regular ``health()`` method instead.  Do
            # not dereference implementation-specific attributes before the
            # injected health function gets a chance to handle an adapter.
            branch1_qdrant = getattr(self.branch1, "qdrant", None)
            branch1_encoder = getattr(self.branch1, "encoder", None)
            branch1_health_method = getattr(self.branch1, "health", None)
            if callable(branch1_health_method):
                branch1_state = self._safe_health(self.branch1)
            else:
                branch1_state = self._branch1_health(
                    self.data_root,
                    branch1_qdrant,
                    branch1_encoder,
                    self.state_root,
                )
        except Exception as error:
            branch1_state = {
                "ready": False,
                "production_ready": False,
                "status": "not_ready",
                "fail_closed": True,
                "error": str(error),
            }
        if not isinstance(branch1_state, dict):
            branch1_state = {"ready": False, "error": "invalid Branch-1 health response"}
        branch2_state = self._safe_health(self.branch2)
        asr_state = self._safe_health(self.asr)
        ocr_state = self._safe_health(self.ocr)

        canonical_path = (
            self.data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
        )
        raw_branch2_components = branch2_state.get("components")
        # A malformed or absent component map must never be interpreted as a
        # green dependency.  Fusion specifically needs the concrete
        # frame-mapping and BEiT-3 collection/ingestion/text-encoder checks;
        # an aggregate Branch-2 ``ready`` flag is not sufficient evidence.
        branch2_components = (
            raw_branch2_components if isinstance(raw_branch2_components, dict) else {}
        )

        def component_is_ready(name: str) -> bool:
            value = branch2_components.get(name)
            return isinstance(value, dict) and value.get("ready") is True

        mapping_component = branch2_components.get("frame_mapping")
        mapping_component_present = bool(branch2_components)
        mapping_state = {
            "ready": bool(
                canonical_path.is_file()
                and mapping_component_present
                and isinstance(mapping_component, dict)
                and mapping_component.get("ready") is True
            ),
            "path": str(canonical_path),
        }
        shared_cache = getattr(self.branch1, "cache", None)
        if shared_cache is None:
            shared_cache = getattr(self.branch2, "cache", None)
        cache_state = {
            "ready": shared_cache is not None
            and all(
                callable(getattr(shared_cache, method, None)) for method in ("key", "get", "put")
            ),
            "persistent": True,
        }
        # Branch 2 health owns the exact Qdrant collection and BEiT text
        # encoder checks.  Keep a compact diagnostic here rather than issuing
        # a second Qdrant scan.
        beit_state = {
            "ready": (
                all(
                    component_is_ready(name)
                    for name in ("beit3_collection", "beit3_ingestion", "beit3_text_encoder")
                )
            ),
            "source": "branch2 health",
        }
        execution_state = {
            branch: callable(getattr(service, "_execute_locked", None))
            for branch, service in (
                ("branch1", self.branch1),
                ("branch2", self.branch2),
                ("asr", self.asr),
                ("ocr", self.ocr),
            )
        }
        execution_contract_state = {
            "ready": all(execution_state.values()),
            "branches": execution_state,
        }
        try:
            resource_state = resource_qualification(self.state_root)
            if not isinstance(resource_state, dict):
                raise ValueError("invalid resource qualification response")
        except Exception as error:
            # Health is an observability boundary.  A malformed or partially
            # written qualification manifest must make fusion not-ready, not
            # turn the health endpoint into a 500 (or a false green state).
            resource_state = {
                "ready": False,
                "production_ready": False,
                "fail_closed": True,
                "error": str(error),
            }
        states = {
            "branch1": branch1_state,
            "branch2": branch2_state,
            "asr": asr_state,
            "ocr": ocr_state,
            "canonical_frame_mapping": mapping_state,
            "beit3": beit_state,
            "query_cache": cache_state,
            "resource_qualification": resource_state,
            "execution_contract": execution_contract_state,
        }
        ready = all(
            states[name].get("ready") is True
            for name in (
                "branch1",
                "branch2",
                "asr",
                "ocr",
                "canonical_frame_mapping",
                "beit3",
                "query_cache",
                "execution_contract",
            )
        )
        # ``ready`` is the operational search gate.  Production qualification
        # is deliberately stricter: every voter must explicitly attest its
        # own production state and the measured resource report must be valid.
        # Do not fall back to ``ready`` here; an adapter that omits provenance
        # must remain fail-closed.
        production_ready = (
            ready
            and resource_state.get("production_ready") is True
            and all(
                states[name].get("production_ready") is True
                for name in ("branch1", "branch2", "asr", "ocr")
            )
        )
        return {
            "schema_version": "kis.fusion.health.v1",
            "branch": "final_fusion",
            "task_type": "KIS",
            "status": "ready" if ready else "not_ready",
            "ready": ready,
            "required": False,
            "production_ready": production_ready,
            "fail_closed": True,
            "rrf_k": RRF_K,
            "final_top_k": FINAL_TOP_K,
            "rerank_top_k": FINAL_RERANK_TOP_K,
            "branch_pool_limits": dict(BRANCH_POOL_LIMITS),
            "beit3_weight": 0.25,
            "rrf_weight": 0.75,
            "branch_weights": dict(DEFAULT_BRANCH_WEIGHTS),
            "components": states,
            "api_rss_bytes": current_process_rss_bytes(),
            "resource_qualification": resource_state,
            "warnings": ["KIS fusion requires all four branch pools to be ready"]
            if not ready
            else [],
        }

    @staticmethod
    def _query_texts(query_bundle: dict[str, Any]) -> tuple[dict[str, dict[str, str]], list[str]]:
        if not isinstance(query_bundle, dict):
            raise ValueError("KIS query_bundle must be an object")
        if query_bundle.get("schema_version") != "branch1.query.v1":
            raise ValueError("KIS query_bundle.schema_version must be branch1.query.v1")
        queries = query_bundle.get("queries")
        if not isinstance(queries, list) or len(queries) != len(QUERY_ROLES):
            raise ValueError("KIS requires exactly six query variants")
        by_role: dict[str, dict[str, str]] = {}
        for item in queries:
            if not isinstance(item, dict):
                raise ValueError("KIS query variants must be objects")
            role_value = item.get("role")
            if not isinstance(role_value, str):
                raise ValueError("KIS query role must be a string")
            role = role_value.strip()
            if role in by_role:
                raise ValueError("KIS query roles must be unique")
            vi_value = item.get("vi")
            en_value = item.get("en")
            if not isinstance(vi_value, str) or not isinstance(en_value, str):
                raise ValueError("KIS Vietnamese and English query variants must be strings")
            vi = vi_value.strip()
            en = en_value.strip()
            if not vi or not en:
                raise ValueError("KIS Vietnamese and English query variants must not be empty")
            by_role[role] = {"vi": vi, "en": en}
        if set(by_role) != set(QUERY_ROLES):
            raise ValueError("KIS query roles must contain each role exactly once")
        return by_role, [by_role[role]["en"] for role in QUERY_ROLES]

    def _load_frame_point_ids(self) -> dict[str, int]:
        with self._resource_lock:
            if self._frame_point_ids is not None:
                return self._frame_point_ids
            path = self.data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
            mapping: dict[str, int] = {}
            point_ids: set[int] = set()
            minimum = BRANCH_POOL_LIMITS["branch1"] + 1
            maximum = 0
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    uid = str(item.get("frame_uid") or "")
                    video_id = str(item.get("video_id") or "")
                    frame_idx = int(item.get("frame_idx", -1))
                    point_id = int(item.get("point_id", 0))
                    if not uid or uid != f"{video_id}:{frame_idx}" or point_id < 1:
                        raise ValueError(f"Invalid canonical frame at line {line_number}")
                    if uid in mapping or point_id in point_ids:
                        raise ValueError(
                            f"Duplicate canonical frame identity at line {line_number}"
                        )
                    mapping[uid] = point_id
                    point_ids.add(point_id)
                    minimum = min(minimum, point_id)
                    maximum = max(maximum, point_id)
            expected = 247_956
            if len(mapping) != expected or minimum != 1 or maximum != expected:
                raise ValueError("Canonical frame point map is incomplete")
            self._frame_point_ids = mapping
            return mapping

    def _beit_encoder(self) -> Any:
        """Return the canonical BEiT-3 text encoder used by both branches.

        Branch 2 owns the explicit BEiT encoder in the production wiring.  A
        few lightweight adapters expose the same encoder through Branch 1,
        so retain that fallback without making final fusion depend on a
        Branch-1 implementation detail.
        """

        # Treat an adapter's malformed ``beit_encoders`` attribute as absent
        # rather than allowing it to mask a valid Branch-1 encoder fallback.
        # Production wiring supplies the Branch-2 encoder, while this keeps
        # the canonical service usable with the small test adapters used by
        # the API contract suite.
        for owner, attribute in (
            (self.branch2, "beit_encoders"),
            (self.branch1, "encoder"),
        ):
            encoder = getattr(owner, attribute, None)
            if callable(getattr(encoder, "encode", None)):
                return encoder
        raise RuntimeError("BEiT-3 text encoder is not initialized")

    def _query_cache(self) -> Any:
        """Return the shared persistent query cache, if configured."""

        cache = getattr(self.branch1, "cache", None)
        if cache is None:
            cache = getattr(self.branch2, "cache", None)
        return cache

    def _get_reranker(self) -> Beit3CosineReranker:
        with self._resource_lock:
            if self._reranker is None:
                # Branch 2 constructs the same canonical map immediately
                # before its internal rerank. Reuse it when available so a
                # fusion request does not read 247,956 metadata rows twice.
                shared_mapping = getattr(self.branch2, "_frame_point_ids", None)
                if isinstance(shared_mapping, dict) and shared_mapping:
                    self._frame_point_ids = shared_mapping
                qdrant = None
                for owner in (self.branch2, self.branch1):
                    candidate = getattr(owner, "qdrant", None)
                    if callable(getattr(candidate, "query", None)):
                        qdrant = candidate
                        break
                if qdrant is None:
                    raise RuntimeError("BEiT-3 Qdrant client is not initialized")
                self._reranker = Beit3CosineReranker(
                    qdrant,
                    self._beit_encoder(),
                    self._frame_point_ids or self._load_frame_point_ids(),
                )
            return self._reranker

    def _encode_beit_queries(
        self,
        texts: list[str],
    ) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
        encoder = self._beit_encoder()
        cache = self._query_cache()
        raw_revisions = getattr(encoder, "revisions", None)
        revisions = raw_revisions if isinstance(raw_revisions, dict) else {}
        revision = str(revisions.get("beit3", "unknown-revision"))
        stream_contract = [
            {"role": role, "language": "en", "text": text}
            for role, text in zip(QUERY_ROLES, texts, strict=True)
        ]

        def cache_key_for(device: str) -> str:
            if cache is None:
                raise RuntimeError("Query cache is unavailable")
            return cache.key(
                "beit3",
                revision,
                texts,
                tokenizer_config="languages=en;max_tokens=64;output=language_head;normalization=l2",
                stream_contract=stream_contract,
                device=device,
            )

        if cache is not None:
            cache_device_for_model = getattr(encoder, "cache_device_for_model", None)
            lookup_device = (
                str(cache_device_for_model("beit3"))
                if callable(cache_device_for_model)
                else str(getattr(encoder, "cache_device", "cpu"))
            )
            cached = cache.get(cache_key_for(lookup_device))
        else:
            cache_device_for_model = None
            lookup_device = "cpu"
            cached = None
        if cached is not None:
            vectors, diagnostics = cached
            return (
                vectors,
                self._tag_diagnostics(diagnostics),
                {
                    "cache_hit": True,
                    "model_loading_ms": 0.0,
                    "inference_ms": 0.0,
                    "worker_reused": False,
                    "worker_spawned": False,
                    "worker_pid": None,
                    "worker_load_count": 0,
                },
            )
        started = time.perf_counter()
        vectors, diagnostics = encoder.encode("beit3", texts)
        diagnostics = self._tag_diagnostics(diagnostics)
        if cache is not None:
            actual_device = (
                str(cache_device_for_model("beit3"))
                if callable(cache_device_for_model)
                else lookup_device
            )
            cache.put(cache_key_for(actual_device), "beit3", vectors, diagnostics)
        timing = dict(getattr(getattr(encoder, "manager", None), "last_timing", {}))
        timing.update(
            {
                "cache_hit": False,
                "encoding_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "worker_reused": bool(timing.get("worker_reused", False)),
                "worker_spawned": bool(timing.get("worker_spawned", True)),
            }
        )
        return vectors, diagnostics, timing

    @staticmethod
    def _tag_diagnostics(diagnostics: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        values = list(diagnostics or [])
        if len(values) != len(QUERY_ROLES):
            raise ValueError("BEiT-3 tokenizer diagnostics must contain six rows")
        return [
            {
                **dict(value),
                "role": role,
                "language": "en",
                "stream": f"{role}:en",
            }
            for value, role in zip(values, QUERY_ROLES, strict=True)
        ]

    @staticmethod
    def _call_locked(service: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        method = getattr(service, "_execute_locked", None)
        if not callable(method):
            raise RuntimeError("KIS branch does not expose canonical _execute_locked")
        value = method(*args, **kwargs)
        if not isinstance(value, dict):
            raise RuntimeError("KIS branch returned an invalid response")
        return value

    @staticmethod
    def _validate_final_beit_output(
        original: list[dict[str, Any]],
        reranked: Any,
        rerank_info: Any,
        rerank_count: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Fail closed when a final-rerank adapter violates the KIS boundary.

        ``Beit3CosineReranker`` is the production implementation, but the
        service also accepts an injectable adapter in tests and API wiring.
        The adapter may reorder only the first 100 RRF candidates; it cannot
        widen, delete, duplicate, or mutate the canonical final pool.
        """

        if not isinstance(reranked, list):
            raise ValueError("final BEiT reranker must return a candidate list")
        if not isinstance(rerank_info, dict):
            raise ValueError("final BEiT reranker must return an info object")
        if not 1 <= rerank_count <= min(FINAL_RERANK_TOP_K, len(original)):
            raise ValueError("final BEiT rerank count is outside the canonical gate")
        if len(original) > FINAL_TOP_K:
            raise ValueError("RRF output exceeds the final KIS gate")
        if len(reranked) != len(original):
            raise ValueError(
                "final BEiT reranker changed the candidate count "
                f"({len(original)} expected, {len(reranked)} received)"
            )
        if rerank_info.get("candidate_count") != len(original):
            raise ValueError("final BEiT reranker reported an invalid candidate count")
        if rerank_info.get("rerank_count") != rerank_count:
            raise ValueError("final BEiT reranker reported an invalid rerank count")

        canonical_fields = (
            "frame_uid",
            "point_id",
            "global_idx",
            "video_id",
            "frame_idx",
            "keyframe_n",
            "pts_time_s",
            "fps",
            "image_relpath",
            "submission_string",
        )
        rerank_fields = (
            "beit3_raw_cosine",
            "beit3_normalized",
            "rrf_normalized",
            "final_score",
            "score",
        )
        final_beit_fields = (
            "beit3_raw_cosine",
            "beit3_normalized",
            "rrf_normalized",
            "beit3_best_query_role",
            "beit3_best_query_language",
            "beit3_query_scores",
            "rank_delta",
            "rerank_formula",
        )
        expected_formula = {
            "beit3_weight": 0.25,
            "previous_weight": 0.75,
            "previous_score_field": "rrf_score",
            "expression": ("beit3_weight * normalized_beit3 + previous_weight * normalized_rrf"),
        }

        def canonical_identity(item: Any, source: str) -> tuple[Any, ...]:
            if not isinstance(item, dict):
                raise ValueError(f"final BEiT reranker returned a non-object {source} candidate")
            missing = [
                field
                for field in canonical_fields
                if field not in item or item[field] is None or item[field] == ""
            ]
            if missing:
                raise ValueError(
                    f"final BEiT reranker returned {source} candidate without canonical "
                    f"fields: {', '.join(missing)}"
                )
            return tuple(item[field] for field in canonical_fields)

        rrf_snapshot_fields = canonical_fields + (
            "pre_rerank_rank",
            "rrf_score",
            "branch_agreement_count",
            "observed_branches",
            "branch_ranks",
            "rrf_contributions",
            "branch_normalized_scores",
            "weighted_normalized_score",
            "best_branch_rank",
            "branch_provenance",
        )
        expected_by_uid: dict[str, tuple[Any, ...]] = {}
        snapshot_by_uid: dict[str, dict[str, Any]] = {}
        expected_uids: list[str] = []
        for index, item in enumerate(original, 1):
            identity = canonical_identity(item, f"source #{index}")
            uid = str(identity[0])
            if uid in expected_by_uid:
                raise ValueError("RRF output contains duplicate canonical frame_uid values")
            missing_snapshot = [field for field in rrf_snapshot_fields if field not in item]
            if missing_snapshot:
                raise ValueError(
                    "rrf_snapshot_mismatch: source candidate is missing immutable fields: "
                    f"{', '.join(missing_snapshot)}"
                )
            expected_by_uid[uid] = identity
            snapshot_by_uid[uid] = item
            expected_uids.append(uid)

        returned_uids: list[str] = []
        for index, item in enumerate(reranked, 1):
            identity = canonical_identity(item, f"returned #{index}")
            uid = str(identity[0])
            if expected_by_uid.get(uid) != identity:
                raise ValueError(
                    f"final BEiT reranker changed canonical identity for frame_uid {uid!r}"
                )
            returned_uids.append(uid)
        if len(set(returned_uids)) != len(returned_uids):
            raise ValueError("final BEiT reranker returned duplicate canonical frame_uid values")
        if set(returned_uids) != set(expected_uids):
            raise ValueError("final BEiT reranker changed the canonical candidate set")

        expected_top = set(expected_uids[:rerank_count])
        if set(returned_uids[:rerank_count]) != expected_top:
            raise ValueError("final BEiT reranker moved a tail candidate into the rerank window")
        if returned_uids[rerank_count:] != expected_uids[rerank_count:]:
            raise ValueError("final BEiT reranker changed the RRF tail order")

        def finite_number(item: dict[str, Any], field: str, source: str) -> float:
            value = item.get(field)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"final BEiT reranker returned {source} {field} as a non-number")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"final BEiT reranker returned {source} {field} as non-finite")
            return value

        observed_pre_ranks: set[int] = set()
        top_numeric: dict[str, dict[str, float]] = {}
        beit_raw_by_uid: dict[str, float] = {}
        for output_rank, item in enumerate(reranked[:rerank_count], 1):
            source = f"reranked #{output_rank}"
            uid = str(item["frame_uid"])
            snapshot = snapshot_by_uid[uid]
            for field in rrf_snapshot_fields:
                if field not in item or item[field] != snapshot[field]:
                    raise ValueError(
                        f"rrf_snapshot_mismatch: {uid} changed immutable RRF field {field}"
                    )
            pre_rank = item.get("pre_rerank_rank")
            if isinstance(pre_rank, bool) or not isinstance(pre_rank, int):
                raise ValueError(
                    f"final BEiT reranker returned {source} without an integer pre_rerank_rank"
                )
            if not 1 <= pre_rank <= rerank_count:
                raise ValueError(
                    f"final BEiT reranker returned {source} pre_rerank_rank outside rerank window"
                )
            if pre_rank != snapshot["pre_rerank_rank"]:
                raise ValueError(
                    f"rrf_snapshot_mismatch: {uid} pre_rerank_rank does not match its RRF snapshot"
                )
            observed_pre_ranks.add(pre_rank)
            if item.get("rank") != output_rank:
                raise ValueError(
                    f"final BEiT reranker returned {source} with an invalid final rank"
                )
            if item.get("rank_delta") != pre_rank - output_rank:
                raise ValueError(
                    f"final BEiT reranker returned {source} with an invalid rank_delta"
                )

            values = {field: finite_number(item, field, source) for field in rerank_fields}
            if not 0.0 <= values["beit3_normalized"] <= 1.0:
                raise ValueError(
                    f"final BEiT reranker returned {source} beit3_normalized outside [0, 1]"
                )
            if not 0.0 <= values["rrf_normalized"] <= 1.0:
                raise ValueError(
                    f"final BEiT reranker returned {source} rrf_normalized outside [0, 1]"
                )
            if item.get("score_type") != "beit3_coco_cosine_blend":
                raise ValueError(
                    f"final BEiT reranker returned {source} with an invalid score_type"
                )
            if not math.isclose(values["score"], values["final_score"], rel_tol=0.0, abs_tol=1e-7):
                raise ValueError(f"final BEiT reranker returned {source} with score != final_score")
            expected_final = 0.25 * values["beit3_normalized"] + 0.75 * values["rrf_normalized"]
            if not math.isclose(values["final_score"], expected_final, rel_tol=0.0, abs_tol=1e-7):
                raise ValueError(
                    f"final BEiT reranker returned {source} with an invalid 25/75 final_score"
                )

            best_role = item.get("beit3_best_query_role")
            if best_role not in QUERY_ROLES:
                raise ValueError(
                    f"final BEiT reranker returned {source} with an invalid BEiT best role"
                )
            if item.get("beit3_best_query_language") != "en":
                raise ValueError(
                    f"final BEiT reranker returned {source} with a non-English BEiT best language"
                )
            query_scores = item.get("beit3_query_scores")
            if not isinstance(query_scores, dict) or set(query_scores) != set(QUERY_ROLES):
                raise ValueError(
                    f"final BEiT reranker returned {source} without exactly six BEiT query scores"
                )
            parsed_query_scores: dict[str, tuple[float, int]] = {}
            for role in QUERY_ROLES:
                evidence = query_scores[role]
                if not isinstance(evidence, dict):
                    raise ValueError(
                        f"final BEiT reranker returned {source} {role} evidence as a non-object"
                    )
                if evidence.get("role") != role:
                    raise ValueError(
                        f"final BEiT reranker returned {source} {role} evidence with a mismatched role"
                    )
                if evidence.get("language") != "en":
                    raise ValueError(
                        f"final BEiT reranker returned {source} {role} evidence with a non-English language"
                    )
                finite_number(evidence, "cosine", f"{source} {role} evidence")
                evidence_rank = evidence.get("rank")
                if (
                    isinstance(evidence_rank, bool)
                    or not isinstance(evidence_rank, int)
                    or evidence_rank < 1
                ):
                    raise ValueError(
                        f"final BEiT reranker returned {source} {role} evidence with an invalid rank"
                    )
                parsed_query_scores[role] = (
                    finite_number(evidence, "cosine", f"{source} {role} evidence"),
                    evidence_rank,
                )
            expected_best_role = min(
                QUERY_ROLES,
                key=lambda role: (
                    -parsed_query_scores[role][0],
                    parsed_query_scores[role][1],
                    QUERY_ROLES.index(role),
                ),
            )
            expected_beit_raw = parsed_query_scores[expected_best_role][0]
            if not math.isclose(
                values["beit3_raw_cosine"], expected_beit_raw, rel_tol=0.0, abs_tol=1e-7
            ):
                raise ValueError(
                    f"final BEiT reranker returned {source} with a raw cosine not matching query evidence"
                )
            if best_role != expected_best_role:
                raise ValueError(
                    f"final BEiT reranker returned {source} with a non-winning BEiT best role"
                )
            if item.get("rerank_formula") != expected_formula:
                raise ValueError(
                    f"final BEiT reranker returned {source} with an invalid rerank_formula"
                )
            top_numeric[uid] = values
            beit_raw_by_uid[uid] = expected_beit_raw
        if observed_pre_ranks != set(range(1, rerank_count + 1)):
            raise ValueError(
                "final BEiT reranker returned duplicate or missing pre_rerank_rank values"
            )

        expected_beit_normalized = {
            uid: {"raw": raw, "observed": True} for uid, raw in beit_raw_by_uid.items()
        }
        expected_rrf_normalized = {
            str(item["frame_uid"]): {
                "raw": finite_number(item, "rrf_score", "RRF snapshot"),
                "observed": True,
            }
            for item in original[:rerank_count]
        }
        normalize_scores(expected_beit_normalized, "raw")
        normalize_scores(expected_rrf_normalized, "raw")
        for output_rank, item in enumerate(reranked[:rerank_count], 1):
            uid = str(item["frame_uid"])
            source = f"reranked #{output_rank}"
            values = top_numeric[uid]
            expected_beit = float(expected_beit_normalized[uid]["normalized_score"])
            expected_rrf = float(expected_rrf_normalized[uid]["normalized_score"])
            if not math.isclose(
                values["beit3_normalized"], expected_beit, rel_tol=0.0, abs_tol=1e-7
            ):
                raise ValueError(
                    f"final BEiT reranker returned {source} with invalid BEiT normalization"
                )
            if not math.isclose(values["rrf_normalized"], expected_rrf, rel_tol=0.0, abs_tol=1e-7):
                raise ValueError(
                    f"final BEiT reranker returned {source} with invalid RRF normalization"
                )
            expected_final = 0.25 * expected_beit + 0.75 * expected_rrf
            if not math.isclose(values["final_score"], expected_final, rel_tol=0.0, abs_tol=1e-7):
                raise ValueError(
                    f"final BEiT reranker returned {source} with an invalid recomputed 25/75 final_score"
                )

        expected_top_order = sorted(
            reranked[:rerank_count],
            key=lambda item: (
                -top_numeric[str(item["frame_uid"])]["final_score"],
                int(item["pre_rerank_rank"]),
                str(item["frame_uid"]),
            ),
        )
        if [str(item["frame_uid"]) for item in reranked[:rerank_count]] != [
            str(item["frame_uid"]) for item in expected_top_order
        ]:
            raise ValueError("final BEiT reranker did not sort the rerank window by final score")

        # The tail is never BEiT-scored.  Preserve every public RRF field,
        # including nested branch provenance, so an injected adapter cannot
        # silently alter evidence while keeping frame UID/order unchanged.
        tail_snapshot = original[rerank_count:]
        for tail_index, (returned, expected) in enumerate(
            zip(reranked[rerank_count:], tail_snapshot, strict=True),
            rerank_count + 1,
        ):
            leaked = [field for field in final_beit_fields if field in returned]
            if leaked:
                raise ValueError(
                    f"final BEiT reranker leaked final BEiT fields into tail rank {tail_index}: {', '.join(leaked)}"
                )
            if returned.get("rank") != returned.get("pre_rerank_rank"):
                raise ValueError(
                    f"final BEiT reranker returned tail rank {tail_index} with rank != pre_rerank_rank"
                )
            if not (
                returned.get("score") == returned.get("final_score") == returned.get("rrf_score")
                and returned.get("score_type") == "weighted_rrf"
            ):
                raise ValueError(
                    f"final BEiT reranker returned tail rank {tail_index} with invalid RRF scoring"
                )
            if returned != expected:
                raise ValueError(
                    "rrf_snapshot_mismatch: "
                    f"final BEiT reranker mutated RRF tail candidate at rank {tail_index}"
                )
        return reranked, rerank_info

    def execute(
        self,
        query_bundle: dict[str, Any],
        branch_weights: dict[str, float] | None = None,
        *,
        _health_already_checked: bool = False,
        _lock_already_held: bool = False,
    ) -> dict[str, Any]:
        # Validate request contracts before taking the heavy lock so malformed
        # requests are reported as 422 even when another search is running.
        _by_role, en_texts = self._query_texts(query_bundle)
        weights = normalize_branch_weights(branch_weights)
        acquired = False
        if not _lock_already_held:
            if not self.search_lock.acquire(blocking=False):
                raise RuntimeError("KIS_FUSION_SEARCH_BUSY")
            acquired = True
        started = time.perf_counter()
        try:
            if not _health_already_checked:
                aggregate_health = self.health()
                if aggregate_health.get("ready") is not True:
                    raise RuntimeError("KIS_FUSION_NOT_READY")

            branch_started = time.perf_counter()
            try:
                branch1_result = self._call_locked(
                    self.branch1,
                    query_bundle,
                    {"siglip2": 0.45, "metaclip2": 0.30, "beit3": 0.25},
                    2_000,
                    BRANCH1_FINAL_TOP_K,
                )
                branch1_ms = round((time.perf_counter() - branch_started) * 1000.0, 2)

                branch_started = time.perf_counter()
                branch2_result = self._call_locked(
                    self.branch2,
                    query_bundle,
                    {"dense": 0.70, "sparse": 0.30},
                    {"beit3": 0.40, "previous": 0.60},
                    BRANCH2_PER_STREAM_TOP_K,
                    BRANCH2_PRE_RERANK_TOP_K,
                    BRANCH2_RERANK_TOP_K,
                )
                branch2_ms = round((time.perf_counter() - branch_started) * 1000.0, 2)

                branch_started = time.perf_counter()
                asr_result = self._call_locked(self.asr, query_bundle, 2_000, 500)
                asr_ms = round((time.perf_counter() - branch_started) * 1000.0, 2)

                branch_started = time.perf_counter()
                ocr_result = self._call_locked(self.ocr, query_bundle, 2_000, 500)
                ocr_ms = round((time.perf_counter() - branch_started) * 1000.0, 2)
            except Exception as error:
                raise RuntimeError(f"KIS_FUSION_BRANCH_FAILED: {error}") from error

            pools = {
                "branch1": branch1_result,
                "branch2": branch2_result,
                "asr": asr_result,
                "ocr": ocr_result,
            }
            candidate_uids: set[str] = set()
            for payload in pools.values():
                raw_results = payload.get("results")
                if isinstance(raw_results, list):
                    candidate_uids.update(
                        str(item.get("frame_uid"))
                        for item in raw_results
                        if isinstance(item, dict) and item.get("frame_uid")
                    )
            rrf_started = time.perf_counter()
            try:
                fused, pool_counts = fuse_branch_pools(
                    pools,
                    weights,
                    rrf_k=RRF_K,
                    top_k=FINAL_TOP_K,
                )
            except Exception as error:
                raise RuntimeError(f"KIS_FUSION_RRF_FAILED: {error}") from error
            rrf_ms = round((time.perf_counter() - rrf_started) * 1000.0, 2)

            # RRF keeps the first branch record internally to make identity
            # reconciliation cheap.  Materialize an allow-listed candidate
            # before BEiT so neither the first branch's full stream evidence
            # nor a large DAM/ASR/OCR payload can leak into the public result.
            fused = [
                materialize_fusion_candidate(item, include_rerank_fields=False) for item in fused
            ]

            rerank_info: dict[str, Any] = {
                "candidate_count": len(fused),
                "rerank_count": 0,
                "weights": {"beit3": 0.25, "previous": 0.75},
                "scoring": "cosine",
                "checkpoint_task": "BEiT-3 COCO Retrieval",
                "text_encoder_output": "language_head",
                "query_language": "en",
                "cache_hit": None,
            }
            rerank_started = time.perf_counter()
            timing: dict[str, Any] = {}
            rerank_count = 0
            if fused:
                rerank_count = min(FINAL_RERANK_TOP_K, len(fused))
                try:
                    # Query-cache lookup, worker lifecycle, text encoding and
                    # the candidate-only Qdrant/rerank pass are one BEiT
                    # phase.  Keep all of them behind the same error boundary
                    # so callers can distinguish a BEiT failure from a
                    # generic orchestration failure.
                    beit_vectors, beit_diagnostics, beit_timing = self._encode_beit_queries(
                        en_texts
                    )
                    timing["beit3_cache_hit"] = bool(beit_timing.get("cache_hit", False))
                    timing["beit3_model_loading_ms"] = float(
                        beit_timing.get("model_loading_ms", 0.0)
                    )
                    timing["beit3_encode_ms"] = float(
                        beit_timing.get("encoding_ms", beit_timing.get("inference_ms", 0.0))
                    )
                    timing["beit3_worker_pid"] = beit_timing.get("worker_pid")
                    timing["beit3_worker_reused"] = bool(beit_timing.get("worker_reused", False))
                    timing["beit3_worker_spawned"] = bool(beit_timing.get("worker_spawned", False))
                    # The adapter interface is injectable.  Freeze the
                    # complete compact RRF pool before handing it over so an
                    # adapter that mutates its input in-place is subject to
                    # the same top-window and immutable-tail contract.
                    rrf_snapshot = deepcopy(fused)
                    reranked, rerank_info = self._get_reranker().rerank(
                        fused,
                        en_texts,
                        top_k=rerank_count,
                        weights={"beit3": 0.25, "previous": 0.75},
                        text_vectors=beit_vectors,
                        tokenizer_diagnostics=beit_diagnostics,
                        previous_score_field="rrf_score",
                        previous_rank_field="pre_rerank_rank",
                        previous_score_label="rrf",
                    )
                    try:
                        fused, rerank_info = self._validate_final_beit_output(
                            rrf_snapshot,
                            reranked,
                            rerank_info,
                            rerank_count,
                        )
                    except Exception as error:
                        raise RuntimeError(
                            f"KIS_FUSION_BEIT3_FAILED: output_validation: {error}"
                        ) from error
                    rerank_info = {
                        **rerank_info,
                        "cache_hit": beit_timing.get("cache_hit"),
                        "worker": beit_timing,
                    }
                    timing["beit3_qdrant_ms"] = float(rerank_info.get("qdrant_ms", 0.0))
                    timing["beit3_scoring_ms"] = float(rerank_info.get("scoring_ms", 0.0))
                    for item in fused:
                        item.setdefault("final_score", item.get("score"))

                    # Re-apply the public allow-list after the injectable
                    # reranker returns.  Validation above guarantees this
                    # cannot silently truncate, widen, or alter the RRF tail.
                    for rank, item in enumerate(fused, 1):
                        item["rank"] = rank
                        if "final_score" not in item:
                            item["final_score"] = item.get("score", item.get("rrf_score", 0.0))
                    fused = [
                        materialize_fusion_candidate(
                            item,
                            include_rerank_fields=index < rerank_count,
                        )
                        for index, item in enumerate(fused)
                    ]
                    for rank, item in enumerate(fused, 1):
                        item["rank"] = rank
                except Exception as error:
                    if str(error).startswith("KIS_FUSION_BEIT3_FAILED"):
                        raise
                    raise RuntimeError(f"KIS_FUSION_BEIT3_FAILED: {error}") from error
            rerank_ms = round((time.perf_counter() - rerank_started) * 1000.0, 2)

            timing.update(
                {
                    "branches": {
                        "branch1_ms": branch1_ms,
                        "branch2_ms": branch2_ms,
                        "asr_ms": asr_ms,
                        "ocr_ms": ocr_ms,
                    },
                    "rrf_ms": rrf_ms,
                    "beit3_rerank_ms": rerank_ms,
                    "total_ms": round((time.perf_counter() - started) * 1000.0, 2),
                }
            )
            return {
                "schema_version": FINAL_FUSION_RESULT_SCHEMA_VERSION,
                "task_type": "KIS",
                "fusion_applied": True,
                "fusion_method": "weighted_rrf",
                "reranking_applied": bool(fused),
                "rrf_k": RRF_K,
                "branch_weights": weights,
                "branch_pool_counts": pool_counts,
                "branch_pool_limits": dict(BRANCH_POOL_LIMITS),
                "candidate_count_before_gate": len(candidate_uids),
                "final_top_k": FINAL_TOP_K,
                "pre_rerank_top_k": FINAL_TOP_K,
                # This is the fixed server-side policy, not the observed
                # number of candidates.  The actual work performed is
                # reported separately as ``rerank.rerank_count``.
                "rerank_top_k": FINAL_RERANK_TOP_K,
                "beit3_weight": 0.25,
                "rrf_weight": 0.75,
                "result_count": len(fused),
                "timing": timing,
                "rerank": rerank_info,
                "query_streams": {
                    "branch1": 30,
                    "branch2": 18,
                    "asr": 12,
                    "ocr": 12,
                },
                "results": fused,
            }
        finally:
            # Branch workers keep their bounded idle process where possible;
            # the shared lock is always released even when a dependency fails.
            gc.collect()
            if acquired:
                self.search_lock.release()

    def execute_batch(
        self,
        query_bundles: list[dict[str, Any]],
        branch_weights: dict[str, float] | None = None,
        *,
        _health_already_checked: bool = False,
    ) -> list[dict[str, Any]]:
        """Run two to six complete KIS searches under one atomic lock.

        Ordered-event retrieval must not interleave another heavy request
        between E1 and E2.  Validate every event before locking, check health
        once, and then reuse the unchanged single-query implementation while
        the caller owns the shared lock.
        """

        if not 2 <= len(query_bundles) <= 6:
            raise ValueError("Ordered KIS fusion requires between 2 and 6 event bundles")
        for bundle in query_bundles:
            self._query_texts(bundle)
        weights = normalize_branch_weights(branch_weights)
        if not self.search_lock.acquire(blocking=False):
            raise RuntimeError("KIS_FUSION_SEARCH_BUSY")
        try:
            if not _health_already_checked:
                aggregate_health = self.health()
                if aggregate_health.get("ready") is not True:
                    raise RuntimeError("KIS_FUSION_NOT_READY")
            return [
                self.execute(
                    bundle,
                    weights,
                    _health_already_checked=True,
                    _lock_already_held=True,
                )
                for bundle in query_bundles
            ]
        finally:
            self.search_lock.release()

    def _execute_prechecked(
        self,
        query_bundle: dict[str, Any],
        branch_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Execute after the API admitted the same cached health generation."""

        return self.execute(
            query_bundle,
            branch_weights,
            _health_already_checked=True,
        )


__all__ = ["KisFusionSearch"]
