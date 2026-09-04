"""Behavioural tests for KIS weighted RRF and final BEiT-3 reranking.

These tests use small canonical fixtures and never require model weights or a
running Qdrant service.  They are also useful as a contract for the four
standalone branch adapters used by :class:`KisFusionSearch`.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from online.cpu_server import _fusion_production_ready, _kis_runtime_error_code
from online.src.retrieval.branches.branch1.contracts import QUERY_ROLES
from online.src.retrieval.branches.final_fusion.contracts import normalize_branch_weights
from online.src.retrieval.branches.final_fusion.provenance import materialize_fusion_candidate
from online.src.retrieval.branches.final_fusion.rrf import fuse_branch_pools
from online.src.retrieval.branches.final_fusion.service import KisFusionSearch
from online.src.retrieval.branches.rerankers.beit3_cosine import Beit3CosineReranker
from online.src.retrieval.infrastructure.persistent_cache import (
    PersistentQueryEmbeddingCache,
)
from online.src.retrieval.infrastructure.scoring import normalize_scores


def _frame(uid: str, rank: int) -> dict[str, object]:
    video_id, frame_text = uid.split(":", 1)
    frame_idx = int(frame_text)
    return {
        "frame_uid": uid,
        "point_id": rank,
        "global_idx": rank,
        "video_id": video_id,
        "frame_idx": frame_idx,
        "keyframe_n": rank,
        "pts_time_s": float(frame_idx) / 30.0,
        "fps": 30.0,
        "image_relpath": f"keyframes/{video_id}/{frame_idx:08d}.jpg",
        "rank": rank,
        "score": 1.0 / rank,
        "normalized_score": 1.0 / rank,
    }


def _pool(schema: str, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": schema,
        "future_fusion_eligible": True,
        "result_count": len(rows),
        "results": rows,
    }


def _query_bundle() -> dict[str, object]:
    return {
        "schema_version": "branch1.query.v1",
        "queries": [
            {"role": role, "vi": f"vi {role}", "en": f"en {role}"}
            for role in ("original", "entity", "action", "context", "synonym", "keyword")
        ],
    }


class FinalFusionContractTests(unittest.TestCase):
    def test_weights_are_positive_and_normalized(self) -> None:
        self.assertEqual(
            normalize_branch_weights({"branch1": 4, "branch2": 3, "ocr": 2, "asr": 1}),
            {"branch1": 0.4, "branch2": 0.3, "ocr": 0.2, "asr": 0.1},
        )
        with self.assertRaises(ValueError):
            normalize_branch_weights({"branch1": 0, "branch2": 1, "ocr": 1, "asr": 1})

    def test_weighted_rrf_uses_rank_only_and_deduplicates_uid(self) -> None:
        common = _frame("L01_V001:4", 1)
        branch1 = _frame("L01_V001:8", 2)
        branch2 = dict(common, rank=1)
        pools = {
            "branch1": _pool("branch1.result.v1", [common, branch1]),
            "branch2": _pool("branch2.result.v1", [branch2]),
            "ocr": _pool("branch3.ocr.result.v1", []),
            "asr": _pool("branch3.asr.result.v1", []),
        }
        results, counts = fuse_branch_pools(
            pools,
            {"branch1": 0.4, "branch2": 0.3, "ocr": 0.15, "asr": 0.15},
        )
        self.assertEqual(counts, {"branch1": 2, "branch2": 1, "ocr": 0, "asr": 0})
        self.assertEqual([item["frame_uid"] for item in results], ["L01_V001:4", "L01_V001:8"])
        self.assertAlmostEqual(results[0]["rrf_score"], 0.4 / 61 + 0.3 / 61)
        self.assertEqual(results[0]["branch_agreement_count"], 2)
        self.assertEqual(results[0]["score_type"], "weighted_rrf")

    def test_identity_conflict_is_fail_closed(self) -> None:
        first = _frame("L01_V001:4", 1)
        second = dict(first, point_id=2, global_idx=2)
        pools = {
            "branch1": _pool("branch1.result.v1", [first]),
            "branch2": _pool("branch2.result.v1", [second]),
            "ocr": _pool("branch3.ocr.result.v1", []),
            "asr": _pool("branch3.asr.result.v1", []),
        }
        with self.assertRaises(ValueError):
            fuse_branch_pools(pools, None)  # type: ignore[arg-type]

    def test_point_id_and_global_idx_alias_must_agree(self) -> None:
        row = _frame("L01_V001:4", 1)
        pools = {
            "branch1": _pool("branch1.result.v1", [dict(row, global_idx=2)]),
            "branch2": _pool("branch2.result.v1", []),
            "ocr": _pool("branch3.ocr.result.v1", []),
            "asr": _pool("branch3.asr.result.v1", []),
        }
        with self.assertRaises(ValueError):
            fuse_branch_pools(
                pools,
                {"branch1": 0.4, "branch2": 0.3, "ocr": 0.15, "asr": 0.15},
            )

    def test_canonical_point_ids_are_unique_within_each_pool(self) -> None:
        first = _frame("L01_V001:4", 1)
        second = dict(_frame("L01_V001:8", 1), rank=2)
        pools = {
            "branch1": _pool("branch1.result.v1", [first, second]),
            "branch2": _pool("branch2.result.v1", []),
            "ocr": _pool("branch3.ocr.result.v1", []),
            "asr": _pool("branch3.asr.result.v1", []),
        }
        with self.assertRaises(ValueError):
            fuse_branch_pools(
                pools,
                {"branch1": 0.4, "branch2": 0.3, "ocr": 0.15, "asr": 0.15},
            )

    def test_pool_gate_and_sequential_rank_are_enforced(self) -> None:
        row = _frame("L01_V001:4", 2)
        pools = {
            "branch1": _pool("branch1.result.v1", [row]),
            "branch2": _pool("branch2.result.v1", []),
            "ocr": _pool("branch3.ocr.result.v1", []),
            "asr": _pool("branch3.asr.result.v1", []),
        }
        with self.assertRaises(ValueError):
            fuse_branch_pools(pools, {"branch1": 0.4, "branch2": 0.3, "ocr": 0.15, "asr": 0.15})

    def test_missing_voter_contributes_zero_without_affecting_rank_formula(self) -> None:
        row = _frame("L01_V001:4", 1)
        pools = {
            "branch1": _pool("branch1.result.v1", [row]),
            "branch2": _pool("branch2.result.v1", []),
            "ocr": _pool("branch3.ocr.result.v1", []),
            "asr": _pool("branch3.asr.result.v1", []),
        }
        results, _ = fuse_branch_pools(
            pools,
            {"branch1": 0.4, "branch2": 0.3, "ocr": 0.15, "asr": 0.15},
        )
        self.assertEqual(results[0]["branch_agreement_count"], 1)
        self.assertEqual(results[0]["rrf_contributions"]["branch2"], 0.0)
        self.assertAlmostEqual(results[0]["rrf_score"], 0.4 / 61.0)

    def test_result_count_and_identity_path_are_strict(self) -> None:
        row = _frame("L01_V001:4", 1)
        base = {
            "branch1": _pool("branch1.result.v1", [row]),
            "branch2": _pool("branch2.result.v1", []),
            "ocr": _pool("branch3.ocr.result.v1", []),
            "asr": _pool("branch3.asr.result.v1", []),
        }
        with self.assertRaises(ValueError):
            fuse_branch_pools(
                {
                    **base,
                    "branch1": {**base["branch1"], "result_count": True},
                },
                None,  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            fuse_branch_pools(
                {
                    **base,
                    "branch1": _pool(
                        "branch1.result.v1", [dict(row, image_relpath="../outside.jpg")]
                    ),
                },
                {"branch1": 0.4, "branch2": 0.3, "ocr": 0.15, "asr": 0.15},
            )


class _FilteredQdrant:
    def query(self, _collection, _vector_name, _vector, limit, _query_filter=None):
        return [
            {"id": 1, "score": 0.8, "payload": {"frame_uid": "L01_V001:4"}},
            {"id": 2, "score": 0.7, "payload": {"frame_uid": "L01_V001:8"}},
        ][:limit]


class _RecordingQdrant:
    def __init__(self, point_payloads: dict[int, str]) -> None:
        self.point_payloads = point_payloads
        self.filters: list[object] = []

    def query(self, _collection, _vector_name, _vector, limit, query_filter=None):
        self.filters.append(query_filter)
        has_id = (query_filter or {}).get("must", [{}])[0].get("has_id", [])
        return [
            {
                "id": point_id,
                "score": 0.9 - point_id / 10_000.0,
                "payload": {"frame_uid": self.point_payloads[point_id]},
            }
            for point_id in has_id[:limit]
        ]


class _EvidenceQdrant(_RecordingQdrant):
    """Qdrant fixture that corrupts one final-BEiT evidence invariant."""

    def __init__(self, point_payloads: dict[int, str], mode: str) -> None:
        super().__init__(point_payloads)
        self.mode = mode
        self.calls = 0

    def query(self, *args, **kwargs):
        self.calls += 1
        values = super().query(*args, **kwargs)
        if self.mode == "missing_point":
            return values[:-1]
        if self.mode == "duplicate_point":
            return values + [dict(values[0])] if values else values
        if self.mode == "payload_uid_mismatch":
            corrupted = [dict(item) for item in values]
            if corrupted:
                corrupted[0] = {
                    **corrupted[0],
                    "payload": {"frame_uid": "L99_V999:999"},
                }
            return corrupted
        if self.mode == "missing_stream" and self.calls == 6:
            return []
        return values


class _ConstantScoreQdrant(_RecordingQdrant):
    def query(self, *args, **kwargs):
        values = super().query(*args, **kwargs)
        return [{**item, "score": 0.8} for item in values]


class _FaultyCache:
    def __init__(self, stage: str) -> None:
        self.stage = stage

    def key(self, *_args, **_kwargs):
        if self.stage == "key":
            raise RuntimeError("cache key failed")
        return "beit3-test-key"

    def get(self, _key):
        if self.stage == "get":
            raise RuntimeError("cache get failed")
        return None

    def put(self, *_args, **_kwargs):
        if self.stage == "put":
            raise RuntimeError("cache put failed")


class _HitCache(_FaultyCache):
    def __init__(self) -> None:
        super().__init__("hit")

    def get(self, _key):
        return np.ones((6, 768), dtype=np.float32), [{} for _ in range(6)]


class _TinyBeitEncoder:
    revisions = {"beit3": "test-revision"}

    def __init__(self, failure: str | None = None) -> None:
        self.failure = failure
        self.encode_calls = 0

    def encode(self, _model_name, _texts):
        self.encode_calls += 1
        if self.failure:
            raise RuntimeError(self.failure)
        return np.ones((6, 768), dtype=np.float32), [{} for _ in range(6)]


class _FallbackBeitEncoder(_TinyBeitEncoder):
    cache_device = "mps"

    def __init__(self) -> None:
        super().__init__()
        self.device = "mps"

    def cache_device_for_model(self, _model_name: str) -> str:
        return self.device

    def encode(self, model_name, texts):
        vectors, diagnostics = super().encode(model_name, texts)
        self.device = "cpu"
        return vectors, diagnostics


class _FakeFusionBranch:
    def __init__(self, name: str, payload: dict[str, object], calls: list[str]) -> None:
        self.name = name
        self.payload = payload
        self.calls = calls

    def health(self) -> dict[str, object]:
        return {"ready": True, "production_ready": False}

    def _execute_locked(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        self.calls.append(self.name)
        return self.payload


class _RecordingFusionReranker:
    def __init__(self) -> None:
        self.selected_uids: list[str] = []

    def rerank(self, candidates, _texts, *, top_k, **_kwargs):
        selected = candidates[:top_k]
        self.selected_uids = [str(item["frame_uid"]) for item in selected]
        query_evidence: dict[str, dict[str, dict[str, object]]] = {}
        beit_items: dict[str, dict[str, object]] = {}
        rrf_items: dict[str, dict[str, object]] = {}
        for position, item in enumerate(selected, 1):
            uid = str(item["frame_uid"])
            query_scores = {
                role: {
                    "cosine": 0.9 - (position * 0.001) - (role_index * 0.0001),
                    "rank": position,
                    "role": role,
                    "language": "en",
                }
                for role_index, role in enumerate(QUERY_ROLES)
            }
            query_evidence[uid] = query_scores
            beit_items[uid] = {"raw": query_scores["original"]["cosine"], "observed": True}
            rrf_items[uid] = {"raw": item["rrf_score"], "observed": True}
        normalize_scores(beit_items, "raw")
        normalize_scores(rrf_items, "raw")
        reranked = []
        for item in selected:
            uid = str(item["frame_uid"])
            pre_rank = int(item["pre_rerank_rank"])
            beit_normalized = float(beit_items[uid]["normalized_score"])
            rrf_normalized = float(rrf_items[uid]["normalized_score"])
            final_score = 0.25 * beit_normalized + 0.75 * rrf_normalized
            query_scores = query_evidence[uid]
            copied = dict(item)
            copied.update(
                {
                    "pre_rerank_rank": pre_rank,
                    "beit3_raw_cosine": query_scores["original"]["cosine"],
                    "beit3_normalized": beit_normalized,
                    "rrf_normalized": rrf_normalized,
                    "beit3_best_query_role": "original",
                    "beit3_best_query_language": "en",
                    "beit3_query_scores": query_scores,
                    "final_score": final_score,
                    "score": final_score,
                    "score_type": "beit3_coco_cosine_blend",
                    "reranked_score": final_score,
                    "rerank_score_type": "beit3_coco_cosine_blend",
                    "rerank_formula": {
                        "beit3_weight": 0.25,
                        "previous_weight": 0.75,
                        "previous_score_field": "rrf_score",
                        "expression": (
                            "beit3_weight * normalized_beit3 + previous_weight * normalized_rrf"
                        ),
                    },
                }
            )
            reranked.append(copied)
        reranked.sort(
            key=lambda value: (
                -float(value["final_score"]),
                int(value["pre_rerank_rank"]),
                str(value["frame_uid"]),
            )
        )
        for rank, item in enumerate(reranked, 1):
            item["rank"] = rank
            item["rank_delta"] = int(item["pre_rerank_rank"]) - rank
        return reranked + [deepcopy(item) for item in candidates[top_k:]], {
            "candidate_count": len(candidates),
            "rerank_count": len(selected),
            "weights": {"beit3": 0.25, "previous": 0.75},
            "qdrant_ms": 0.0,
            "scoring_ms": 0.0,
        }


class _FailIfUsed:
    def __getattr__(self, _name: str):
        raise AssertionError("empty KIS fusion must not load or score BEiT-3")


class RerankerContractTests(unittest.TestCase):
    def test_final_rerank_accepts_rrf_field_and_keeps_tail(self) -> None:
        reranker = Beit3CosineReranker(
            _FilteredQdrant(), object(), {"L01_V001:4": 1, "L01_V001:8": 2}
        )
        candidates = [
            dict(_frame("L01_V001:4", 1), rrf_score=0.02, pre_rerank_rank=1),
            dict(_frame("L01_V001:8", 2), rrf_score=0.01, pre_rerank_rank=2),
        ]
        result, info = reranker.rerank(
            candidates,
            ["query"] * 6,
            top_k=1,
            weights={"beit3": 0.25, "previous": 0.75},
            text_vectors=np.ones((6, 768), dtype=np.float32),
            previous_score_field="rrf_score",
            previous_rank_field="pre_rerank_rank",
            previous_score_label="rrf",
        )
        self.assertEqual(info["rerank_count"], 1)
        self.assertEqual(result[1]["frame_uid"], "L01_V001:8")
        self.assertNotIn("beit3_raw_cosine", result[1])
        self.assertEqual(result[0]["rerank_formula"]["previous_score_field"], "rrf_score")

    def test_final_rerank_queries_only_the_first_100_of_a_150_candidate_pool(self) -> None:
        point_payloads = {point_id: f"L01_V001:{point_id * 4}" for point_id in range(1, 151)}
        frame_point_ids = {frame_uid: point_id for point_id, frame_uid in point_payloads.items()}
        qdrant = _RecordingQdrant(point_payloads)
        reranker = Beit3CosineReranker(
            qdrant,
            object(),
            frame_point_ids,
        )
        candidates = [
            dict(
                _frame(uid, point_id),
                rrf_score=1.0 / point_id,
                pre_rerank_rank=point_id,
            )
            for point_id, uid in point_payloads.items()
        ]
        result, info = reranker.rerank(
            candidates,
            [f"query-{index}" for index in range(6)],
            top_k=100,
            weights={"beit3": 0.25, "previous": 0.75},
            text_vectors=np.ones((6, 768), dtype=np.float32),
            previous_score_field="rrf_score",
            previous_rank_field="pre_rerank_rank",
            previous_score_label="rrf",
        )
        self.assertEqual(info["rerank_count"], 100)
        self.assertEqual(len(qdrant.filters), 6)
        expected_filter = {"must": [{"has_id": list(range(1, 101))}]}
        self.assertTrue(all(value == expected_filter for value in qdrant.filters))
        self.assertEqual(len(result), 150)
        self.assertEqual(
            [item["frame_uid"] for item in result[100:]],
            [point_payloads[point_id] for point_id in range(101, 151)],
        )
        self.assertTrue(all("beit3_raw_cosine" not in item for item in result[100:]))
        self.assertTrue(all("rank_delta" not in item for item in result[100:]))
        self.assertEqual(
            set(result[0]["beit3_query_scores"]),
            {
                "original",
                "entity",
                "action",
                "context",
                "synonym",
                "keyword",
            },
        )

    def test_constant_beit_and_rrf_scores_use_half_normalization_and_fixed_blend(self) -> None:
        point_payloads = {1: "L01_V001:4", 2: "L01_V001:8"}
        reranker = Beit3CosineReranker(
            _ConstantScoreQdrant(point_payloads),
            object(),
            {frame_uid: point_id for point_id, frame_uid in point_payloads.items()},
        )
        result, _info = reranker.rerank(
            [
                dict(_frame("L01_V001:4", 1), rrf_score=0.5, pre_rerank_rank=1),
                dict(_frame("L01_V001:8", 2), rrf_score=0.5, pre_rerank_rank=2),
            ],
            ["query"] * 6,
            top_k=2,
            weights={"beit3": 0.25, "previous": 0.75},
            text_vectors=np.ones((6, 768), dtype=np.float32),
            previous_score_field="rrf_score",
            previous_rank_field="pre_rerank_rank",
            previous_score_label="rrf",
        )
        self.assertTrue(all(item["beit3_normalized"] == 0.5 for item in result))
        self.assertTrue(all(item["rrf_normalized"] == 0.5 for item in result))
        self.assertTrue(all(item["final_score"] == 0.5 for item in result))


class FusionProvenanceContractTests(unittest.TestCase):
    def test_materialized_candidate_is_allowlisted_and_branch2_languages_are_explicit(self) -> None:
        row = _frame("L01_V001:4", 1)
        branch1_record = {
            **row,
            "model_provenance": {
                "siglip2": {
                    "observed": True,
                    "raw_cosine": 0.8,
                    "normalized_score": 0.7,
                    "best_query_role": "original",
                    "best_query_rank": 1,
                    "query_scores": {"all": "large"},
                }
            },
            "query_scores": {"all": "large"},
        }
        row.update(
            {
                "large_raw_stream_dump": "x" * 10_000,
                "branch_provenance": {
                    "branch1": branch1_record,
                    "branch2": {
                        "dense_observed": True,
                        "dense_raw": 0.7,
                        "sparse_observed": True,
                        "sparse_raw": 4.2,
                        "beit3_raw_cosine": 0.6,
                        "beit3_query_scores": {"all": "large"},
                        "dense_query_scores": {"all": "large"},
                        "hybrid_provenance": {"dense": {"raw": 0.7}},
                    },
                },
                "branch_ranks": {"branch1": 1},
                "rrf_contributions": {"branch1": 0.4 / 61},
                "branch_normalized_scores": {"branch1": 0.5},
                "observed_branches": ["branch1"],
                "branch_agreement_count": 1,
                "weighted_normalized_score": 0.2,
                "best_branch_rank": 1,
                "rrf_score": 0.01,
                "pre_rerank_rank": 1,
            }
        )
        materialized = materialize_fusion_candidate(row)
        self.assertNotIn("large_raw_stream_dump", materialized)
        self.assertNotIn("model_provenance", materialized)
        self.assertNotIn("dam_winner", materialized)
        self.assertNotIn("reranked_score", materialized)
        self.assertNotIn("rerank_score_type", materialized)
        self.assertNotIn(
            "query_scores", materialized.get("branch_provenance", {}).get("branch1", {})
        )
        branch1_model = materialized["branch_provenance"]["branch1"]["model_provenance"]
        self.assertNotIn("query_scores", branch1_model["siglip2"])
        branch2 = materialized["branch_provenance"]["branch2"]
        self.assertEqual(branch2["dense_best_query_language"], "en")
        self.assertEqual(branch2["sparse_best_query_language"], "en")
        self.assertEqual(branch2["beit3_best_query_language"], "en")
        self.assertNotIn("dense_query_scores", branch2)
        self.assertNotIn("beit3_query_scores", branch2)
        self.assertNotIn("hybrid_provenance", branch2)
        self.assertNotIn("rerank_score_type", branch2)
        self.assertNotIn("query_scores", branch2.get("dam_winner", {}))

    def test_unobserved_branch2_evidence_does_not_receive_a_language(self) -> None:
        compact = materialize_fusion_candidate(
            {
                **_frame("L01_V001:4", 1),
                "rrf_score": 0.01,
                "branch_provenance": {
                    "branch2": {
                        "dense_observed": False,
                        "dense_raw": None,
                        "sparse_observed": False,
                        "sparse_raw": None,
                    }
                },
            }
        )
        evidence = compact["branch_provenance"]["branch2"]
        self.assertNotIn("dense_best_query_language", evidence)
        self.assertNotIn("sparse_best_query_language", evidence)


class FusionHealthContractTests(unittest.TestCase):
    def test_resource_qualification_is_required_only_for_production(self) -> None:
        branches = tuple({"ready": True, "production_ready": True} for _ in range(4))
        self.assertFalse(_fusion_production_ready(True, branches, {"production_ready": False}))
        self.assertTrue(_fusion_production_ready(True, branches, {"production_ready": True}))
        self.assertFalse(
            _fusion_production_ready(
                True, (branches[0],) * 3 + ({"ready": True},), {"production_ready": True}
            )
        )
        self.assertFalse(_fusion_production_ready(False, branches, {"production_ready": True}))
        self.assertFalse(_fusion_production_ready(True, branches, {}))
        self.assertFalse(_fusion_production_ready(True, branches, None))

    def test_runtime_error_codes_preserve_fusion_phase(self) -> None:
        self.assertEqual(
            _kis_runtime_error_code("KIS_FUSION_SEARCH_BUSY"), "KIS_FUSION_SEARCH_BUSY"
        )
        self.assertEqual(
            _kis_runtime_error_code("KIS_FUSION_BRANCH_FAILED: OCR unavailable"),
            "KIS_FUSION_BRANCH_FAILED",
        )
        self.assertEqual(
            _kis_runtime_error_code("KIS_FUSION_RRF_FAILED: duplicate UID"), "KIS_FUSION_RRF_FAILED"
        )
        self.assertEqual(
            _kis_runtime_error_code("KIS_FUSION_BEIT3_FAILED: missing point"),
            "KIS_FUSION_BEIT3_FAILED",
        )
        self.assertEqual(
            _kis_runtime_error_code("unexpected runtime failure"), "KIS_FUSION_EXECUTION_FAILED"
        )

    def test_cache_contract_and_malformed_resource_report_fail_closed_without_crashing(
        self,
    ) -> None:
        class _HealthBranch(_FakeFusionBranch):
            def __init__(self, name: str, health: dict[str, object]) -> None:
                super().__init__(name, _pool("branch1.result.v1", []), [])
                self._health = health

            def health(self) -> dict[str, object]:
                return self._health

        branch2_health = {
            "ready": True,
            "production_ready": True,
            "components": {
                "frame_mapping": {"ready": True},
                "beit3_collection": {"ready": True},
                "beit3_ingestion": {"ready": True},
                "beit3_text_encoder": {"ready": True},
            },
        }
        ready_health = {"ready": True, "production_ready": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
            metadata.parent.mkdir(parents=True)
            metadata.write_text('{"frame_uid":"L01_V001:4"}\n', encoding="utf-8")
            branch1 = _HealthBranch("branch1", ready_health)
            branch2 = _HealthBranch("branch2", branch2_health)
            asr = _HealthBranch("asr", ready_health)
            ocr = _HealthBranch("ocr", ready_health)
            service = KisFusionSearch(branch1, branch2, asr, ocr, data_root=root, state_root=root)

            branch1.cache = object()
            with patch(
                "online.src.retrieval.branches.final_fusion.service.resource_qualification",
                return_value={"production_ready": True},
            ):
                cache_health = service.health()
            self.assertFalse(cache_health["ready"])
            self.assertFalse(cache_health["components"]["query_cache"]["ready"])

            branch1.cache = _FaultyCache("none")
            with patch(
                "online.src.retrieval.branches.final_fusion.service.resource_qualification",
                side_effect=ValueError("resource report is malformed"),
            ):
                resource_health = service.health()
            self.assertTrue(resource_health["ready"])
            self.assertFalse(resource_health["production_ready"])
            self.assertEqual(
                resource_health["resource_qualification"]["error"], "resource report is malformed"
            )


class FusionServiceContractTests(unittest.TestCase):
    def _service(self, pools: dict[str, dict[str, object]], calls: list[str]) -> KisFusionSearch:
        branches = {
            name: _FakeFusionBranch(name, pools[name], calls)
            for name in ("branch1", "branch2", "asr", "ocr")
        }
        service = KisFusionSearch(
            branches["branch1"],
            branches["branch2"],
            branches["asr"],
            branches["ocr"],
            data_root=Path("."),
            state_root=Path("."),
            search_lock=threading.Lock(),
        )
        # The service-level health contract is exercised separately.  This
        # adapter keeps the orchestration tests independent of real artifacts.
        service.health = lambda: {"ready": True, "production_ready": False}  # type: ignore[method-assign]
        return service

    @staticmethod
    def _pools_with_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
        return {
            "branch1": _pool("branch1.result.v1", rows),
            "branch2": _pool("branch2.result.v1", []),
            "ocr": _pool("branch3.ocr.result.v1", []),
            "asr": _pool("branch3.asr.result.v1", []),
        }

    @staticmethod
    def _valid_beit_timing() -> dict[str, object]:
        return {"cache_hit": True, "model_loading_ms": 0.0, "encoding_ms": 0.0}

    def _install_actual_reranker(
        self,
        service: KisFusionSearch,
        qdrant: object,
        frame_point_ids: dict[str, int],
    ) -> None:
        reranker = Beit3CosineReranker(qdrant, object(), frame_point_ids)
        service._get_reranker = lambda: reranker  # type: ignore[method-assign]

    def test_calls_four_branches_in_fixed_order_and_reranks_only_first_100(self) -> None:
        rows = [_frame(f"L01_V001:{index}", index) for index in range(1, 151)]
        pools = {
            "branch1": _pool("branch1.result.v1", rows),
            "branch2": _pool("branch2.result.v1", []),
            "ocr": _pool("branch3.ocr.result.v1", []),
            "asr": _pool("branch3.asr.result.v1", []),
        }
        calls: list[str] = []
        service = self._service(pools, calls)
        reranker = _RecordingFusionReranker()
        service._encode_beit_queries = lambda _texts: (  # type: ignore[method-assign]
            np.ones((6, 768), dtype=np.float32),
            [],
            {"cache_hit": True},
        )
        service._get_reranker = lambda: reranker  # type: ignore[method-assign]
        expected_rrf, _ = fuse_branch_pools(
            deepcopy(pools),
            {"branch1": 0.40, "branch2": 0.30, "ocr": 0.15, "asr": 0.15},
        )
        expected_rrf_public = [
            materialize_fusion_candidate(item, include_rerank_fields=False) for item in expected_rrf
        ]
        expected_tail = expected_rrf_public[100:]
        expected_top_by_uid = {str(item["frame_uid"]): item for item in expected_rrf_public[:100]}
        response = service.execute(_query_bundle())
        self.assertEqual(calls, ["branch1", "branch2", "asr", "ocr"])
        self.assertEqual(len(reranker.selected_uids), 100)
        self.assertEqual(reranker.selected_uids, [f"L01_V001:{index}" for index in range(1, 101)])
        self.assertEqual(response["result_count"], 150)
        top = response["results"][:100]
        self.assertTrue(all(set(item["beit3_query_scores"]) == set(QUERY_ROLES) for item in top))
        self.assertTrue(
            all(
                item["rerank_formula"]
                == {
                    "beit3_weight": 0.25,
                    "previous_weight": 0.75,
                    "previous_score_field": "rrf_score",
                    "expression": "beit3_weight * normalized_beit3 + previous_weight * normalized_rrf",
                }
                for item in top
            )
        )
        self.assertTrue(
            all(
                abs(
                    item["final_score"]
                    - (0.25 * item["beit3_normalized"] + 0.75 * item["rrf_normalized"])
                )
                <= 1e-7
                for item in top
            )
        )
        expected_beit_normalized = {
            str(item["frame_uid"]): {
                "raw": item["beit3_raw_cosine"],
                "observed": True,
            }
            for item in top
        }
        expected_rrf_normalized = {
            str(item["frame_uid"]): {
                "raw": expected_top_by_uid[str(item["frame_uid"])]["rrf_score"],
                "observed": True,
            }
            for item in top
        }
        normalize_scores(expected_beit_normalized, "raw")
        normalize_scores(expected_rrf_normalized, "raw")
        for item in top:
            uid = str(item["frame_uid"])
            query_scores = item["beit3_query_scores"]
            winning_role = min(
                QUERY_ROLES,
                key=lambda role: (
                    -query_scores[role]["cosine"],
                    query_scores[role]["rank"],
                    QUERY_ROLES.index(role),
                ),
            )
            self.assertEqual(item["pre_rerank_rank"], expected_top_by_uid[uid]["pre_rerank_rank"])
            self.assertEqual(item["rrf_score"], expected_top_by_uid[uid]["rrf_score"])
            self.assertEqual(
                item["branch_provenance"], expected_top_by_uid[uid]["branch_provenance"]
            )
            self.assertEqual(item["beit3_best_query_role"], winning_role)
            self.assertAlmostEqual(
                item["beit3_raw_cosine"], query_scores[winning_role]["cosine"], places=7
            )
            self.assertAlmostEqual(
                item["beit3_normalized"],
                expected_beit_normalized[uid]["normalized_score"],
                places=7,
            )
            self.assertAlmostEqual(
                item["rrf_normalized"],
                expected_rrf_normalized[uid]["normalized_score"],
                places=7,
            )
        tail = response["results"][100:]
        self.assertEqual(
            [item["frame_uid"] for item in tail], [f"L01_V001:{index}" for index in range(101, 151)]
        )
        self.assertTrue(
            all("beit3_raw_cosine" not in item and "rank_delta" not in item for item in tail)
        )
        self.assertTrue(
            all(item["score"] == item["final_score"] == item["rrf_score"] for item in tail)
        )
        self.assertTrue(all(item["score_type"] == "weighted_rrf" for item in tail))
        self.assertEqual(tail, expected_tail)
        self.assertTrue(all("large_raw_stream_dump" not in item for item in response["results"]))
        self.assertTrue(
            all(
                "reranked_score" not in item and "rerank_score_type" not in item
                for item in response["results"]
            )
        )

    def test_empty_pools_return_without_encoder_or_reranker(self) -> None:
        pools = {
            "branch1": _pool("branch1.result.v1", []),
            "branch2": _pool("branch2.result.v1", []),
            "ocr": _pool("branch3.ocr.result.v1", []),
            "asr": _pool("branch3.asr.result.v1", []),
        }
        calls: list[str] = []
        service = self._service(pools, calls)
        service._encode_beit_queries = _FailIfUsed()  # type: ignore[method-assign]
        service._get_reranker = _FailIfUsed()  # type: ignore[method-assign]
        response = service.execute(_query_bundle())
        self.assertEqual(calls, ["branch1", "branch2", "asr", "ocr"])
        self.assertEqual(response["results"], [])
        self.assertEqual(response["result_count"], 0)
        self.assertFalse(response["reranking_applied"])

    def test_beit_query_encoding_failure_is_classified_as_beit3_failure(self) -> None:
        row = _frame("L01_V001:4", 1)
        pools = {
            "branch1": _pool("branch1.result.v1", [row]),
            "branch2": _pool("branch2.result.v1", []),
            "ocr": _pool("branch3.ocr.result.v1", []),
            "asr": _pool("branch3.asr.result.v1", []),
        }
        service = self._service(pools, [])

        def fail_encode(_texts):
            raise RuntimeError("worker failed to encode BEiT query")

        service._encode_beit_queries = fail_encode  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, r"^KIS_FUSION_BEIT3_FAILED: worker failed"):
            service.execute(_query_bundle())

    def test_cache_failures_are_classified_as_beit3_failures(self) -> None:
        row = _frame("L01_V001:4", 1)
        for stage in ("key", "get", "put"):
            with self.subTest(stage=stage):
                service = self._service(self._pools_with_rows([row]), [])
                service.branch1.cache = _FaultyCache(stage)
                service.branch2.beit_encoders = _TinyBeitEncoder()
                service._get_reranker = lambda: _FailIfUsed()  # type: ignore[method-assign]
                with self.assertRaisesRegex(RuntimeError, r"^KIS_FUSION_BEIT3_FAILED: cache"):
                    service.execute(_query_bundle())

    def test_beit_cache_hit_does_not_invoke_the_encoder(self) -> None:
        row = _frame("L01_V001:4", 1)
        service = self._service(self._pools_with_rows([row]), [])
        encoder = _TinyBeitEncoder()
        reranker = _RecordingFusionReranker()
        service.branch1.cache = _HitCache()
        service.branch2.beit_encoders = encoder
        service._get_reranker = lambda: reranker  # type: ignore[method-assign]
        response = service.execute(_query_bundle())
        self.assertEqual(encoder.encode_calls, 0)
        self.assertEqual(reranker.selected_uids, [str(row["frame_uid"])])
        self.assertTrue(response["rerank"]["cache_hit"])

    def test_final_beit_fallback_cache_uses_actual_cpu_device(self) -> None:
        service = self._service(self._pools_with_rows([]), [])
        encoder = _FallbackBeitEncoder()
        service.branch2.beit_encoders = encoder
        texts = [f"en {role}" for role in QUERY_ROLES]
        streams = [
            {"role": role, "language": "en", "text": text}
            for role, text in zip(QUERY_ROLES, texts, strict=True)
        ]
        with tempfile.TemporaryDirectory() as directory:
            cache = PersistentQueryEmbeddingCache(Path(directory) / "cache.sqlite3")
            try:
                service.branch1.cache = cache
                _vectors, _diagnostics, first_timing = service._encode_beit_queries(texts)
                key_args = {
                    "tokenizer_config": (
                        "languages=en;max_tokens=64;output=language_head;normalization=l2"
                    ),
                    "stream_contract": streams,
                }
                cpu_key = cache.key(
                    "beit3",
                    "test-revision",
                    texts,
                    device="cpu",
                    **key_args,
                )
                mps_key = cache.key(
                    "beit3",
                    "test-revision",
                    texts,
                    device="mps",
                    **key_args,
                )
                self.assertFalse(first_timing["cache_hit"])
                self.assertIsNotNone(cache.get(cpu_key))
                self.assertIsNone(cache.get(mps_key))

                _vectors, _diagnostics, second_timing = service._encode_beit_queries(texts)
                self.assertTrue(second_timing["cache_hit"])
                self.assertEqual(encoder.encode_calls, 1)
            finally:
                cache.close()

    def test_beit_vector_validation_failures_are_classified_at_fusion_boundary(self) -> None:
        row = _frame("L01_V001:4", 1)
        point_map = {str(row["frame_uid"]): 1}
        for label, vectors in (
            ("shape", np.ones((5, 768), dtype=np.float32)),
            ("non_finite", np.full((6, 768), np.nan, dtype=np.float32)),
            ("zero", np.zeros((6, 768), dtype=np.float32)),
        ):
            with self.subTest(label=label):
                service = self._service(self._pools_with_rows([row]), [])
                service._encode_beit_queries = lambda _texts, values=vectors: (  # type: ignore[method-assign]
                    values,
                    [{} for _ in range(6)],
                    self._valid_beit_timing(),
                )
                self._install_actual_reranker(
                    service, _RecordingQdrant({1: str(row["frame_uid"])}), point_map
                )
                with self.assertRaisesRegex(RuntimeError, r"^KIS_FUSION_BEIT3_FAILED:"):
                    service.execute(_query_bundle())

    def test_qdrant_evidence_failures_are_classified_at_fusion_boundary(self) -> None:
        row = _frame("L01_V001:4", 1)
        point_map = {str(row["frame_uid"]): 1}
        for mode in ("missing_point", "duplicate_point", "payload_uid_mismatch", "missing_stream"):
            with self.subTest(mode=mode):
                service = self._service(self._pools_with_rows([row]), [])
                service._encode_beit_queries = lambda _texts: (  # type: ignore[method-assign]
                    np.ones((6, 768), dtype=np.float32),
                    [{} for _ in range(6)],
                    self._valid_beit_timing(),
                )
                self._install_actual_reranker(
                    service,
                    _EvidenceQdrant({1: str(row["frame_uid"])}, mode),
                    point_map,
                )
                with self.assertRaisesRegex(RuntimeError, r"^KIS_FUSION_BEIT3_FAILED:"):
                    service.execute(_query_bundle())

    def test_invalid_reranker_output_is_fail_closed_as_beit3_failure(self) -> None:
        rows = [_frame(f"L01_V001:{index}", index) for index in range(1, 151)]

        class _InvalidReranker:
            def __init__(self, mode: str) -> None:
                self.mode = mode

            def rerank(self, candidates, _texts, *, top_k, **_kwargs):
                values, info = _RecordingFusionReranker().rerank(
                    candidates,
                    _texts,
                    top_k=top_k,
                )
                if self.mode == "wrong_type":
                    return "not-a-candidate-list", info
                if self.mode == "wrong_info":
                    return values, []
                if self.mode == "wrong_count":
                    return values[:-1], info
                if self.mode == "duplicate_uid":
                    values[1] = dict(values[0])
                    return values, info
                if self.mode == "tail_order":
                    return values[:top_k] + list(reversed(values[top_k:])), info
                if self.mode == "tail_promoted":
                    values[0], values[top_k] = values[top_k], values[0]
                    return values, info
                if self.mode == "missing_raw_cosine":
                    values[0].pop("beit3_raw_cosine")
                    return values, info
                if self.mode == "nonfinite_normalized":
                    values[0]["beit3_normalized"] = float("nan")
                    return values, info
                if self.mode == "out_of_range_normalized":
                    values[0]["rrf_normalized"] = 1.1
                    return values, info
                if self.mode == "missing_query_role":
                    values[0]["beit3_query_scores"].pop("keyword")
                    return values, info
                if self.mode == "extra_query_role":
                    values[0]["beit3_query_scores"]["unexpected"] = {
                        "cosine": 0.2,
                        "rank": 1,
                        "role": "unexpected",
                        "language": "en",
                    }
                    return values, info
                if self.mode == "mismatched_query_role":
                    values[0]["beit3_query_scores"]["original"]["role"] = "entity"
                    return values, info
                if self.mode == "non_english_query":
                    values[0]["beit3_query_scores"]["original"]["language"] = "vi"
                    return values, info
                if self.mode == "nonfinite_query_cosine":
                    values[0]["beit3_query_scores"]["original"]["cosine"] = float("inf")
                    return values, info
                if self.mode == "invalid_query_rank":
                    values[0]["beit3_query_scores"]["original"]["rank"] = 0
                    return values, info
                if self.mode == "invalid_best_role":
                    values[0]["beit3_best_query_role"] = "unexpected"
                    return values, info
                if self.mode == "non_english_best_language":
                    values[0]["beit3_best_query_language"] = "vi"
                    return values, info
                if self.mode == "invalid_score_type":
                    values[0]["score_type"] = "weighted_rrf"
                    return values, info
                if self.mode == "invalid_rank_delta":
                    values[0]["rank_delta"] = 999
                    return values, info
                if self.mode == "invalid_final_rank":
                    values[0]["rank"] = 999
                    return values, info
                if self.mode == "invalid_formula":
                    values[0]["rerank_formula"]["expression"] = "wrong"
                    return values, info
                if self.mode == "invalid_weight":
                    values[0]["rerank_formula"]["beit3_weight"] = 0.4
                    return values, info
                if self.mode == "invalid_final_score":
                    values[0]["final_score"] = 0.123
                    values[0]["score"] = 0.123
                    return values, info
                if self.mode == "score_mismatch":
                    values[0]["score"] = values[0]["final_score"] + 0.01
                    return values, info
                if self.mode == "duplicate_pre_rank":
                    values[1]["pre_rerank_rank"] = values[0]["pre_rerank_rank"]
                    return values, info
                if self.mode == "missing_pre_rank":
                    values[0].pop("pre_rerank_rank")
                    return values, info
                if self.mode == "swapped_pre_ranks":
                    values[0]["pre_rerank_rank"], values[1]["pre_rerank_rank"] = (
                        values[1]["pre_rerank_rank"],
                        values[0]["pre_rerank_rank"],
                    )
                    for rank, item in enumerate(values[:top_k], 1):
                        item["rank_delta"] = item["pre_rerank_rank"] - rank
                    return values, info
                if self.mode == "top_rrf_score":
                    values[0]["rrf_score"] += 0.01
                    return values, info
                if self.mode == "top_branch_ranks":
                    values[0]["branch_ranks"] = {"branch1": 999}
                    return values, info
                if self.mode == "top_contributions":
                    values[0]["rrf_contributions"] = {"branch1": 999.0}
                    return values, info
                if self.mode == "top_provenance":
                    values[0]["branch_provenance"] = {"branch1": {"best_model": "mutated"}}
                    return values, info
                if self.mode == "in_place_tail_mutation":
                    for index, value in enumerate(values):
                        candidates[index].clear()
                        candidates[index].update(value)
                    candidates[top_k]["branch_provenance"] = {"branch1": {"best_model": "mutated"}}
                    return candidates, info
                if self.mode == "raw_not_query_max":
                    values[0]["beit3_raw_cosine"] -= 0.1
                    return values, info
                if self.mode == "tie_wrong_best_role":
                    scores = values[0]["beit3_query_scores"]
                    scores["entity"]["cosine"] = scores["original"]["cosine"]
                    scores["entity"]["rank"] = scores["original"]["rank"]
                    values[0]["beit3_best_query_role"] = "entity"
                    return values, info
                if self.mode == "fake_beit_normalization":
                    values[0]["beit3_normalized"] = 0.5
                    values[0]["final_score"] = (
                        0.25 * values[0]["beit3_normalized"] + 0.75 * values[0]["rrf_normalized"]
                    )
                    values[0]["score"] = values[0]["final_score"]
                    return values, info
                if self.mode == "fake_rrf_normalization":
                    values[0]["rrf_normalized"] = 0.5
                    values[0]["final_score"] = (
                        0.25 * values[0]["beit3_normalized"] + 0.75 * values[0]["rrf_normalized"]
                    )
                    values[0]["score"] = values[0]["final_score"]
                    return values, info
                if self.mode == "unsorted_final_scores":
                    values[0], values[1] = values[1], values[0]
                    for rank, item in enumerate(values[:top_k], 1):
                        item["rank"] = rank
                        item["rank_delta"] = item["pre_rerank_rank"] - rank
                    return values, info
                if self.mode == "tail_rrf_score":
                    values[top_k]["rrf_score"] += 0.01
                    values[top_k]["score"] = values[top_k]["rrf_score"]
                    values[top_k]["final_score"] = values[top_k]["rrf_score"]
                    return values, info
                if self.mode == "tail_branch_ranks":
                    values[top_k]["branch_ranks"] = {"branch1": 999}
                    return values, info
                if self.mode == "tail_contributions":
                    values[top_k]["rrf_contributions"] = {"branch1": 999.0}
                    return values, info
                if self.mode == "tail_provenance":
                    values[top_k]["branch_provenance"] = {"branch1": {"best_model": "mutated"}}
                    return values, info
                if self.mode == "tail_beit_field":
                    values[top_k]["rank_delta"] = 0
                    return values, info
                raise AssertionError(f"unexpected mode {self.mode}")

        for mode in (
            "wrong_type",
            "wrong_info",
            "wrong_count",
            "duplicate_uid",
            "tail_order",
            "tail_promoted",
            "missing_raw_cosine",
            "nonfinite_normalized",
            "out_of_range_normalized",
            "missing_query_role",
            "extra_query_role",
            "mismatched_query_role",
            "non_english_query",
            "nonfinite_query_cosine",
            "invalid_query_rank",
            "invalid_best_role",
            "non_english_best_language",
            "invalid_score_type",
            "invalid_rank_delta",
            "invalid_final_rank",
            "invalid_formula",
            "invalid_weight",
            "invalid_final_score",
            "score_mismatch",
            "duplicate_pre_rank",
            "missing_pre_rank",
            "swapped_pre_ranks",
            "top_rrf_score",
            "top_branch_ranks",
            "top_contributions",
            "top_provenance",
            "in_place_tail_mutation",
            "raw_not_query_max",
            "tie_wrong_best_role",
            "fake_beit_normalization",
            "fake_rrf_normalization",
            "unsorted_final_scores",
            "tail_rrf_score",
            "tail_branch_ranks",
            "tail_contributions",
            "tail_provenance",
            "tail_beit_field",
        ):
            with self.subTest(mode=mode):
                service = self._service(self._pools_with_rows(rows), [])
                service._encode_beit_queries = lambda _texts: (  # type: ignore[method-assign]
                    np.ones((6, 768), dtype=np.float32),
                    [{} for _ in range(6)],
                    self._valid_beit_timing(),
                )
                service._get_reranker = lambda value=mode: _InvalidReranker(value)  # type: ignore[method-assign]
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^KIS_FUSION_BEIT3_FAILED: output_validation:",
                ):
                    service.execute(_query_bundle())

    def test_shared_lock_busy_is_fail_fast(self) -> None:
        pools = {
            "branch1": _pool("branch1.result.v1", []),
            "branch2": _pool("branch2.result.v1", []),
            "ocr": _pool("branch3.ocr.result.v1", []),
            "asr": _pool("branch3.asr.result.v1", []),
        }
        calls: list[str] = []
        service = self._service(pools, calls)
        self.assertTrue(service.search_lock.acquire(blocking=False))
        try:
            with self.assertRaisesRegex(RuntimeError, "KIS_FUSION_SEARCH_BUSY"):
                service.execute(_query_bundle())
        finally:
            service.search_lock.release()

    def test_ordered_batch_reuses_single_lock_and_runs_every_full_kis_event(self) -> None:
        pools = {
            "branch1": _pool("branch1.result.v1", []),
            "branch2": _pool("branch2.result.v1", []),
            "ocr": _pool("branch3.ocr.result.v1", []),
            "asr": _pool("branch3.asr.result.v1", []),
        }
        calls: list[str] = []
        service = self._service(pools, calls)
        health_checks = 0

        def ready_health() -> dict[str, bool]:
            nonlocal health_checks
            health_checks += 1
            return {"ready": True}

        service.health = ready_health  # type: ignore[method-assign]
        responses = service.execute_batch([_query_bundle(), _query_bundle()])

        self.assertEqual(len(responses), 2)
        self.assertEqual(
            calls,
            ["branch1", "branch2", "asr", "ocr"] * 2,
        )
        self.assertEqual(health_checks, 1)
        self.assertTrue(service.search_lock.acquire(blocking=False))
        service.search_lock.release()

    def test_ordered_batch_validates_all_events_before_acquiring_the_lock(self) -> None:
        pools = {
            "branch1": _pool("branch1.result.v1", []),
            "branch2": _pool("branch2.result.v1", []),
            "ocr": _pool("branch3.ocr.result.v1", []),
            "asr": _pool("branch3.asr.result.v1", []),
        }
        calls: list[str] = []
        service = self._service(pools, calls)
        invalid = {**_query_bundle(), "queries": []}
        with self.assertRaises(ValueError):
            service.execute_batch([_query_bundle(), invalid])
        self.assertEqual(calls, [])
        self.assertTrue(service.search_lock.acquire(blocking=False))
        service.search_lock.release()

    def test_unready_aggregate_stops_before_any_branch_execution(self) -> None:
        pools = {
            "branch1": _pool("branch1.result.v1", []),
            "branch2": _pool("branch2.result.v1", []),
            "ocr": _pool("branch3.ocr.result.v1", []),
            "asr": _pool("branch3.asr.result.v1", []),
        }
        calls: list[str] = []
        service = self._service(pools, calls)
        service.health = lambda: {"ready": False, "production_ready": False}  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "KIS_FUSION_NOT_READY"):
            service.execute(_query_bundle())
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
