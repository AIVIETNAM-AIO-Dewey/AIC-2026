"""BEiT-3 COCO cosine validation for a bounded candidate pool.

This is deliberately a dual-encoder scorer.  It does not invoke BEiT-3
cross-attention, a learned scoring head, or the vision encoder at request
time.  Branch 2 and the final KIS fusion use the same implementation while
providing a different field for the pre-rerank score.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from ...infrastructure.qdrant import QdrantHttpClient
from ...branches.branch1.contracts import QUERY_ROLES
from ...infrastructure.scoring import normalize_scores, normalize_weights


MAX_RERANK_CANDIDATES = 100


def _strict_int(value: Any, field: str) -> int:
    """Accept integral values without silently truncating floats/bools."""

    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, float):
        if not math.isfinite(value) or value != math.trunc(value):
            raise ValueError(f"{field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be an integer") from error


class Beit3CosineReranker:
    """Score at most 100 canonical frames with BEiT-3 COCO cosine."""

    def __init__(
        self,
        qdrant: QdrantHttpClient,
        encoder: Any,
        frame_point_ids: dict[str, int],
    ) -> None:
        self.qdrant = qdrant
        self.encoder = encoder
        self.frame_point_ids = frame_point_ids

    def rerank(
        self,
        candidates: list[dict[str, Any]],
        texts: list[str],
        *,
        top_k: int,
        weights: dict[str, float],
        text_vectors: np.ndarray | None = None,
        tokenizer_diagnostics: list[dict[str, Any]] | None = None,
        previous_score_field: str = "hybrid_score",
        previous_rank_field: str = "hybrid_rank",
        previous_score_label: str = "hybrid",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if len(texts) != 6:
            raise ValueError("BEiT-3 reranking requires six English query variants")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("BEiT-3 reranking query variants must not be empty")
        requested_top_k = _strict_int(top_k, "top_k")
        if not 1 <= requested_top_k <= MAX_RERANK_CANDIDATES:
            raise ValueError(
                f"BEiT-3 rerank top_k must be in 1..{MAX_RERANK_CANDIDATES}"
            )
        # Validate the weight contract even for an empty candidate pool.  This
        # keeps direct callers from bypassing the fixed 25/75 (or Branch-2's
        # normalized equivalent) blend merely because there is nothing to
        # score, while still avoiding any encoder/Qdrant work for an empty
        # search result.
        normalized_weights = normalize_weights(weights, ("beit3", "previous"))
        if not candidates:
            return [], {
                "candidate_count": 0,
                "rerank_count": 0,
                "weights": normalized_weights,
                "text_encoder_output": "language_head",
                "query_language": "en",
                "checkpoint_task": "BEiT-3 COCO Retrieval",
                "scoring": "cosine",
                "previous_score_field": previous_score_field,
                "tokenizer_diagnostics": list(tokenizer_diagnostics or []),
                "qdrant_ms": 0.0,
                "scoring_ms": 0.0,
            }
        selected = candidates[:requested_top_k]
        selected_uids = [str(item.get("frame_uid") or "") for item in selected]
        if any(not frame_uid for frame_uid in selected_uids):
            raise ValueError("BEiT-3 reranking requires a canonical frame_uid")
        if len(set(selected_uids)) != len(selected_uids):
            raise ValueError("BEiT-3 reranking received duplicate canonical frame_uid values")

        if text_vectors is None:
            vectors, diagnostics = self.encoder.encode("beit3", texts)
        else:
            vectors = np.asarray(text_vectors, dtype=np.float32)
            diagnostics = tokenizer_diagnostics or []
        if diagnostics and len(diagnostics) != 6:
            raise ValueError("BEiT-3 tokenizer diagnostics must contain six rows")
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (6, 768):
            raise ValueError(f"BEiT-3 text vectors must have shape (6, 768), got {vectors.shape}")
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("BEiT-3 text vectors must be finite and non-zero")

        ids = [self.frame_point_ids.get(str(item["frame_uid"])) for item in selected]
        missing_uids = [
            str(item["frame_uid"])
            for item, point_id in zip(selected, ids, strict=True)
            if point_id is None
        ]
        if missing_uids:
            raise ValueError(f"BEiT-3 point mapping is missing {len(missing_uids)} candidates")
        valid_ids = [int(point_id) for point_id in ids if point_id is not None]
        if len(set(valid_ids)) != len(valid_ids):
            raise ValueError("BEiT-3 frame mapping contains duplicate point IDs")
        expected_ids = set(valid_ids)
        expected_frame_by_id = {
            int(point_id): str(item["frame_uid"])
            for item, point_id in zip(selected, valid_ids, strict=True)
        }

        # Filtered Qdrant queries are used only to obtain exact cosine scores
        # for the selected IDs.  The has_id filter prevents a second global
        # retrieval stage and makes missing evidence fail closed.
        query_filter = {"must": [{"has_id": valid_ids}]}
        qdrant_started = time.perf_counter()
        streams = [
            self.qdrant.query(
                "aic_beit3_frames",
                "beit3",
                vector,
                len(valid_ids),
                query_filter,
            )
            for vector in vectors
        ]
        qdrant_ms = round((time.perf_counter() - qdrant_started) * 1000.0, 2)
        scoring_started = time.perf_counter()
        for stream in streams:
            returned_ids = {int(point["id"]) for point in stream}
            if returned_ids != expected_ids or len(stream) != len(expected_ids):
                missing = sorted(expected_ids - returned_ids)
                raise RuntimeError(
                    f"BEiT-3 filtered scoring returned incomplete evidence; missing={missing[:10]}"
                )
            for point in stream:
                point_id = int(point["id"])
                payload_uid = str((point.get("payload") or {}).get("frame_uid") or "")
                if payload_uid != expected_frame_by_id[point_id]:
                    raise RuntimeError(
                        f"BEiT-3 point identity mismatch for point {point_id}: {payload_uid!r}"
                    )

        scores: dict[int, dict[str, Any]] = {}
        for role, stream in zip(QUERY_ROLES, streams, strict=True):
            for rank, point in enumerate(stream, 1):
                point_id = int(point["id"])
                value = float(point["score"])
                if not math.isfinite(value):
                    raise RuntimeError(f"BEiT-3 point {point_id} returned a non-finite cosine")
                record = scores.setdefault(
                    point_id,
                    {
                        "query_scores": {},
                        "best": -math.inf,
                        "best_role": None,
                        "best_rank": None,
                    },
                )
                record["query_scores"][role] = {
                    "cosine": value,
                    "rank": rank,
                    "role": role,
                    "language": "en",
                }
                if value > record["best"] or (
                    value == record["best"]
                    and (record["best_rank"] is None or rank < record["best_rank"])
                ):
                    record["best"] = value
                    record["best_role"] = role
                    record["best_rank"] = rank

        score_items = {
            str(point_id): {
                "raw": value["best"],
                "observed": math.isfinite(value["best"]),
            }
            for point_id, value in scores.items()
        }
        normalize_scores(score_items, "raw")
        previous: dict[str, dict[str, Any]] = {}
        for item in selected:
            uid = str(item["frame_uid"])
            if previous_score_field not in item:
                raise ValueError(
                    f"BEiT-3 reranking candidate {uid} is missing {previous_score_field}"
                )
            previous[uid] = {
                "raw": float(item[previous_score_field]),
                "observed": True,
            }
        normalize_scores(previous, "raw")
        reranked: list[dict[str, Any]] = []
        for original_rank, item in enumerate(selected, 1):
            point_id = self.frame_point_ids.get(str(item["frame_uid"]))
            evidence = scores.get(point_id or -1)
            beit_raw = float(evidence["best"]) if evidence else None
            beit_norm = float(score_items.get(str(point_id), {}).get("normalized_score", 0.0))
            previous_norm = float(previous[str(item["frame_uid"])] ["normalized_score"])
            final_score = (
                normalized_weights["beit3"] * beit_norm
                + normalized_weights["previous"] * previous_norm
            )
            copied = dict(item)
            copied.update(
                {
                    "pre_rerank_rank": original_rank,
                    "beit3_raw_cosine": beit_raw,
                    "beit3_normalized": beit_norm,
                    f"{previous_score_label}_normalized": previous_norm,
                    "beit3_best_query_role": evidence.get("best_role") if evidence else None,
                    "beit3_best_query_language": "en" if evidence else None,
                    "beit3_query_scores": evidence.get("query_scores", {}) if evidence else {},
                    "reranked_score": final_score,
                    "final_score": final_score,
                    "score": final_score,
                    "score_type": "beit3_coco_cosine_blend",
                    "rerank_score_type": "beit3_coco_cosine_blend",
                    "rerank_formula": {
                        "beit3_weight": normalized_weights["beit3"],
                        "previous_weight": normalized_weights["previous"],
                        "previous_score_field": previous_score_field,
                        "expression": (
                            "beit3_weight * normalized_beit3 + "
                            f"previous_weight * normalized_{previous_score_label}"
                        ),
                    },
                }
            )
            reranked.append(copied)
        reranked.sort(
            key=lambda value: (
                -float(value["reranked_score"]),
                value.get(previous_rank_field, math.inf),
                value["frame_uid"],
            )
        )
        for index, item in enumerate(reranked, 1):
            item["rank"] = index
            item["rank_delta"] = int(item["pre_rerank_rank"]) - index

        output = reranked + [dict(item) for item in candidates[requested_top_k:]]
        for index, item in enumerate(output, 1):
            item["rank"] = index
        return output, {
            "candidate_count": len(candidates),
            "rerank_count": len(selected),
            "weights": normalized_weights,
            "text_encoder_output": "language_head",
            "query_language": "en",
            "checkpoint_task": "BEiT-3 COCO Retrieval",
            "scoring": "cosine",
            "previous_score_field": previous_score_field,
            "tokenizer_diagnostics": diagnostics,
            "qdrant_ms": qdrant_ms,
            "scoring_ms": round((time.perf_counter() - scoring_started) * 1000.0, 2),
        }


__all__ = ["Beit3CosineReranker", "MAX_RERANK_CANDIDATES"]
