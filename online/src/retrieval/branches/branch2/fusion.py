"""Dense/sparse score normalization and hybrid fusion."""

from __future__ import annotations

import math
from typing import Any

from .contracts import normalize_scores, normalize_weights


def fuse_dense_sparse(
    dense: dict[str, dict[str, Any]],
    sparse: dict[str, dict[str, Any]],
    weights: dict[str, float],
    top_k: int,
) -> list[dict[str, Any]]:
    normalized = normalize_weights(weights, ("dense", "sparse"))
    frame_uids = set(dense) | set(sparse)
    records: dict[str, dict[str, Any]] = {}
    for uid in frame_uids:
        dense_item = dense.get(uid)
        sparse_item = sparse.get(uid)
        if dense_item is not None and sparse_item is not None:
            for field in ("frame_uid", "global_idx", "video_id", "frame_idx"):
                dense_value = dense_item.get(field)
                sparse_value = sparse_item.get(field)
                if (
                    dense_value is not None
                    and sparse_value is not None
                    and dense_value != sparse_value
                ):
                    raise ValueError(
                        f"Dense/sparse canonical identity mismatch for {uid}: "
                        f"{field}={dense_value!r}!={sparse_value!r}"
                    )
        # Both sources carry canonical metadata. Preserve it for sparse-only
        # results instead of fabricating timestamp or image path from indexes.
        item = dict(dense_item or sparse_item or {"frame_uid": uid})
        item["dense_raw"] = float(dense_item["dense_raw"]) if dense_item else None
        item["dense_observed"] = dense_item is not None
        item["sparse_raw"] = float(sparse_item["sparse_raw"]) if sparse_item else None
        item["sparse_observed"] = sparse_item is not None
        records[uid] = item
    dense_obs = {uid: {"raw": item["dense_raw"], "observed": True} for uid, item in dense.items()}
    sparse_obs = {uid: {"raw": item["sparse_raw"], "observed": True} for uid, item in sparse.items()}
    normalize_scores(dense_obs, "raw")
    normalize_scores(sparse_obs, "raw")
    for uid, item in records.items():
        dense_item = dense.get(uid)
        sparse_item = sparse.get(uid)
        item["dense_normalized"] = float(dense_obs.get(uid, {}).get("normalized_score", 0.0))
        item["sparse_normalized"] = float(sparse_obs.get(uid, {}).get("normalized_score", 0.0))
        item["dense_normalization_mean"] = dense_obs.get(uid, {}).get("normalization_mean")
        item["dense_normalization_std"] = dense_obs.get(uid, {}).get("normalization_std")
        item["sparse_normalization_mean"] = sparse_obs.get(uid, {}).get("normalization_mean")
        item["sparse_normalization_std"] = sparse_obs.get(uid, {}).get("normalization_std")
        item["hybrid_score"] = normalized["dense"] * item["dense_normalized"] + normalized["sparse"] * item["sparse_normalized"]
        item["score"] = item["hybrid_score"]
        item["score_type"] = "dam_dense_bm25_hybrid"
        if dense_item:
            item["dam_winner"] = dense_item.get("dam_winner")
            item["dense_best_query_role"] = dense_item.get("dense_best_query_role")
            item["dense_best_query_language"] = dense_item.get("dense_best_query_language", "en")
            item["dense_query_scores"] = dense_item.get("dense_query_scores", {})
            item["dense_rank"] = dense_item.get("dense_rank")
        if sparse_item:
            item["sparse_winner"] = sparse_item.get("sparse_winner")
            item["sparse_best_query_role"] = sparse_item.get("sparse_best_query_role")
            item["sparse_best_query_language"] = sparse_item.get("sparse_best_query_language", "en")
            item["sparse_bm25_raw"] = sparse_item.get("sparse_bm25_raw")
            item["sparse_rank"] = sparse_item.get("sparse_rank")
            item["sparse_query_scores"] = sparse_item.get("sparse_query_scores", {})
        item["hybrid_provenance"] = {
            "dense": None if dense_item is None else {
                "rank": dense_item.get("dense_rank"),
                "raw": dense_item.get("dense_raw"),
                "best_query_role": dense_item.get("dense_best_query_role"),
                "best_query_language": dense_item.get("dense_best_query_language", "en"),
            },
            "sparse": None if sparse_item is None else {
                "rank": sparse_item.get("sparse_rank"),
                "raw": sparse_item.get("sparse_raw"),
                "bm25_raw": sparse_item.get("sparse_bm25_raw"),
                "best_query_role": sparse_item.get("sparse_best_query_role"),
                "best_query_language": sparse_item.get("sparse_best_query_language", "en"),
            },
            "weights": normalized,
        }
    ordered = sorted(records.values(), key=lambda value: (-float(value["hybrid_score"]), min(value.get("dense_rank") or math.inf, value.get("sparse_rank") or math.inf), value["frame_uid"]))
    for rank, item in enumerate(ordered[:top_k], 1):
        item["hybrid_rank"] = rank
        item["rank"] = rank
    return ordered[:top_k]
