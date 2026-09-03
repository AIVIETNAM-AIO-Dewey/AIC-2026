"""Branch-1 per-model normalization and weighted score fusion."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contracts import EXPECTED_FRAMES, MODEL_SPECS, normalize_model_weights


def _stream_identity(stream: str) -> tuple[str, str | None]:
    """Split a canonical ``role:language`` key while keeping legacy role keys."""

    role, separator, language = str(stream).partition(":")
    return role, language or None


def normalize_model_candidates(candidates: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not candidates:
        return candidates
    scores = np.asarray([item["raw_score"] for item in candidates.values()], dtype=np.float64)
    mean = float(scores.mean())
    std = float(scores.std())
    for item in candidates.values():
        normalized = 0.5 if std < 1e-6 else 1.0 / (1.0 + math.exp(-max(-4.0, min(4.0, (float(item["raw_score"]) - mean) / std))))
        item["normalized_score"] = normalized
        item["normalization_mean"] = mean
        item["normalization_std"] = std
    return candidates


def aggregate_model_streams(
    roles: tuple[str, ...], streams: list[list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for role, points in zip(roles, streams, strict=True):
        role_name, language = _stream_identity(role)
        for rank, point in enumerate(points, start=1):
            payload = dict(point.get("payload") or {})
            frame_uid = str(payload.get("frame_uid") or "")
            if not frame_uid:
                raise ValueError("Qdrant Branch-1 point is missing canonical frame_uid")
            try:
                point_id = int(point["id"])
                video_id = str(payload["video_id"])
                frame_idx = int(payload["frame_idx"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "Qdrant Branch-1 point is missing canonical frame identity"
                ) from error
            if not 1 <= point_id <= EXPECTED_FRAMES:
                raise ValueError(
                    f"Qdrant Branch-1 point id {point_id} is outside the canonical frame range"
                )
            if frame_uid != f"{video_id}:{frame_idx}":
                raise ValueError(
                    f"Qdrant Branch-1 point {point_id} has inconsistent frame_uid {frame_uid!r}"
                )
            score = float(point["score"])
            if not math.isfinite(score):
                raise ValueError("Qdrant Branch-1 point has a non-finite cosine score")
            candidate = candidates.setdefault(
                frame_uid,
                {
                    "point_id": point_id,
                    "frame_uid": frame_uid,
                    "payload": payload,
                    "raw_score": -math.inf,
                    "best_query_role": None,
                    "best_query_language": None,
                    "best_query_rank": None,
                    "query_scores": {},
                },
            )
            if int(candidate["point_id"]) != point_id:
                raise ValueError(
                    f"Frame {frame_uid} maps to multiple Qdrant point IDs "
                    f"({candidate['point_id']} and {point_id})"
                )
            candidate["query_scores"][role] = {
                "role": role_name,
                "language": language,
                "cosine": score,
                "rank": rank,
                "observed": True,
            }
            current_rank = candidate["best_query_rank"] or math.inf
            if score > candidate["raw_score"] or (
                score == candidate["raw_score"]
                and (rank, str(role)) < (current_rank, str(candidate["best_query_role"] or ""))
            ):
                candidate["raw_score"] = score
                candidate["best_query_role"] = role_name
                candidate["best_query_language"] = language
                candidate["best_query_rank"] = rank
    # Make the audit shape explicit for candidates that only appeared in one
    # or more of the six approximate top-k streams.  Missing streams are not
    # treated as zero cosine during the max operation; they are simply marked
    # unobserved and receive the model-level missing score at fusion time.
    for candidate in candidates.values():
        for role in roles:
            role_name, language = _stream_identity(role)
            candidate["query_scores"].setdefault(
                role,
                {
                    "role": role_name,
                    "language": language,
                    "cosine": None,
                    "rank": None,
                    "observed": False,
                },
            )
    return normalize_model_candidates(candidates)


def fuse_model_candidates(model_candidates: dict[str, dict[str, dict[str, Any]]], weights: dict[str, float], final_top_k: int) -> list[dict[str, Any]]:
    weights = normalize_model_weights(weights)
    frame_uids = set().union(*(set(values) for values in model_candidates.values()))
    results: list[dict[str, Any]] = []
    for frame_uid in frame_uids:
        payload: dict[str, Any] = {}
        canonical_point_id: int | None = None
        provenance: dict[str, Any] = {}
        best_rank = math.inf
        best_model = None
        best_model_order = math.inf
        best_query_role = None
        best_query_language = None
        final_score = 0.0
        for model_order, model_name in enumerate(MODEL_SPECS):
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
            item_payload = dict(item["payload"])
            if payload:
                for field in ("frame_uid", "video_id", "frame_idx", "keyframe_n"):
                    if (
                        field in payload
                        and field in item_payload
                        and payload[field] != item_payload[field]
                    ):
                        raise ValueError(
                            f"Branch-1 model payload mismatch for {frame_uid}: "
                            f"{field}={payload[field]!r}!={item_payload[field]!r}"
                        )
            else:
                payload = item_payload
            item_point_id = int(item["point_id"])
            if canonical_point_id is None:
                canonical_point_id = item_point_id
            elif canonical_point_id != item_point_id:
                raise ValueError(
                    f"Branch-1 model point ID mismatch for {frame_uid}: "
                    f"{canonical_point_id}!={item_point_id}"
                )
            candidate_rank = int(item["best_query_rank"])
            if (candidate_rank, model_order) < (best_rank, best_model_order):
                best_rank = candidate_rank
                best_model = model_name
                best_model_order = model_order
                best_query_role = item.get("best_query_role")
                best_query_language = item.get("best_query_language")
            normalized = float(item["normalized_score"])
            final_score += weights[model_name] * normalized
            provenance[model_name] = {
                "observed": True,
                "raw_cosine": float(item["raw_score"]),
                "normalized_score": normalized,
                "best_query_role": item["best_query_role"],
                "best_query_language": item.get("best_query_language"),
                "best_query_rank": int(item["best_query_rank"]),
                "query_scores": item["query_scores"],
                "normalization_mean": float(item["normalization_mean"]),
                "normalization_std": float(item["normalization_std"]),
            }
        results.append({
            **payload,
            "frame_uid": frame_uid,
            "global_idx": next(
                (
                    int(model_candidates[model_name][frame_uid]["point_id"])
                    for model_name in MODEL_SPECS
                    if frame_uid in model_candidates.get(model_name, {})
                ),
                None,
            ),
            "final_score": final_score,
            "score": final_score,
            "score_type": "weighted_zsigmoid_fusion",
            "best_stream_rank": None if math.isinf(best_rank) else int(best_rank),
            "best_model": best_model,
            "best_query_role": best_query_role,
            "best_query_language": best_query_language,
            "model_provenance": provenance,
        })
    results.sort(
        key=lambda item: (
            -item["final_score"],
            item["best_stream_rank"] if item["best_stream_rank"] is not None else math.inf,
            list(MODEL_SPECS).index(item["best_model"]) if item["best_model"] in MODEL_SPECS else math.inf,
            item["frame_uid"],
        )
    )
    for rank, item in enumerate(results[:final_top_k], start=1):
        item["rank"] = rank
        item["final_score"] = round(float(item["final_score"]), 8)
        item["score"] = item["final_score"]
    return results[:final_top_k]

__all__ = ["aggregate_model_streams", "fuse_model_candidates", "normalize_model_candidates"]
