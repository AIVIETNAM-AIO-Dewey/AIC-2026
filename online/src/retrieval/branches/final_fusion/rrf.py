"""Rank-only weighted reciprocal-rank fusion for canonical frames."""

from __future__ import annotations

import math
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from .contracts import (
    BRANCH_POOL_LIMITS,
    BRANCH_SCHEMA_VERSIONS,
    FINAL_TOP_K,
    RRF_K,
    normalize_branch_weights,
)


_IDENTITY_FIELDS = (
    "point_id",
    "global_idx",
    "video_id",
    "frame_idx",
    "keyframe_n",
    "image_relpath",
)
_FLOAT_IDENTITY_FIELDS = ("pts_time_s", "fps")
_CANONICAL_FRAME_COUNT = 247_956


def _is_safe_relative_path(value: Any) -> bool:
    """Accept only a canonical, repository-relative image path.

    Branch responses are later used by the UI/media handlers.  An absolute or
    parent-traversal path is not a canonical frame identity and must not be
    allowed to win the cross-branch merge merely because its UID is valid.
    """

    raw = str(value or "")
    if not raw or "\x00" in raw:
        return False
    normalized = raw.replace("\\", "/")
    try:
        return (
            not PurePosixPath(normalized).is_absolute()
            and not PureWindowsPath(raw).is_absolute()
            and ".." not in PurePosixPath(normalized).parts
        )
    except (TypeError, ValueError):
        return False


def _canonical_int(value: Any, field: str) -> int:
    """Parse an integer identity without silently truncating a bad value."""

    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, float):
        if not math.isfinite(value) or value != math.trunc(value):
            raise ValueError(f"{field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be an integer") from error


def _positive_int(value: Any, field: str) -> int:
    parsed = _canonical_int(value, field)
    if parsed < 1:
        raise ValueError(f"{field} must be positive")
    return parsed


def _identity_value(item: dict[str, Any], field: str) -> Any:
    if field == "point_id" and field not in item and "global_idx" in item:
        return item.get("global_idx")
    return item.get(field)


def _validate_identity(item: dict[str, Any], branch: str, rank: int) -> str:
    uid = str(item.get("frame_uid") or "")
    if not uid:
        raise ValueError(f"{branch} result rank {rank} is missing frame_uid")
    required = ("video_id", "frame_idx", "keyframe_n", "image_relpath", "point_id")
    if any(_identity_value(item, field) is None for field in required):
        raise ValueError(f"{branch} result {uid} is missing canonical identity")
    try:
        point_id = _canonical_int(_identity_value(item, "point_id"), "point_id")
        # ``global_idx`` is the legacy spelling still emitted by some
        # standalone adapters.  When a response includes both aliases they
        # must describe the same canonical point; otherwise a malformed pool
        # could pass UID checks while contributing provenance for a different
        # frame to the final RRF voter.
        if "global_idx" in item and item.get("global_idx") is not None:
            raw_global_idx = item.get("global_idx")
            if isinstance(raw_global_idx, bool):
                raise ValueError("global_idx must be an integer")
            if isinstance(raw_global_idx, float) and (
                not math.isfinite(raw_global_idx)
                or raw_global_idx != math.trunc(raw_global_idx)
            ):
                raise ValueError("global_idx must be an integer")
            global_idx = _canonical_int(raw_global_idx, "global_idx")
            if global_idx != point_id:
                raise ValueError("point_id and global_idx do not identify the same point")
        frame_idx = _canonical_int(item["frame_idx"], "frame_idx")
        keyframe_n = _canonical_int(item["keyframe_n"], "keyframe_n")
        pts_time_s = float(item.get("pts_time_s"))
        fps = float(item.get("fps"))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{branch} result {uid} has invalid canonical identity") from error
    if (
        point_id < 1
        or point_id > _CANONICAL_FRAME_COUNT
        or frame_idx < 0
        or keyframe_n < 1
        or not math.isfinite(pts_time_s)
        or pts_time_s < 0
        or not math.isfinite(fps)
        or fps <= 0
        or not str(item.get("video_id") or "")
        or not _is_safe_relative_path(item.get("image_relpath"))
    ):
        raise ValueError(f"{branch} result {uid} has invalid canonical identity")
    if uid != f"{item['video_id']}:{frame_idx}":
        raise ValueError(f"{branch} result {uid} has inconsistent video/frame identity")
    return uid


def _same_identity(first: dict[str, Any], second: dict[str, Any], uid: str) -> None:
    for field in _IDENTITY_FIELDS:
        left = _identity_value(first, field)
        right = _identity_value(second, field)
        if left is None or right is None:
            continue
        if field in {"point_id", "global_idx", "frame_idx", "keyframe_n"}:
            try:
                mismatch = _canonical_int(left, field) != _canonical_int(right, field)
            except ValueError as error:
                raise ValueError(f"Canonical identity mismatch for {uid}: {field}") from error
        else:
            mismatch = str(left) != str(right)
        if mismatch:
            raise ValueError(f"Canonical identity mismatch for {uid}: {field}={left!r}!={right!r}")
    for field in _FLOAT_IDENTITY_FIELDS:
        left = first.get(field)
        right = second.get(field)
        if left is not None and right is not None:
            try:
                if abs(float(left) - float(right)) > 1e-6:
                    raise ValueError(
                        f"Canonical identity mismatch for {uid}: {field}={left!r}!={right!r}"
                    )
            except (TypeError, ValueError, OverflowError) as error:
                if isinstance(error, ValueError) and str(error).startswith("Canonical identity"):
                    raise
                raise ValueError(f"Canonical identity mismatch for {uid}: {field}") from error


def _normalized_score(item: dict[str, Any], branch: str) -> float:
    fields = {
        "branch1": ("final_score", "normalized_score", "score"),
        "branch2": ("reranked_score", "score", "hybrid_score"),
        "ocr": ("ocr_normalized_score", "normalized_score", "score"),
        "asr": ("asr_normalized_score", "normalized_score", "score"),
    }[branch]
    for field in fields:
        if field in item and item[field] is not None:
            try:
                value = float(item[field])
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value):
                return value
    return 0.0


def validate_branch_pool(branch: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate a standalone branch response before it contributes to RRF."""

    if not isinstance(payload, dict):
        raise ValueError(f"{branch} branch response must be an object")
    if payload.get("schema_version") != BRANCH_SCHEMA_VERSIONS[branch]:
        raise ValueError(f"{branch} branch response schema is unsupported")
    if payload.get("future_fusion_eligible") is not True:
        raise ValueError(f"{branch} branch response is not eligible for fusion")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{branch} branch response results must be a list")
    if len(results) > BRANCH_POOL_LIMITS[branch]:
        raise ValueError(f"{branch} branch exceeded its pool gate")
    if "gate_top_k" in payload:
        try:
            gate_top_k = _canonical_int(payload["gate_top_k"], "gate_top_k")
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{branch} branch gate_top_k is invalid") from error
        if gate_top_k != BRANCH_POOL_LIMITS[branch]:
            raise ValueError(
                f"{branch} branch gate_top_k must be {BRANCH_POOL_LIMITS[branch]}"
            )
    if "result_count" in payload:
        try:
            reported_count = _canonical_int(payload["result_count"], "result_count")
            if reported_count != len(results):
                raise ValueError(f"{branch} branch result_count does not match results")
        except (TypeError, ValueError, OverflowError) as error:
            if isinstance(error, ValueError) and "result_count does not match" in str(error):
                raise
            raise ValueError(f"{branch} branch result_count is invalid") from error
    seen: set[str] = set()
    seen_point_ids: dict[int, str] = {}
    for expected_rank, item in enumerate(results, 1):
        if not isinstance(item, dict):
            raise ValueError(f"{branch} result must be an object")
        uid = _validate_identity(item, branch, expected_rank)
        if uid in seen:
            raise ValueError(f"{branch} branch contains duplicate frame_uid {uid}")
        seen.add(uid)
        point_id = _canonical_int(_identity_value(item, "point_id"), "point_id")
        previous_uid = seen_point_ids.get(point_id)
        if previous_uid is not None and previous_uid != uid:
            raise ValueError(
                f"{branch} branch reuses canonical point_id {point_id} "
                f"for {previous_uid} and {uid}"
            )
        seen_point_ids[point_id] = uid
        if "rank" not in item:
            raise ValueError(f"{branch} result {uid} is missing rank")
        try:
            reported_rank = _canonical_int(item["rank"], "rank")
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{branch} result {uid} has invalid rank") from error
        if reported_rank != expected_rank:
            raise ValueError(f"{branch} result {uid} rank is not sequential")
    return results


def fuse_branch_pools(
    pools: dict[str, dict[str, Any]],
    weights: dict[str, float],
    *,
    rrf_k: int = RRF_K,
    top_k: int = FINAL_TOP_K,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Merge validated branch responses using rank-only weighted RRF."""

    rrf_k_value = _positive_int(rrf_k, "rrf_k")
    top_k_value = _positive_int(top_k, "top_k")
    if not 1 <= top_k_value <= FINAL_TOP_K:
        raise ValueError(f"fusion top_k must be between 1 and {FINAL_TOP_K}")
    if not isinstance(pools, dict):
        raise ValueError("fusion pools must be an object")
    unexpected = set(pools) - set(BRANCH_POOL_LIMITS)
    if unexpected:
        raise ValueError(f"fusion received unexpected branch pools: {sorted(unexpected)}")
    try:
        weights = normalize_branch_weights(weights)
        branch_results = {
            branch: validate_branch_pool(branch, pools[branch])
            for branch in BRANCH_POOL_LIMITS
        }
    except KeyError as error:
        raise ValueError(f"fusion is missing required branch pool: {error.args[0]}") from error
    by_uid: dict[str, dict[str, Any]] = {}
    point_to_uid: dict[int, str] = {}
    for branch, results in branch_results.items():
        for rank, item in enumerate(results, 1):
            uid = str(item["frame_uid"])
            point_id = _canonical_int(_identity_value(item, "point_id"), "point_id")
            previous_point_uid = point_to_uid.get(point_id)
            if previous_point_uid is not None and previous_point_uid != uid:
                raise ValueError(
                    f"Canonical point_id {point_id} is associated with both "
                    f"{previous_point_uid} and {uid}"
                )
            point_to_uid[point_id] = uid
            existing = by_uid.get(uid)
            if existing is None:
                existing = {
                    **dict(item),
                    "frame_uid": uid,
                    "branch_ranks": {},
                    "rrf_contributions": {},
                    "branch_normalized_scores": {},
                    "branch_provenance": {},
                }
                # Branch-1/Branch-2 textual adapters historically exposed the
                # canonical point as ``global_idx``.  Materialize both names
                # at the fusion boundary so the final contract is stable no
                # matter which voter encountered the UID first.
                canonical_point_id = _identity_value(item, "point_id")
                if canonical_point_id is not None:
                    existing["point_id"] = int(canonical_point_id)
                    existing.setdefault("global_idx", int(canonical_point_id))
                by_uid[uid] = existing
            else:
                _same_identity(existing, item, uid)
            contribution = float(weights[branch]) / (rrf_k_value + rank)
            existing["branch_ranks"][branch] = rank
            existing["rrf_contributions"][branch] = contribution
            existing["branch_normalized_scores"][branch] = _normalized_score(item, branch)
            existing["branch_provenance"][branch] = item

    fused: list[dict[str, Any]] = []
    for item in by_uid.values():
        ranks = item["branch_ranks"]
        # Make missing-voter contributions explicit zeros in the audit
        # contract.  branch_ranks remains sparse because an absent voter has
        # no rank, while score maps are total over the four voters.
        item["rrf_contributions"] = {
            branch: float(item["rrf_contributions"].get(branch, 0.0))
            for branch in BRANCH_POOL_LIMITS
        }
        item["branch_normalized_scores"] = {
            branch: float(item["branch_normalized_scores"].get(branch, 0.0))
            for branch in BRANCH_POOL_LIMITS
        }
        normalized = item["branch_normalized_scores"]
        item["rrf_score"] = sum(item["rrf_contributions"].values())
        # Keep the generic score fields useful to downstream consumers while
        # preserving the rank-only RRF value as the canonical score.
        item["score"] = float(item["rrf_score"])
        item["final_score"] = float(item["rrf_score"])
        item["score_type"] = "weighted_rrf"
        item["branch_agreement_count"] = len(ranks)
        item["observed_branches"] = [branch for branch in BRANCH_POOL_LIMITS if branch in ranks]
        item["weighted_normalized_score"] = sum(
            float(weights[branch]) * float(normalized.get(branch, 0.0))
            for branch in BRANCH_POOL_LIMITS
        )
        item["best_branch_rank"] = min(ranks.values()) if ranks else math.inf
        fused.append(item)
    fused.sort(
        key=lambda item: (
            -float(item["rrf_score"]),
            -int(item["branch_agreement_count"]),
            -float(item["weighted_normalized_score"]),
            int(item["best_branch_rank"]),
            str(item["frame_uid"]),
        )
    )
    selected = fused[:top_k_value]
    for rank, item in enumerate(selected, 1):
        item["pre_rerank_rank"] = rank
        item["rank"] = rank
    return selected, {branch: len(results) for branch, results in branch_results.items()}


__all__ = ["fuse_branch_pools", "validate_branch_pool"]
