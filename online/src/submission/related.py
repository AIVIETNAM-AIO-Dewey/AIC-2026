"""Frame-to-frame recommendations isolated from the text-query pipeline."""

from __future__ import annotations

import math
from typing import Any

RRF_K = 60
RELATED_STREAMS = (
    ("siglip2", "aic_frames", "siglip2", 0.45),
    ("metaclip2", "aic_frames", "metaclip2", 0.30),
    ("beit3", "aic_beit3_frames", "beit3", 0.25),
)


def _frame_uid(item: dict[str, Any]) -> str:
    video_id = str(item.get("video_id") or "").strip().upper().replace("-", "_")
    try:
        frame_idx = int(item["frame_idx"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return ""
    return f"{video_id}:{frame_idx}" if video_id and frame_idx >= 0 else ""


def fuse_related_pools(
    pools: dict[str, list[dict[str, Any]]],
    *,
    seed_uid: str,
    limit: int,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Fuse aligned visual-neighbour ranks with deterministic weighted RRF."""

    if not 1 <= int(limit) <= 99:
        raise ValueError("related-frame limit must be between 1 and 99")
    weights = {name: weight for name, _collection, _vector, weight in RELATED_STREAMS}
    unexpected = set(pools) - set(weights)
    if unexpected:
        raise ValueError(f"unexpected related-frame pools: {sorted(unexpected)}")

    fused: dict[str, dict[str, Any]] = {}
    point_to_uid: dict[int, str] = {}
    for stream, _collection, _vector, _weight in RELATED_STREAMS:
        seen: set[str] = set()
        for rank, point in enumerate(pools.get(stream, []), 1):
            payload = point.get("payload") if isinstance(point, dict) else None
            if not isinstance(payload, dict):
                continue
            uid = _frame_uid(payload)
            if not uid or uid == seed_uid or uid in seen:
                continue
            seen.add(uid)
            try:
                point_id = int(point["id"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            previous_uid = point_to_uid.get(point_id)
            if previous_uid is not None and previous_uid != uid:
                raise ValueError(
                    f"Qdrant point {point_id} identifies both {previous_uid} and {uid}"
                )
            point_to_uid[point_id] = uid
            row = fused.setdefault(
                uid,
                {
                    **payload,
                    "point_id": point_id,
                    "global_idx": point_id,
                    "frame_uid": uid,
                    "related_ranks": {},
                    "related_contributions": {},
                },
            )
            if _frame_uid(row) != uid or int(row["point_id"]) != point_id:
                raise ValueError(f"canonical identity mismatch for related frame {uid}")
            contribution = weights[stream] / (int(rrf_k) + rank)
            row["related_ranks"][stream] = rank
            row["related_contributions"][stream] = contribution

    results = list(fused.values())
    for row in results:
        score = sum(float(value) for value in row["related_contributions"].values())
        if not math.isfinite(score):
            raise ValueError("related-frame fusion produced a non-finite score")
        row["score"] = score
        row["related_score"] = score
        row["score_type"] = "visual_neighbor_weighted_rrf"
        row["retrieval_modality"] = "related_frame"
        row["source"] = "auto-related"
        row["validation"] = "canonical"
        row["submission_string"] = f"{row['video_id']}, {int(row['frame_idx'])}"
    results.sort(
        key=lambda row: (
            -float(row["related_score"]),
            -len(row["related_ranks"]),
            str(row["frame_uid"]),
        )
    )
    selected = results[: int(limit)]
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
        row["related_rank"] = rank
    return selected


class RelatedFrameSearch:
    """Recommend indexed frames from stored visual vectors only."""

    def __init__(self, qdrant: Any, metadata: Any) -> None:
        self.qdrant = qdrant
        self.metadata = metadata

    def _canonical_frame(self, video_id: str, frame_idx: int) -> dict[str, Any] | None:
        for frame in self.metadata.video_frames(video_id):
            if int(frame.get("frame_idx", -1)) == int(frame_idx):
                return dict(frame)
        return None

    def execute(self, video_id: str, frame_idx: int, limit: int) -> dict[str, Any]:
        canonical = self._canonical_frame(video_id, frame_idx)
        if canonical is None:
            raise ValueError("Seed frame is not present in the canonical index")
        seed_uid = _frame_uid(canonical)
        seed_point = self.qdrant.find_frame_point("aic_frames", video_id, frame_idx)
        if seed_point is None:
            raise RuntimeError("RELATED_FRAME_SEED_NOT_INDEXED")
        point_id = int(seed_point["id"])
        if _frame_uid(seed_point.get("payload") or {}) != seed_uid:
            raise RuntimeError("RELATED_FRAME_SEED_IDENTITY_MISMATCH")

        pool_limit = min(500, max(100, int(limit) * 4 + 1))
        pools: dict[str, list[dict[str, Any]]] = {}
        for stream, collection, vector_name, _weight in RELATED_STREAMS:
            pools[stream] = self.qdrant.query_by_id(
                collection,
                vector_name,
                point_id,
                pool_limit,
            )
        raw_results = fuse_related_pools(pools, seed_uid=seed_uid, limit=limit)

        results: list[dict[str, Any]] = []
        for raw in raw_results:
            frame = self._canonical_frame(str(raw["video_id"]), int(raw["frame_idx"]))
            if frame is None or _frame_uid(frame) != str(raw["frame_uid"]):
                raise RuntimeError("RELATED_FRAME_CANONICAL_LOOKUP_FAILED")
            # Canonical organizer metadata owns identity and timing. Qdrant
            # contributes only its global point plus neighbour provenance.
            results.append(
                {
                    **raw,
                    **frame,
                    "point_id": int(raw["point_id"]),
                    "global_idx": int(raw["point_id"]),
                    "frame_uid": _frame_uid(frame),
                    "submission_string": (f"{frame['video_id']}, {int(frame['frame_idx'])}"),
                    "validation": "canonical",
                }
            )

        source_frame = {
            **canonical,
            "point_id": point_id,
            "global_idx": point_id,
            "frame_uid": seed_uid,
            "validation": "canonical",
        }
        return {
            "schema_version": "submission.related-frames.v1",
            "algorithm": "stored-vector weighted RRF",
            "query_pipeline_invoked": False,
            "rrf_k": RRF_K,
            "weights": {name: weight for name, _c, _v, weight in RELATED_STREAMS},
            "seed": source_frame,
            "result_count": len(results),
            "results": results,
        }


__all__ = ["RELATED_STREAMS", "RRF_K", "RelatedFrameSearch", "fuse_related_pools"]
