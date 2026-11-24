"""Ordered event orchestration over the unchanged full KIS fusion search.

Every event is focused into a complete six-role bilingual query bundle and
then executed by :class:`KisFusionSearch`.  This module only links the returned
frame pools in time; it never changes branch weights or the per-frame KIS
score.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Any

from ..branch1.contracts import QUERY_ROLES

MAX_KIS_TEMPORAL_EVENTS = 6
MAX_KIS_TEMPORAL_SEQUENCES = 100
MAX_FOCUSED_QUERY_CHARACTERS = 4096


def _focused_text(event_text: str, context_text: str, *, event_only: bool) -> str:
    value = event_text if event_only else f"{event_text}. Context: {context_text}"
    # Event text is deliberately first because every encoder applies a token
    # limit.  Character bounding also preserves the public query contract.
    return value[:MAX_FOCUSED_QUERY_CHARACTERS].strip()


def focus_event_query_bundle(
    query_bundle: dict[str, Any],
    event: dict[str, Any],
    *,
    shared_context_only: bool = False,
) -> dict[str, Any]:
    """Create one deterministic full-KIS bundle focused on a single event.

    ``shared_context_only`` is reserved for ordered KIS orchestration.  It
    prevents details from the parent entity/action/keyword roles from leaking
    into every event while retaining the parent's common scene context.  The
    default preserves the existing scoped-video behavior for other callers.
    """

    if not isinstance(query_bundle, dict):
        raise ValueError("Ordered KIS search requires a branch1.query.v1 query bundle")
    queries = query_bundle.get("queries")
    if query_bundle.get("schema_version") != "branch1.query.v1" or not isinstance(queries, list):
        raise ValueError("Ordered KIS search requires a branch1.query.v1 query bundle")
    by_role = {str(item.get("role") or ""): item for item in queries if isinstance(item, dict)}
    if set(by_role) != set(QUERY_ROLES) or len(by_role) != len(QUERY_ROLES):
        raise ValueError("Ordered KIS base bundle must contain every query role exactly once")

    description = str(event.get("description") or "").strip()
    event_vi = str(event.get("vi") or description).strip()
    event_en = str(event.get("en") or description).strip()
    if not description or not event_vi or not event_en:
        raise ValueError("Every ordered KIS event requires description, vi, and en text")

    shared_vi = str(by_role["context"].get("vi") or "").strip()
    shared_en = str(by_role["context"].get("en") or "").strip()
    focused_queries: list[dict[str, str]] = []
    for role in QUERY_ROLES:
        base = by_role[role]
        base_vi = str(base.get("vi") or "").strip()
        base_en = str(base.get("en") or "").strip()
        if not base_vi or not base_en:
            raise ValueError("Ordered KIS base bundle contains an empty query stream")
        focused_queries.append(
            {
                "role": role,
                "vi": _focused_text(
                    event_vi,
                    shared_vi if shared_context_only else base_vi,
                    event_only=role in {"original", "action"},
                ),
                "en": _focused_text(
                    event_en,
                    shared_en if shared_context_only else base_en,
                    event_only=role in {"original", "action"},
                ),
            }
        )
    return {"schema_version": "branch1.query.v1", "queries": focused_queries}


def _canonical_event_frame(value: Any, *, event_order: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"KIS event E{event_order} returned a non-object frame")
    frame = deepcopy(value)
    try:
        video_id = str(frame["video_id"])
        frame_idx = int(frame["frame_idx"])
        timestamp = float(frame["pts_time_s"])
        rank = int(frame["rank"])
        score_value = frame.get("final_score")
        score = float(frame["score"] if score_value is None else score_value)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"KIS event E{event_order} returned incomplete frame identity") from error
    if (
        not video_id
        or frame_idx < 0
        or rank < 1
        or not math.isfinite(timestamp)
        or timestamp < 0
        or not math.isfinite(score)
    ):
        raise ValueError(f"KIS event E{event_order} returned an invalid frame")
    frame_uid = str(frame.get("frame_uid") or "")
    if frame_uid != f"{video_id}:{frame_idx}":
        raise ValueError(f"KIS event E{event_order} returned inconsistent frame identity")
    frame["video_id"] = video_id
    frame["frame_idx"] = frame_idx
    frame["pts_time_s"] = timestamp
    frame["rank"] = rank
    frame["frame_uid"] = frame_uid
    frame["submission_string"] = f"{video_id}, {frame_idx}"
    frame["score"] = score
    frame["final_score"] = score
    frame["retrieval_modality"] = "kis_fusion"
    return frame


def order_kis_event_results(
    *,
    events: list[dict[str, Any]],
    event_responses: list[dict[str, Any]],
    top_k_sequences: int,
    max_gap_seconds: float | None,
    task_type: str,
    execution_time_ms: float,
) -> dict[str, Any]:
    """Find one best strict chronological KIS path per common video."""

    if len(events) != len(event_responses) or not 2 <= len(events) <= MAX_KIS_TEMPORAL_EVENTS:
        raise ValueError("Ordered KIS event responses do not match the event contract")
    if not 1 <= int(top_k_sequences) <= MAX_KIS_TEMPORAL_SEQUENCES:
        raise ValueError(f"top_k_sequences must be between 1 and {MAX_KIS_TEMPORAL_SEQUENCES}")
    if max_gap_seconds is not None and (
        not math.isfinite(float(max_gap_seconds)) or float(max_gap_seconds) <= 0
    ):
        raise ValueError("max_gap_seconds must be positive or null")

    ordered_pairs = sorted(
        ((dict(event), response) for event, response in zip(events, event_responses, strict=True)),
        key=lambda value: int(value[0]["order"]),
    )
    ordered_events = [event for event, _response in ordered_pairs]
    if [int(event["order"]) for event in ordered_events] != list(range(1, len(events) + 1)):
        raise ValueError("Ordered KIS event orders must be contiguous from 1")

    grouped_pools: list[dict[str, list[dict[str, Any]]]] = []
    event_pool_audit: list[dict[str, Any]] = []
    common_video_ids: set[str] | None = None
    for event, response in ordered_pairs:
        if not isinstance(response, dict) or response.get("fusion_applied") is not True:
            raise ValueError(f"KIS event E{event['order']} did not return a fused result pool")
        raw_results = response.get("results")
        if not isinstance(raw_results, list):
            raise ValueError(f"KIS event E{event['order']} returned an invalid result list")

        by_video: dict[str, list[dict[str, Any]]] = {}
        for raw_result in raw_results:
            result = _canonical_event_frame(raw_result, event_order=int(event["order"]))
            by_video.setdefault(str(result["video_id"]), []).append(result)
        for frames in by_video.values():
            frames.sort(
                key=lambda frame: (
                    float(frame["pts_time_s"]),
                    int(frame["frame_idx"]),
                    int(frame["rank"]),
                )
            )
        grouped_pools.append(by_video)
        video_ids = set(by_video)
        common_video_ids = video_ids if common_video_ids is None else common_video_ids & video_ids
        event_pool_audit.append(
            {
                "order": int(event["order"]),
                "description": str(event["description"]),
                "query": str(event.get("en") or event["description"]),
                "event_order": int(event["order"]),
                "event_description": str(event["description"]),
                "event_query": str(event.get("en") or event["description"]),
                "query_source": "events[].en",
                "modality": "kis_fusion",
                "score_type": "kis_final_score",
                "score_description": "Unchanged final score from the complete four-branch KIS run",
                "top_k_requested": int(response.get("final_top_k", 150)),
                "result_count": len(raw_results),
                "candidate_video_count": len(by_video),
                "branch_pool_counts": dict(response.get("branch_pool_counts") or {}),
                "execution_time_ms": float((response.get("timing") or {}).get("total_ms", 0.0)),
            }
        )

    def follows(previous: dict[str, Any], current: dict[str, Any]) -> bool:
        gap = float(current["pts_time_s"]) - float(previous["pts_time_s"])
        return (
            gap > 0
            and int(current["frame_idx"]) > int(previous["frame_idx"])
            and (max_gap_seconds is None or gap <= float(max_gap_seconds))
        )

    def extend(state: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": [*state["path"], frame],
            "minimum_score": min(float(state["minimum_score"]), float(frame["score"])),
            "score_sum": float(state["score_sum"]) + float(frame["score"]),
            "rank_sum": int(state["rank_sum"]) + int(frame["rank"]),
        }

    def state_key(state: dict[str, Any]) -> tuple[Any, ...]:
        path = state["path"]
        return (
            -float(state["minimum_score"]),
            -float(state["score_sum"]),
            int(state["rank_sum"]),
            float(path[-1]["pts_time_s"]) - float(path[0]["pts_time_s"]),
            tuple(int(frame["frame_idx"]) for frame in path),
        )

    sequences: list[dict[str, Any]] = []
    for video_id in sorted(common_video_ids or set()):
        candidates_by_event = [pool[video_id] for pool in grouped_pools]
        states = [
            {
                "path": [frame],
                "minimum_score": float(frame["score"]),
                "score_sum": float(frame["score"]),
                "rank_sum": int(frame["rank"]),
            }
            for frame in candidates_by_event[0]
        ]
        for candidates in candidates_by_event[1:]:
            next_states: list[dict[str, Any]] = []
            for candidate in candidates:
                valid = [
                    extend(state, candidate)
                    for state in states
                    if follows(state["path"][-1], candidate)
                ]
                if valid:
                    next_states.append(min(valid, key=state_key))
            states = next_states
            if not states:
                break
        if not states:
            continue
        best = min(states, key=state_key)
        matched_events: list[dict[str, Any]] = []
        for event, frame in zip(ordered_events, best["path"], strict=True):
            matched_events.append(
                {
                    **frame,
                    "event_order": int(event["order"]),
                    "event_description": str(event["description"]),
                    "event_query": str(event.get("en") or event["description"]),
                }
            )
        timestamps = [float(frame["pts_time_s"]) for frame in matched_events]
        minimum_score = round(float(best["minimum_score"]), 8)
        mean_score = round(float(best["score_sum"]) / len(matched_events), 8)
        anchor = matched_events[0]
        sequences.append(
            {
                "video_id": video_id,
                "video_path_rank": 1,
                "minimum_event_score": minimum_score,
                "mean_event_score": mean_score,
                "sequence_score": minimum_score,
                "score_type": "minimum_then_mean_kis_final_score",
                "ranking_values": {
                    "sequence_score": minimum_score,
                    "minimum_event_score": minimum_score,
                    "mean_event_score": mean_score,
                },
                "global_rank_sum": int(best["rank_sum"]),
                "span_seconds": round(timestamps[-1] - timestamps[0], 6),
                "gaps_seconds": [
                    round(current - previous, 6)
                    for previous, current in zip(timestamps, timestamps[1:], strict=False)
                ],
                "matched_frames": [int(frame["frame_idx"]) for frame in matched_events],
                "timestamps": timestamps,
                "matched_events": matched_events,
                "anchor_event_order": 1,
                "anchor_rank": int(anchor["rank"]),
                "anchor_score": float(anchor["score"]),
                "anchor_frame": dict(anchor),
                "submission_string": str(anchor["submission_string"]),
            }
        )

    sequences.sort(
        key=lambda sequence: (
            -float(sequence["sequence_score"]),
            -float(sequence["mean_event_score"]),
            int(sequence["global_rank_sum"]),
            float(sequence["span_seconds"]),
            str(sequence["video_id"]),
            tuple(sequence["matched_frames"]),
        )
    )
    for rank, sequence in enumerate(sequences, 1):
        sequence["reservoir_rank"] = rank
    returned = sequences[: int(top_k_sequences)]
    reserve = sequences[int(top_k_sequences) : 500]
    for rank, sequence in enumerate(returned, 1):
        sequence["rank"] = rank

    return {
        "schema_version": "kis.temporal.result.v1",
        "task_type": task_type,
        "operation": "ordered_kis_fusion",
        "experiment_mode": "full_kis_per_event_with_temporal_linking",
        "modality": "kis_fusion",
        "events": ordered_events,
        "event_count": len(ordered_events),
        "top_k_per_event": 150,
        "top_k_sequences": int(top_k_sequences),
        "event_candidate_reservoir_size": 150,
        "sequence_reservoir_size": min(500, len(sequences)),
        "sequence_reservoir_count": min(500, len(sequences)),
        "computed_sequence_count": len(sequences),
        "paths_per_video": 1,
        "path_beam_applied": False,
        "path_search_mode": "deterministic_best_prefix_dp",
        "max_gap_seconds": max_gap_seconds,
        "anchor_event_order": 1,
        "anchor_query": "",
        "anchor_query_applied": False,
        "anchor_pool": None,
        "event_pools": event_pool_audit,
        "intersection_video_count": len(common_video_ids or set()),
        "monotonic_video_count": len(sequences),
        "ordered_sequence_count": len(sequences),
        "result_count": len(returned),
        "score_type": "minimum_then_mean_kis_final_score",
        "score_policy": (
            "Rank valid same-video paths by their weakest unchanged KIS final score, "
            "then mean KIS score, lower event-rank sum, shorter span, video_id, and frame indexes."
        ),
        "query_focus_policy": (
            "Each event replaces original/action and prefixes only the shared parent context "
            "into the remaining roles; parent event-specific details are not copied."
        ),
        "complete_sequence_required": True,
        "event_fusion_applied": True,
        "cross_modal_fusion_applied": True,
        "fusion_applied": True,
        "reranking_applied": True,
        "temporal_constraint_applied": True,
        "frame_index_base": 0,
        "frame_identity_policy": "canonical_indexed_source_frame_idx",
        "ordering_fields": ["frame_idx", "pts_time_s"],
        "sequences": returned,
        "reserve_sequences": reserve,
        "execution_time_ms": round(float(execution_time_ms), 2),
    }


class KisTemporalFusionSearch:
    """Run one complete KIS search per event, then order the frame pools."""

    def __init__(self, fusion_searcher: Any) -> None:
        self.fusion_searcher = fusion_searcher

    def execute(
        self,
        *,
        query_bundle: dict[str, Any],
        events: list[dict[str, Any]],
        branch_weights: dict[str, float] | None,
        top_k_sequences: int,
        max_gap_seconds: float | None,
        task_type: str,
        health_already_checked: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        event_bundles = [
            focus_event_query_bundle(query_bundle, event, shared_context_only=True)
            for event in events
        ]
        event_responses = self.fusion_searcher.execute_batch(
            event_bundles,
            branch_weights,
            _health_already_checked=health_already_checked,
        )
        return order_kis_event_results(
            events=events,
            event_responses=event_responses,
            top_k_sequences=top_k_sequences,
            max_gap_seconds=max_gap_seconds,
            task_type=task_type,
            execution_time_ms=(time.perf_counter() - started) * 1000.0,
        )


__all__ = [
    "KisTemporalFusionSearch",
    "focus_event_query_bundle",
    "order_kis_event_results",
]
