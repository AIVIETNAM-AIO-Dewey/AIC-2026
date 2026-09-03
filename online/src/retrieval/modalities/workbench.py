"""Orchestrate four independent modality searches without cross-modal fusion."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from online.src.contracts.query import ParsedQuery

MAX_TEMPORAL_EVENT_CANDIDATES = 1_000
MAX_TEMPORAL_RETURNED_SEQUENCES = 100
MAX_TEMPORAL_SEQUENCE_RESERVOIR = 500
MAX_TEMPORAL_PATHS_PER_VIDEO = 10
MAX_TEMPORAL_PATH_BEAM_WIDTH = 2_048


class IndependentModalitySearch:
    """Run every applicable modality and retain four isolated result pools."""

    def __init__(
        self,
        *,
        searcher: Any,
        registry: Any,
        dam_match_threshold: float = 0.50,
    ) -> None:
        if not -1.0 <= dam_match_threshold <= 1.0:
            raise ValueError("dam_match_threshold must be between -1.0 and 1.0")
        self.searcher = searcher
        self.registry = registry
        self.dam_match_threshold = dam_match_threshold

    @staticmethod
    def _not_run_pool(
        *,
        modality: str,
        display_name: str,
        query: str | list[str],
        query_source: str,
        score_type: str,
        score_description: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "modality": modality,
            "display_name": display_name,
            "status": "not_run",
            "reason": reason,
            "query": query,
            "query_source": query_source,
            "score_type": score_type,
            "score_description": score_description,
            "result_count": 0,
            "execution_time_ms": 0.0,
            "results": [],
        }

    @staticmethod
    def _run_pool(
        *,
        modality: str,
        display_name: str,
        query: str | list[str],
        query_source: str,
        score_type: str,
        score_description: str,
        search: Callable[[], list[dict[str, Any]]],
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        results = search()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        pool = {
            "modality": modality,
            "display_name": display_name,
            "status": "ok",
            "reason": "",
            "query": query,
            "query_source": query_source,
            "score_type": score_type,
            "score_description": score_description,
            "result_count": len(results),
            "execution_time_ms": round(elapsed_ms, 2),
            "results": results,
        }
        if diagnostics:
            pool["query_diagnostics"] = diagnostics
        return pool

    def search(
        self,
        parsed_query: ParsedQuery,
        *,
        top_k: int,
    ) -> dict[str, dict[str, Any]]:
        """Return isolated SigLIP, DAM, OCR, and ASR result pools."""
        pools: dict[str, dict[str, Any]] = {}

        visual_query = parsed_query.global_scene_en.strip()
        if visual_query:
            diagnostics_fn = getattr(self.registry, "siglip_text_diagnostics", None)
            visual_diagnostics = diagnostics_fn(visual_query) if diagnostics_fn else None
            pools["siglip"] = self._run_pool(
                modality="siglip",
                display_name="SigLIP visual scene",
                query=visual_query,
                query_source="global_scene_en",
                score_type="cosine",
                score_description="Raw cosine between query text and full-frame image",
                search=lambda: self.searcher.search_visual(
                    self.registry.embed_siglip_text(visual_query), top_k=top_k
                ),
                diagnostics=visual_diagnostics,
            )
        else:
            pools["siglip"] = self._not_run_pool(
                modality="siglip",
                display_name="SigLIP visual scene",
                query="",
                query_source="global_scene_en",
                score_type="cosine",
                score_description="Raw cosine between query text and full-frame image",
                reason="The parsed query has no global_scene_en value.",
            )

        object_queries = [query.strip() for query in parsed_query.objects_en if query.strip()]
        if object_queries:

            def run_dam() -> list[dict[str, Any]]:
                vectors = self.registry.embed_bge_text(object_queries)
                return self.searcher.search_dam(
                    [vectors[index] for index in range(len(object_queries))],
                    object_queries,
                    top_k=top_k,
                    match_threshold=self.dam_match_threshold,
                )

            pools["dam"] = self._run_pool(
                modality="dam",
                display_name="DAM detected objects",
                query=object_queries,
                query_source="objects_en",
                score_type="mean_best_region_cosine",
                score_description=(
                    "Mean of the best region cosine for each object query; "
                    f"a {self.dam_match_threshold:.2f} threshold labels evidence only "
                    "and does not change ranking"
                ),
                search=run_dam,
            )
        else:
            pools["dam"] = self._not_run_pool(
                modality="dam",
                display_name="DAM detected objects",
                query=[],
                query_source="objects_en",
                score_type="mean_best_region_cosine",
                score_description=(
                    "Mean of the best region cosine for each object query; "
                    f"a {self.dam_match_threshold:.2f} threshold labels evidence only "
                    "and does not change ranking"
                ),
                reason="The parsed query has no objects_en values.",
            )

        ocr_keywords = [keyword.strip() for keyword in parsed_query.ocr_keywords if keyword.strip()]
        if ocr_keywords:
            pools["ocr"] = self._run_pool(
                modality="ocr",
                display_name="OCR on-screen text",
                query=ocr_keywords,
                query_source="ocr_keywords",
                score_type="keyword_match_ratio",
                score_description="Matched query keywords divided by all OCR query keywords",
                search=lambda: self.searcher.search_ocr(ocr_keywords, top_k=top_k),
            )
        else:
            pools["ocr"] = self._not_run_pool(
                modality="ocr",
                display_name="OCR on-screen text",
                query=[],
                query_source="ocr_keywords",
                score_type="keyword_match_ratio",
                score_description="Matched query keywords divided by all OCR query keywords",
                reason="The parsed query has no ocr_keywords values.",
            )

        speech_query = parsed_query.speech_vi.strip()
        speech_source = "speech_vi"
        if speech_query:
            pools["asr"] = self._run_pool(
                modality="asr",
                display_name="ASR spoken speech",
                query=speech_query,
                query_source=speech_source,
                score_type="bm25_ngram",
                score_description="SQLite FTS5 Okapi BM25 combined with token and adjacent n-gram coverage",
                search=lambda: self.searcher.search_speech(speech_query, top_k=top_k),
            )
        else:
            pools["asr"] = self._not_run_pool(
                modality="asr",
                display_name="ASR spoken speech",
                query="",
                query_source=speech_source,
                score_type="bm25_ngram",
                score_description="SQLite FTS5 Okapi BM25 combined with token and adjacent n-gram coverage",
                reason=(
                    "The query contains no explicit speech, narration, or spoken topic; "
                    "ASR was not run to avoid searching audio with a visual description."
                ),
            )

        return pools

    def search_temporal_intersection(
        self,
        *,
        events: list[dict[str, Any]],
        anchor_query: str | None = None,
        top_k_per_event: int = 300,
        top_k_sequences: int = 20,
        max_gap_seconds: float | None = 30.0,
        anchor_event_order: int | None = None,
        paths_per_video: int = 1,
        sequence_reservoir_size: int | None = None,
        path_beam_width: int | None = None,
        path_diversity_min_events: int = 1,
    ) -> dict[str, Any]:
        """Intersect raw SigLIP event pools and find monotonic paths per video.

        Every event remains an independent global SigLIP search. The explicit
        same-modality ranking uses an optional shared-scene anchor, then maximizes
        the weakest selected event cosine and their arithmetic mean after
        same-video and chronological filtering. Defaults deliberately retain one
        path per video and the existing ranking semantics. Multi-path mode uses a
        bounded beam and deterministic path diversity so a larger candidate
        reservoir cannot grow combinatorially.
        """
        if not 2 <= len(events) <= 6:
            raise ValueError("Temporal intersection requires between 2 and 6 events")
        if top_k_per_event < 1 or top_k_sequences < 1:
            raise ValueError("Temporal search limits must be positive")
        if top_k_per_event > MAX_TEMPORAL_EVENT_CANDIDATES:
            raise ValueError(
                f"top_k_per_event cannot exceed {MAX_TEMPORAL_EVENT_CANDIDATES}"
            )
        if top_k_sequences > MAX_TEMPORAL_RETURNED_SEQUENCES:
            raise ValueError(
                f"top_k_sequences cannot exceed {MAX_TEMPORAL_RETURNED_SEQUENCES}"
            )
        if max_gap_seconds is not None and max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive or null")
        if not 1 <= paths_per_video <= MAX_TEMPORAL_PATHS_PER_VIDEO:
            raise ValueError(
                f"paths_per_video must be between 1 and {MAX_TEMPORAL_PATHS_PER_VIDEO}"
            )
        if not 1 <= path_diversity_min_events <= len(events):
            raise ValueError(
                "path_diversity_min_events must be between 1 and the event count"
            )

        effective_reservoir_size = (
            top_k_sequences
            if sequence_reservoir_size is None
            else int(sequence_reservoir_size)
        )
        if effective_reservoir_size < top_k_sequences:
            raise ValueError("sequence_reservoir_size cannot be smaller than top_k_sequences")
        if effective_reservoir_size > MAX_TEMPORAL_SEQUENCE_RESERVOIR:
            raise ValueError(
                "sequence_reservoir_size cannot exceed "
                f"{MAX_TEMPORAL_SEQUENCE_RESERVOIR}"
            )

        effective_beam_width = (
            max(64, paths_per_video * 32)
            if path_beam_width is None
            else int(path_beam_width)
        )
        if not 1 <= effective_beam_width <= MAX_TEMPORAL_PATH_BEAM_WIDTH:
            raise ValueError(
                f"path_beam_width must be between 1 and {MAX_TEMPORAL_PATH_BEAM_WIDTH}"
            )
        if effective_beam_width < paths_per_video:
            raise ValueError("path_beam_width cannot be smaller than paths_per_video")
        bounded_path_search = (
            paths_per_video > 1
            or top_k_per_event > 300
            or path_beam_width is not None
        )

        ordered_events: list[dict[str, Any]] = []
        seen_orders: set[int] = set()
        for raw_event in events:
            order = int(raw_event.get("order", 0))
            description = str(raw_event.get("description", "")).strip()
            visual_query = str(raw_event.get("global_scene_en", "")).strip()
            if order < 1 or order in seen_orders:
                raise ValueError("Temporal event orders must be unique positive integers")
            if not description or not visual_query:
                raise ValueError("Every temporal event needs a description and global_scene_en")
            seen_orders.add(order)
            ordered_events.append(
                {
                    "order": order,
                    "description": description,
                    "global_scene_en": visual_query,
                }
            )
        ordered_events.sort(key=lambda event: event["order"])

        effective_anchor_order = (
            int(anchor_event_order)
            if anchor_event_order is not None
            else int(ordered_events[0]["order"])
        )
        if effective_anchor_order not in seen_orders:
            raise ValueError("anchor_event_order must identify one of the supplied events")

        started = time.perf_counter()
        grouped_event_pools: list[dict[str, list[dict[str, Any]]]] = []
        event_pool_audit: list[dict[str, Any]] = []
        common_video_ids: set[str] | None = None

        for event in ordered_events:
            event_started = time.perf_counter()
            query = event["global_scene_en"]
            diagnostics_fn = getattr(self.registry, "siglip_text_diagnostics", None)
            query_diagnostics = diagnostics_fn(query) if diagnostics_fn else None
            query_vector = self.registry.embed_siglip_text(query)
            raw_results = self.searcher.search_visual(query_vector, top_k=top_k_per_event)

            results_by_video: dict[str, list[dict[str, Any]]] = {}
            for raw_result in raw_results:
                result = dict(raw_result)
                results_by_video.setdefault(str(result["video_id"]), []).append(result)
            for video_results in results_by_video.values():
                video_results.sort(
                    key=lambda result: (
                        float(result.get("pts_time_s", 0.0)),
                        int(result.get("keyframe_n", 0)),
                        int(result.get("frame_idx", 0)),
                        int(result.get("global_idx", 0)),
                    )
                )

            grouped_event_pools.append(results_by_video)
            event_video_ids = set(results_by_video)
            common_video_ids = (
                event_video_ids
                if common_video_ids is None
                else common_video_ids.intersection(event_video_ids)
            )

            candidate_videos: list[dict[str, Any]] = []
            for video_id, video_results in results_by_video.items():
                best_result = min(video_results, key=lambda result: int(result["rank"]))
                candidate_videos.append(
                    {
                        "video_id": video_id,
                        "hit_count": len(video_results),
                        "best_global_rank": int(best_result["rank"]),
                        "best_raw_score": float(best_result["score"]),
                    }
                )
            candidate_videos.sort(
                key=lambda video: (
                    -int(video["hit_count"]),
                    int(video["best_global_rank"]),
                    str(video["video_id"]),
                )
            )

            event_pool = {
                "order": int(event["order"]),
                "description": event["description"],
                "query": query,
                "event_order": int(event["order"]),
                "event_description": event["description"],
                "event_query": query,
                "query_source": "events[].global_scene_en",
                "modality": "siglip",
                "score_type": "cosine",
                "score_description": "Raw cosine between event text and full-frame image",
                "top_k_requested": int(top_k_per_event),
                "result_count": len(raw_results),
                "candidate_video_count": len(results_by_video),
                "candidate_videos": candidate_videos,
                "execution_time_ms": round(
                    (time.perf_counter() - event_started) * 1000.0,
                    2,
                ),
            }
            if query_diagnostics:
                event_pool["query_diagnostics"] = query_diagnostics
            event_pool_audit.append(event_pool)

        common_video_ids = common_video_ids or set()
        anchor_query_clean = str(anchor_query or "").strip()
        anchor_results_by_video: dict[str, dict[str, Any]] = {}
        anchor_pool_audit: dict[str, Any] | None = None
        if anchor_query_clean:
            anchor_started = time.perf_counter()
            diagnostics_fn = getattr(self.registry, "siglip_text_diagnostics", None)
            anchor_diagnostics = (
                diagnostics_fn(anchor_query_clean) if diagnostics_fn else None
            )
            anchor_results = self.searcher.search_visual(
                self.registry.embed_siglip_text(anchor_query_clean),
                top_k=top_k_per_event,
            )
            for result in anchor_results:
                video_id = str(result["video_id"])
                current = anchor_results_by_video.get(video_id)
                if current is None or int(result["rank"]) < int(current["rank"]):
                    anchor_results_by_video[video_id] = dict(result)
            common_video_ids.intersection_update(anchor_results_by_video)
            anchor_pool_audit = {
                "query": anchor_query_clean,
                "query_source": "anchor_query",
                "modality": "siglip",
                "score_type": "cosine",
                "score_description": "Raw cosine between shared-scene anchor text and full-frame image",
                "top_k_requested": int(top_k_per_event),
                "result_count": len(anchor_results),
                "candidate_video_count": len(anchor_results_by_video),
                "execution_time_ms": round(
                    (time.perf_counter() - anchor_started) * 1000.0,
                    2,
                ),
            }
            if anchor_diagnostics:
                anchor_pool_audit["query_diagnostics"] = anchor_diagnostics

        sequences: list[dict[str, Any]] = []

        def state_sort_key(state: dict[str, Any]) -> tuple[Any, ...]:
            path = state["path"]
            span = float(path[-1]["pts_time_s"]) - float(path[0]["pts_time_s"])
            return (
                -float(state["minimum_score"]),
                -float(state["score_sum"]),
                int(state["global_rank_sum"]),
                span,
                tuple(int(frame.get("global_idx", 0)) for frame in path),
            )

        def path_signature(state: dict[str, Any]) -> tuple[int, ...]:
            return tuple(int(frame["frame_idx"]) for frame in state["path"])

        def can_follow(previous: dict[str, Any], candidate: dict[str, Any]) -> bool:
            gap_seconds = float(candidate["pts_time_s"]) - float(previous["pts_time_s"])
            if gap_seconds <= 0:
                return False
            if int(candidate["keyframe_n"]) <= int(previous["keyframe_n"]):
                return False
            if int(candidate["frame_idx"]) <= int(previous["frame_idx"]):
                return False
            return max_gap_seconds is None or gap_seconds <= max_gap_seconds

        def extend_state(
            previous_state: dict[str, Any], candidate: dict[str, Any]
        ) -> dict[str, Any]:
            return {
                "minimum_score": min(
                    float(previous_state["minimum_score"]),
                    float(candidate["score"]),
                ),
                "score_sum": float(previous_state["score_sum"])
                + float(candidate["score"]),
                "global_rank_sum": int(previous_state["global_rank_sum"])
                + int(candidate["rank"]),
                "path": [*previous_state["path"], candidate],
            }

        def single_best_path(
            candidates_by_event: list[list[dict[str, Any]]],
        ) -> list[dict[str, Any]]:
            """Retain the exact one-path dynamic-programming policy used previously."""
            states: list[dict[str, Any]] = [
                {
                    "minimum_score": float(candidate["score"]),
                    "score_sum": float(candidate["score"]),
                    "global_rank_sum": int(candidate["rank"]),
                    "path": [candidate],
                }
                for candidate in candidates_by_event[0]
            ]
            for current_candidates in candidates_by_event[1:]:
                next_states: list[dict[str, Any]] = []
                for candidate in current_candidates:
                    candidate_states_by_minimum: dict[float, dict[str, Any]] = {}
                    for previous_state in states:
                        if not can_follow(previous_state["path"][-1], candidate):
                            continue
                        candidate_state = extend_state(previous_state, candidate)
                        minimum_score = float(candidate_state["minimum_score"])
                        existing = candidate_states_by_minimum.get(minimum_score)
                        if existing is None or state_sort_key(candidate_state) < state_sort_key(
                            existing
                        ):
                            candidate_states_by_minimum[minimum_score] = candidate_state
                    next_states.extend(candidate_states_by_minimum.values())
                states = next_states
                if not states:
                    return []
            return [min(states, key=state_sort_key)] if states else []

        def diverse_bounded_paths(
            candidates_by_event: list[list[dict[str, Any]]],
        ) -> list[dict[str, Any]]:
            """Return several deterministic paths while bounding every DP layer."""
            states: list[dict[str, Any]] = [
                {
                    "minimum_score": float(candidate["score"]),
                    "score_sum": float(candidate["score"]),
                    "global_rank_sum": int(candidate["rank"]),
                    "path": [candidate],
                }
                for candidate in candidates_by_event[0]
            ]
            per_endpoint_limit = min(
                effective_beam_width,
                max(8, paths_per_video * 4),
            )
            for current_candidates in candidates_by_event[1:]:
                next_states: list[dict[str, Any]] = []
                for candidate in current_candidates:
                    ending_states: list[dict[str, Any]] = []
                    seen_signatures: set[tuple[int, ...]] = set()
                    for previous_state in states:
                        if not can_follow(previous_state["path"][-1], candidate):
                            continue
                        candidate_state = extend_state(previous_state, candidate)
                        signature = path_signature(candidate_state)
                        if signature in seen_signatures:
                            continue
                        seen_signatures.add(signature)
                        ending_states.append(candidate_state)
                    ending_states.sort(key=state_sort_key)
                    next_states.extend(ending_states[:per_endpoint_limit])

                if not next_states:
                    return []
                next_states.sort(key=state_sort_key)
                states = next_states[:effective_beam_width]

            selected: list[dict[str, Any]] = []
            selected_signatures: list[tuple[int, ...]] = []
            for state in sorted(states, key=state_sort_key):
                signature = path_signature(state)
                if signature in selected_signatures:
                    continue
                if any(
                    sum(
                        left != right
                        for left, right in zip(signature, selected_signature, strict=True)
                    )
                    < path_diversity_min_events
                    for selected_signature in selected_signatures
                ):
                    continue
                selected.append(state)
                selected_signatures.append(signature)
                if len(selected) >= paths_per_video:
                    break
            return selected

        def sequence_from_state(
            *,
            video_id: str,
            best_state: dict[str, Any],
            video_path_rank: int,
        ) -> dict[str, Any]:
            path = best_state["path"]
            matched_events: list[dict[str, Any]] = []
            for event, candidate in zip(ordered_events, path, strict=True):
                matched = dict(candidate)
                matched.update(
                    {
                        "event_order": int(event["order"]),
                        "event_description": event["description"],
                        "event_query": event["global_scene_en"],
                    }
                )
                matched_events.append(matched)

            timestamps = [float(frame["pts_time_s"]) for frame in matched_events]
            gaps_seconds = [
                round(current - previous, 6)
                for previous, current in zip(timestamps, timestamps[1:], strict=False)
            ]
            span_seconds = round(timestamps[-1] - timestamps[0], 6)
            mean_event_score = round(
                float(best_state["score_sum"]) / len(ordered_events),
                6,
            )
            minimum_event_score = round(float(best_state["minimum_score"]), 6)
            anchor_frame = next(
                frame
                for frame in matched_events
                if int(frame["event_order"]) == effective_anchor_order
            )
            context_anchor_frame = anchor_results_by_video.get(video_id)
            sequence_score = (
                round(
                    (float(context_anchor_frame["score"]) + minimum_event_score) / 2.0,
                    6,
                )
                if context_anchor_frame
                else minimum_event_score
            )
            return {
                "video_id": video_id,
                "video_path_rank": video_path_rank,
                "context_anchor_rank": (
                    int(context_anchor_frame["rank"]) if context_anchor_frame else None
                ),
                "context_anchor_score": (
                    float(context_anchor_frame["score"]) if context_anchor_frame else None
                ),
                "context_anchor_frame": (
                    dict(context_anchor_frame) if context_anchor_frame else None
                ),
                "minimum_event_score": minimum_event_score,
                "mean_event_score": mean_event_score,
                "sequence_score": sequence_score,
                "score_type": (
                    "mean_context_anchor_and_minimum_event_raw_siglip_cosine"
                    if context_anchor_frame
                    else "minimum_then_mean_raw_siglip_cosine"
                ),
                "ranking_values": {
                    "sequence_score": sequence_score,
                    "context_anchor_score": (
                        float(context_anchor_frame["score"])
                        if context_anchor_frame
                        else None
                    ),
                    "minimum_event_score": minimum_event_score,
                    "mean_event_score": mean_event_score,
                },
                "global_rank_sum": int(best_state["global_rank_sum"]),
                "span_seconds": span_seconds,
                "gaps_seconds": gaps_seconds,
                "matched_frames": [int(frame["frame_idx"]) for frame in matched_events],
                "timestamps": timestamps,
                "matched_events": matched_events,
                "anchor_event_order": effective_anchor_order,
                "anchor_rank": int(anchor_frame["rank"]),
                "anchor_score": float(anchor_frame["score"]),
                "anchor_frame": dict(anchor_frame),
                "submission_string": anchor_frame["submission_string"],
            }

        for video_id in sorted(common_video_ids):
            candidates_by_event = [pool[video_id] for pool in grouped_event_pools]
            path_states = (
                diverse_bounded_paths(candidates_by_event)
                if bounded_path_search
                else single_best_path(candidates_by_event)
            )
            for video_path_rank, state in enumerate(path_states, 1):
                sequences.append(
                    sequence_from_state(
                        video_id=video_id,
                        best_state=state,
                        video_path_rank=video_path_rank,
                    )
                )

        sequences.sort(
            key=lambda sequence: (
                -float(sequence["sequence_score"]),
                -float(sequence["minimum_event_score"]),
                -float(sequence["mean_event_score"]),
                int(sequence["global_rank_sum"]),
                float(sequence["span_seconds"]),
                str(sequence["video_id"]),
                tuple(int(frame_idx) for frame_idx in sequence["matched_frames"]),
            )
        )
        monotonic_video_count = len({str(sequence["video_id"]) for sequence in sequences})
        computed_sequence_count = len(sequences)
        sequence_reservoir = sequences[:effective_reservoir_size]
        for reservoir_rank, sequence in enumerate(sequence_reservoir, 1):
            sequence["reservoir_rank"] = reservoir_rank
        returned_sequences = sequence_reservoir[:top_k_sequences]
        reserve_sequences = sequence_reservoir[top_k_sequences:]
        for rank, sequence in enumerate(returned_sequences, 1):
            sequence["rank"] = rank

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "task_type": "KIS",
            "operation": "ordered_siglip_intersection",
            "experiment_mode": "nofusion_with_explicit_temporal_intersection",
            "modality": "siglip",
            "events": ordered_events,
            "event_count": len(ordered_events),
            "top_k_per_event": int(top_k_per_event),
            "top_k_sequences": int(top_k_sequences),
            "event_candidate_reservoir_size": int(top_k_per_event),
            "sequence_reservoir_size": effective_reservoir_size,
            "sequence_reservoir_count": len(sequence_reservoir),
            "computed_sequence_count": computed_sequence_count,
            "paths_per_video": paths_per_video,
            "path_beam_width": effective_beam_width,
            "path_beam_applied": bounded_path_search,
            "path_search_mode": (
                "bounded_diverse_beam"
                if bounded_path_search
                else "legacy_exact_single_path"
            ),
            "path_diversity_min_events": path_diversity_min_events,
            "max_gap_seconds": max_gap_seconds,
            "anchor_event_order": effective_anchor_order,
            "anchor_query": anchor_query_clean,
            "anchor_query_applied": bool(anchor_query_clean),
            "anchor_pool": anchor_pool_audit,
            "event_pools": event_pool_audit,
            "intersection_video_count": len(common_video_ids),
            "monotonic_video_count": monotonic_video_count,
            "ordered_sequence_count": computed_sequence_count,
            "result_count": len(returned_sequences),
            "score_type": (
                "mean_context_anchor_and_minimum_event_raw_siglip_cosine"
                if anchor_query_clean
                else "minimum_then_mean_raw_siglip_cosine"
            ),
            "same_modality_event_aggregation": (
                "mean_context_anchor_and_minimum_event_then_event_mean"
                if anchor_query_clean
                else "bottleneck_minimum_then_arithmetic_mean"
            ),
            "score_policy": (
                "When a shared-scene anchor is supplied, rank valid same-video monotonic "
                "paths by the arithmetic mean of anchor cosine and minimum event cosine; "
                "then event mean, lower sum of global event ranks, shorter span, and "
                "video_id. max_gap_seconds is a filter and never changes a score."
            ),
            "same_modality_event_aggregation_applied": True,
            "cross_modal_fusion_applied": False,
            "fusion_applied": False,
            "reranking_applied": False,
            "temporal_constraint_applied": True,
            "sequences": returned_sequences,
            "reserve_sequences": reserve_sequences,
            "execution_time_ms": round(elapsed_ms, 2),
        }

    def discover_dam_to_siglip(
        self,
        parsed_query: ParsedQuery,
        *,
        dam_top_frames_per_object: int = 20,
        siglip_top_frames_per_video: int = 10,
    ) -> dict[str, Any]:
        """Expose a DAM-gated, SigLIP-ranked discovery cascade without score fusion."""
        if dam_top_frames_per_object < 1 or siglip_top_frames_per_video < 1:
            raise ValueError("Cascade limits must be positive")

        visual_query = parsed_query.global_scene_en.strip()
        object_queries = [query.strip() for query in parsed_query.objects_en if query.strip()]
        if not visual_query:
            raise ValueError("global_scene_en is required for discovery")
        if not object_queries:
            raise ValueError("At least one objects_en query is required for discovery")

        started = time.perf_counter()
        diagnostics_fn = getattr(self.registry, "siglip_text_diagnostics", None)
        visual_diagnostics = diagnostics_fn(visual_query) if diagnostics_fn else None
        object_vectors = self.registry.embed_bge_text(object_queries)
        visual_vector = self.registry.embed_siglip_text(visual_query)

        scoped_cache: dict[str, list[dict[str, Any]]] = {}
        all_candidate_video_ids: set[str] = set()
        cascades: list[dict[str, Any]] = []

        for object_offset, object_query in enumerate(object_queries):
            dam_results = self.searcher.search_dam(
                [object_vectors[object_offset]],
                [object_query],
                top_k=dam_top_frames_per_object,
                match_threshold=self.dam_match_threshold,
            )

            candidate_videos: list[dict[str, Any]] = []
            seen_videos: set[str] = set()
            for dam_result in dam_results:
                video_id = dam_result["video_id"]
                if video_id in seen_videos:
                    continue
                seen_videos.add(video_id)
                all_candidate_video_ids.add(video_id)
                candidate_videos.append(
                    {
                        "candidate_video_order": len(candidate_videos) + 1,
                        "video_id": video_id,
                        "dam_raw_frame_rank": int(dam_result["rank"]),
                        "dam_frame_idx": int(dam_result["frame_idx"]),
                        "dam_keyframe_n": int(dam_result["keyframe_n"]),
                        "dam_score": float(dam_result["score"]),
                        "dam_score_type": dam_result["score_type"],
                        "evaluated_frames": self.searcher.get_video_frame_count(video_id),
                    }
                )

            scoped_results: list[dict[str, Any]] = []
            for candidate in candidate_videos:
                video_id = candidate["video_id"]
                if video_id not in scoped_cache:
                    scoped_cache[video_id] = self.searcher.search_visual_in_video(
                        visual_vector,
                        video_id,
                        top_k=siglip_top_frames_per_video,
                    )
                for base_result in scoped_cache[video_id]:
                    result = dict(base_result)
                    result.update(
                        {
                            "video_scope_rank": int(base_result["rank"]),
                            "discovery_object_index": object_offset + 1,
                            "discovery_object_query": object_query,
                            "candidate_video_order": candidate["candidate_video_order"],
                            "dam_discovery_rank": candidate["dam_raw_frame_rank"],
                            "dam_discovery_frame_idx": candidate["dam_frame_idx"],
                            "dam_discovery_keyframe_n": candidate["dam_keyframe_n"],
                            "dam_discovery_score": candidate["dam_score"],
                            "dam_discovery_score_type": candidate["dam_score_type"],
                            "scope": "dam_to_siglip_cascade",
                        }
                    )
                    scoped_results.append(result)

            scoped_results.sort(
                key=lambda result: (-float(result["score"]), int(result["global_idx"]))
            )
            for cascade_rank, result in enumerate(scoped_results, 1):
                result["rank"] = cascade_rank

            cascades.append(
                {
                    "cascade_id": f"dam_object_{object_offset + 1}",
                    "display_name": f"DAM object {object_offset + 1} → scoped SigLIP",
                    "object_query": object_query,
                    "object_query_source": f"objects_en[{object_offset}]",
                    "dam_score_type": "best_region_cosine",
                    "dam_frames_considered": len(dam_results),
                    "candidate_video_count": len(candidate_videos),
                    "candidate_videos": candidate_videos,
                    "siglip_query": visual_query,
                    "siglip_query_source": "global_scene_en",
                    "siglip_score_type": "cosine",
                    "siglip_frames_per_video": siglip_top_frames_per_video,
                    "evaluated_video_frames": sum(
                        candidate["evaluated_frames"] for candidate in candidate_videos
                    ),
                    "result_count": len(scoped_results),
                    "results": scoped_results,
                    "final_ranking": (
                        "Raw SigLIP cosine over the union of each candidate video's "
                        f"top {siglip_top_frames_per_video} scoped frames"
                    ),
                    "dam_score_used_in_final_rank": False,
                    "fusion_applied": False,
                    "cross_modal_gating_applied": True,
                    "learned_reranker_applied": False,
                }
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "operation": "dam_to_siglip_discovery_cascade",
            "experiment_mode": "nofusion_with_explicit_cascade",
            "cascade_applied": True,
            "cross_modal_gating_applied": True,
            "fusion_applied": False,
            "dam_score_used_in_final_rank": False,
            "learned_reranker_applied": False,
            "dam_top_frames_per_object": dam_top_frames_per_object,
            "siglip_top_frames_per_video": siglip_top_frames_per_video,
            "object_query_count": len(object_queries),
            "unique_candidate_video_count": len(all_candidate_video_ids),
            "unique_evaluated_frames": sum(
                self.searcher.get_video_frame_count(video_id)
                for video_id in all_candidate_video_ids
            ),
            "result_count": sum(cascade["result_count"] for cascade in cascades),
            "siglip_query_diagnostics": visual_diagnostics,
            "cascades": cascades,
            "execution_time_ms": round(elapsed_ms, 2),
        }

    def search_visual_in_video(
        self,
        parsed_query: ParsedQuery,
        *,
        video_id: str,
        top_k: int,
    ) -> dict[str, Any]:
        """Run an explicit SigLIP-only drill-down over one video's frame rows."""
        canonical_id = video_id.upper().replace("-", "_")
        visual_query = parsed_query.global_scene_en.strip()
        frame_count = self.searcher.get_video_frame_count(canonical_id)
        score_description = (
            f"Raw full-frame image/text cosine restricted to {canonical_id}; "
            "the source card's score is not reused and no modality scores are combined"
        )
        if not visual_query:
            pool = self._not_run_pool(
                modality="siglip",
                display_name=f"SigLIP inside {canonical_id}",
                query="",
                query_source="global_scene_en",
                score_type="cosine",
                score_description=score_description,
                reason="The parsed query has no global_scene_en value.",
            )
        else:
            diagnostics_fn = getattr(self.registry, "siglip_text_diagnostics", None)
            visual_diagnostics = diagnostics_fn(visual_query) if diagnostics_fn else None

            def run_visual() -> list[dict[str, Any]]:
                query_vector = self.registry.embed_siglip_text(visual_query)
                return self.searcher.search_visual_in_video(
                    query_vector,
                    canonical_id,
                    top_k=top_k,
                )

            pool = self._run_pool(
                modality="siglip",
                display_name=f"SigLIP inside {canonical_id}",
                query=visual_query,
                query_source="global_scene_en",
                score_type="cosine",
                score_description=score_description,
                search=run_visual,
                diagnostics=visual_diagnostics,
            )

        pool.update(
            {
                "scope": "video",
                "video_id": canonical_id,
                "evaluated_frames": frame_count,
                "fusion_applied": False,
                "reranking_applied": False,
            }
        )
        return pool
