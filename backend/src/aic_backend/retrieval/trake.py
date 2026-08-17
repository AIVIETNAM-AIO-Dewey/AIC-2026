"""Coarse video selection followed by dense ordered event alignment."""

from __future__ import annotations

from aic2026.contracts.query import QuerySpec

from .models import TrakeSequence
from .search import SearchService
from .temporal import ordered_event_sequences


class TrakeService:
    def __init__(self, search: SearchService) -> None:
        self.search = search

    def retrieve(self, query: QuerySpec, *, top_k: int = 100) -> list[TrakeSequence]:
        coarse = self.search.retrieve(
            query.model_copy(update={"task_type": "kis", "events": None}), top_k=200
        )
        candidate_videos: list[str] = []
        for hit in coarse:
            if hit.video_id not in candidate_videos:
                candidate_videos.append(hit.video_id)
            if len(candidate_videos) == 10:
                break
        all_sequences: list[TrakeSequence] = []
        for video_id in candidate_videos:
            event_hits = []
            for event in query.events or []:
                event_query = query.model_copy(
                    update={
                        "task_type": "kis",
                        "scene_en": event.scene_en,
                        "objects_en": event.objects_en,
                        "ocr_vi": event.ocr_vi,
                        "audio_vi": event.audio_vi,
                        "audio_events_en": event.audio_events_en,
                        "events": None,
                    }
                )
                event_hits.append(
                    self.search.retrieve(event_query, top_k=80, video_id=video_id, dense=True)
                )
            all_sequences.extend(ordered_event_sequences(event_hits, limit=top_k))
        return sorted(all_sequences, key=lambda item: (-item.score, item.video_id))[:top_k]
