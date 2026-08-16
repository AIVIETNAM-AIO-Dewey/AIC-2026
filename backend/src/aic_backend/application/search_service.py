"""KIS retrieval orchestration with independent modality searches."""

from __future__ import annotations

from aic2026.contracts.query import QuerySpec

from ..domain.assignment import assign_objects
from ..domain.fusion import normalized_weights, temporal_nms, weighted_rrf
from ..domain.models import FrameCandidate, SearchHit
from ..domain.ports import RetrievalRepository


class SearchService:
    def __init__(self, repository: RetrievalRepository) -> None:
        self.repository = repository

    def retrieve(
        self,
        query: QuerySpec,
        *,
        top_k: int = 100,
        video_id: str | None = None,
        dense: bool = False,
    ) -> list[SearchHit]:
        ranked: dict[str, list[FrameCandidate]] = {}
        if query.scene_en:
            scene = self.repository.search_scene(
                query.scene_en, limit=max(100, top_k), video_id=video_id
            )
            if scene:
                ranked["dense" if dense else "scene"] = list(scene)
        if query.objects_en:
            object_rows: list[FrameCandidate] = []
            for slot, item in enumerate(query.objects_en):
                object_rows.extend(
                    self.repository.search_text(
                        "object", item, limit=80, video_id=video_id, object_slot=slot
                    )
                )
            matched = assign_objects(object_rows, len(query.objects_en))
            sample = {row.frame_uid: row for row in object_rows}
            ranked["object"] = [
                FrameCandidate(**{**row.__dict__, "score": score})
                for uid, score in matched.items()
                if (row := sample[uid])
            ]
        if query.ocr_vi:
            ranked["ocr"] = [
                row
                for term in query.ocr_vi
                for row in self.repository.search_text("ocr", term, limit=80, video_id=video_id)
            ]
        if query.audio_vi:
            ranked["asr"] = [
                row
                for term in query.audio_vi
                for row in self.repository.search_text("asr", term, limit=80, video_id=video_id)
            ]
        if query.audio_events_en:
            ranked.setdefault("asr", []).extend(
                row
                for term in query.audio_events_en
                for row in self.repository.search_text("asr", term, limit=80, video_id=video_id)
            )
        weights = normalized_weights(ranked)
        # Dense TRAKE collections carry visual event candidates, but preserve the scene weight.
        if "dense" in ranked:
            weights["dense"] = weights.pop("scene", 1.0)
        return temporal_nms(weighted_rrf(ranked, weights=weights), per_video=20)[:top_k]
