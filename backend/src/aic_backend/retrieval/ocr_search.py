"""Dedicated OCR-only retrieval without query parsing or modality fusion."""

from __future__ import annotations

from .models import SearchHit
from .ports import RetrievalRepository


class OcrSearchService:
    def __init__(self, repository: RetrievalRepository) -> None:
        self.repository = repository

    def retrieve(self, query: str, *, top_k: int, fuzzy: bool) -> list[SearchHit]:
        candidates = self.repository.search_text(
            "ocr",
            query,
            limit=top_k,
            fuzzy=fuzzy,
        )
        return [
            SearchHit(
                video_id=row.video_id,
                frame_idx=row.frame_idx,
                pts_time_s=row.pts_time_s,
                keyframe_n=row.keyframe_n,
                score=row.score,
                modality_scores={"ocr": row.score},
                evidence=(row.evidence,) if row.evidence else (),
                ocr=row.ocr,
                ocr_match=row.ocr_match,
            )
            for row in candidates
        ]
