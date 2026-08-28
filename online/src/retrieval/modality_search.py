"""Orchestrate four independent modality searches without cross-modal fusion."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from online.src.contracts.query import ParsedQuery

if TYPE_CHECKING:
    from online.src.retrieval.embeddings import ModelRegistry
    from online.src.retrieval.vector_search import FastVectorSearchEngine


class IndependentModalitySearch:
    """Run every applicable modality and retain four isolated result pools."""

    def __init__(
        self,
        *,
        searcher: FastVectorSearchEngine,
        registry: ModelRegistry,
    ) -> None:
        self.searcher = searcher
        self.registry = registry

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
    ) -> dict[str, Any]:
        started = time.perf_counter()
        results = search()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
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
                )

            pools["dam"] = self._run_pool(
                modality="dam",
                display_name="DAM detected objects",
                query=object_queries,
                query_source="objects_en",
                score_type="mean_best_region_cosine",
                score_description=(
                    "Mean of the best region cosine for each object query; "
                    "no coverage or synergy bonus"
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
                    "no coverage or synergy bonus"
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
        if not speech_query:
            speech_query = parsed_query.original_query.strip()
            speech_source = "original_query_fallback"
        if speech_query:
            pools["asr"] = self._run_pool(
                modality="asr",
                display_name="ASR spoken speech",
                query=speech_query,
                query_source=speech_source,
                score_type="cosine",
                score_description="Raw cosine between query text and frame-aligned ASR text",
                search=lambda: self.searcher.search_speech(
                    self.registry.embed_bge_text(speech_query), top_k=top_k
                ),
            )
        else:
            pools["asr"] = self._not_run_pool(
                modality="asr",
                display_name="ASR spoken speech",
                query="",
                query_source=speech_source,
                score_type="cosine",
                score_description="Raw cosine between query text and frame-aligned ASR text",
                reason="Neither speech_vi nor original_query contains searchable text.",
            )

        return pools
