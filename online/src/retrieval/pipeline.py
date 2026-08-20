"""High-Level VideoRetrievalEngine API orchestrating Query Parsing, Stage 1, Stage 2, and VQA."""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Optional
from qdrant_client import QdrantClient

from online.src.contracts.query import (
    ParsedQuery,
    SearchResponse,
    SearchResult,
    TaskType,
)
from online.src.retrieval.embeddings import ModelRegistry
from online.src.retrieval.query_parser import QueryParser
from online.src.retrieval.stage1_funnel import Stage1Funnel
from online.src.retrieval.stage2_reranker import Stage2Reranker
from online.src.retrieval.vqa_reasoner import VQAReasoner

logger = logging.getLogger(__name__)


class VideoRetrievalEngine:
    """Production Video Retrieval Engine uniting all stages into a unified search API."""

    def __init__(
        self,
        qdrant_db_path: str = "/Users/khoale/Downloads/AIC_HCM/qdrant_db",
        keyframes_root: str = "/Users/khoale/Downloads/AIC_Challenger/data/keyframes",
        models: Optional[ModelRegistry] = None,
    ) -> None:
        self.qdrant_db_path = Path(qdrant_db_path)
        self.keyframes_root = Path(keyframes_root)
        self.client = QdrantClient(path=str(self.qdrant_db_path))
        self.models = models or ModelRegistry()

        # Initialize Sub-Components
        from online.src.index.qdrant_indexer import QdrantIndexer
        indexer = QdrantIndexer(client=self.client, models=self.models)
        indexer.init_collections(force_recreate=False)

        self.parser = QueryParser()
        self.stage1 = Stage1Funnel(client=self.client, models=self.models)
        self.stage2 = Stage2Reranker(models=self.models, keyframes_root=self.keyframes_root)
        self.vqa_reasoner = VQAReasoner()

        logger.info("VideoRetrievalEngine successfully initialized and ready for search!")

    def parse_query(self, query_text: str, task_type: TaskType = "KIS") -> ParsedQuery:
        """Parse raw query into structured sub-queries (for UI inspection & human adjustment)."""
        return self.parser.parse(query_text, task_type=task_type)

    def search(
        self,
        query: str | ParsedQuery,
        task_type: TaskType = "KIS",
        top_k: int = 50,
    ) -> SearchResponse:
        """Execute complete end-to-end multimodal search."""
        start_time = time.perf_counter()

        # 1. Obtain structured ParsedQuery
        if isinstance(query, str):
            original_text = query
            parsed = self.parser.parse(query, task_type=task_type)
        else:
            parsed = query
            original_text = parsed.original_query

        # 2. Dispatch based on Task Type
        if parsed.task_type == "TRAKE" and parsed.trake_events and len(parsed.trake_events) >= 2:
            # TRAKE Multi-Event Search
            event_candidates = []
            for ev in parsed.trake_events:
                sub_parsed = ParsedQuery(
                    task_type="KIS",
                    original_query=ev.description,
                    global_scene_en=ev.scene_en,
                    objects_en=ev.objects_en,
                    speech_vi=ev.speech_vi,
                    ocr_keywords=ev.ocr_keywords,
                    weights=parsed.weights,
                )
                cands = self.stage1.search_candidates(sub_parsed, top_k=top_k * 2)
                event_candidates.append(cands)

            results = self.stage2.verify_trake_sequence(
                event_candidates,
                max_time_span_s=90.0,
                final_top_k=top_k,
            )
        else:
            # Standard KIS or VQA Search
            candidates = self.stage1.search_candidates(parsed, top_k=top_k)
            results = self.stage2.rerank_kis(parsed, candidates, final_top_k=top_k)

        # 3. Handle VQA Question Answering for Top Evidence Frame
        if parsed.task_type == "VQA" and results:
            top_result = results[0]
            question = parsed.vqa_question or original_text
            extracted_answer = self.vqa_reasoner.answer_question(question, top_result)
            top_result.vqa_answer = extracted_answer

        exec_time_ms = (time.perf_counter() - start_time) * 1000.0

        return SearchResponse(
            task_type=parsed.task_type,
            original_query=original_text,
            parsed_query=parsed,
            execution_time_ms=exec_time_ms,
            total_candidates_evaluated=177321,
            results=results,
        )
