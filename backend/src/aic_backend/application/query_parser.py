"""Safe degradation for KIS only when GPT-4o parsing is unavailable."""

from __future__ import annotations

from aic2026.contracts.query import QuerySpec

from ..infrastructure.openai.gpt4o import CapabilityUnavailable, GPT4oAdapter


class QueryParsingService:
    def __init__(self, adapter: GPT4oAdapter) -> None:
        self.adapter = adapter

    def parse(self, *, task_type: str, raw_query_vi: str) -> QuerySpec:
        try:
            return self.adapter.parse(task_type=task_type, raw_query_vi=raw_query_vi)
        except CapabilityUnavailable:
            if task_type != "kis":
                raise
            return QuerySpec(task_type="kis", raw_query_vi=raw_query_vi, scene_en=raw_query_vi)
