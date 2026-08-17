"""Interfaces injected into application services."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from aic2026.contracts.query import QuerySpec

from .models import FrameCandidate, SearchHit


class RetrievalRepository(Protocol):
    def search_scene(
        self,
        query: str,
        *,
        limit: int,
        video_id: str | None = None,
        dense: bool = False,
    ) -> Sequence[FrameCandidate]: ...

    def search_text(
        self,
        modality: str,
        query: str,
        *,
        limit: int,
        video_id: str | None = None,
        object_slot: int | None = None,
    ) -> Sequence[FrameCandidate]: ...

    def frame_image_path(self, video_id: str, frame_idx: int) -> str | None: ...

    def neighbors(
        self, video_id: str, frame_idx: int, *, radius_s: float
    ) -> Sequence[SearchHit]: ...

    def ready(self) -> bool: ...


class QueryParser(Protocol):
    def parse(self, *, task_type: str, raw_query_vi: str) -> QuerySpec: ...


class Answerer(Protocol):
    def answer(
        self, *, query: QuerySpec, frames: Sequence[SearchHit], use_images: bool
    ) -> tuple[str, float, list[str]]: ...
