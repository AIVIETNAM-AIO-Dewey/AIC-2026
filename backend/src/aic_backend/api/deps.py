"""Application composition root, replaceable in integration tests."""

from __future__ import annotations

from functools import lru_cache

from ..llm.gpt4o import GPT4oAdapter
from ..retrieval.qdrant import QdrantRepository
from ..retrieval.search import SearchService
from ..retrieval.trake import TrakeService
from ..settings import Settings


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_repository() -> QdrantRepository:
    settings = get_settings()
    try:
        from qdrant_client import QdrantClient
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("backend dependencies are not installed") from error
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5)
    encoder = None
    e5_snapshot = settings.artifact_root / "models" / "multilingual-e5-base"
    if e5_snapshot.is_dir():
        from ..retrieval.e5 import E5OnnxEncoder

        encoder = E5OnnxEncoder.from_pretrained(model_path=e5_snapshot, device=settings.device)
    scene_encoder = None
    siglip_snapshot = settings.artifact_root / "models" / "siglip2"
    if siglip_snapshot.is_dir():
        from aic2026.scene_embedding.siglip_backend import SiglipEncoder

        scene_encoder = SiglipEncoder.from_pretrained(
            model_id=str(siglip_snapshot),
            cache_dir=siglip_snapshot,
            device=settings.device,
        )
    return QdrantRepository(
        client,
        artifact_root=settings.artifact_root,
        text_encoder=encoder,
        scene_encoder=scene_encoder,
    )


@lru_cache
def get_gpt() -> GPT4oAdapter:
    settings = get_settings()
    return GPT4oAdapter(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        timeout_s=settings.openai_timeout_s,
    )


def get_search_service() -> SearchService:
    return SearchService(get_repository())


def get_trake_service() -> TrakeService:
    return TrakeService(get_search_service())
