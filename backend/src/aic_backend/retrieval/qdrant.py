"""Qdrant-backed retrieval; imports the client lazily so unit tests stay light."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..ingest.sparse import sparse_vector
from .models import Evidence, FrameCandidate, SearchHit
from .ports import RetrievalRepository


class QdrantRepository(RetrievalRepository):
    def __init__(
        self,
        client: Any,
        *,
        artifact_root: Path,
        text_encoder: Any,
        scene_encoder: Any | None = None,
    ) -> None:
        self.client = client
        self.artifact_root = artifact_root
        self.text_encoder = text_encoder
        self.scene_encoder = scene_encoder

    @staticmethod
    def _candidate(point: Any, modality: str, object_slot: int | None = None) -> FrameCandidate:
        payload = point.payload or {}
        return FrameCandidate(
            video_id=str(payload["video_id"]),
            frame_idx=int(payload["frame_idx"]),
            pts_time_s=float(payload["pts_time_s"]),
            keyframe_n=payload.get("keyframe_n"),
            score=float(point.score),
            modality=modality,  # type: ignore[arg-type]
            object_slot=object_slot,
            region_id=payload.get("region_id"),
            evidence=Evidence(
                modality=modality,
                text=payload.get("text"),
                source_id=str(point.id),
                score=float(point.score),
            ),  # type: ignore[arg-type]
        )

    def _query(
        self,
        collection: str,
        vector: list[float],
        *,
        modality: str,
        limit: int,
        video_id: str | None,
        object_slot: int | None = None,
        lexical_text: str | None = None,
    ) -> Sequence[FrameCandidate]:
        try:
            from qdrant_client import models
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("qdrant-client is required for online retrieval") from error
        query_filter = None
        if video_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))
                ]
            )
        if lexical_text is None:
            response = self.client.query_points(
                collection_name=collection,
                query=vector,
                using="scene",
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        else:
            indexes, values = sparse_vector(lexical_text)
            response = self.client.query_points(
                collection_name=collection,
                prefetch=[
                    models.Prefetch(query=vector, using="dense", limit=limit * 2),
                    models.Prefetch(
                        query=models.SparseVector(indices=indexes, values=values),
                        using="lexical",
                        limit=limit * 2,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        points = response.points
        return [self._candidate(point, modality, object_slot) for point in points]

    def search_scene(
        self,
        query: str,
        *,
        limit: int,
        video_id: str | None = None,
        dense: bool = False,
    ) -> Sequence[FrameCandidate]:
        if self.scene_encoder is None:
            return []
        return self._query(
            "frames_dense_current" if dense else "frames_sparse_current",
            self.scene_encoder.encode_texts([query])[0].tolist(),
            modality="scene",
            limit=limit,
            video_id=video_id,
        )

    def search_text(
        self,
        modality: str,
        query: str,
        *,
        limit: int,
        video_id: str | None = None,
        object_slot: int | None = None,
    ) -> Sequence[FrameCandidate]:
        if self.text_encoder is None:
            return []
        dense = self.text_encoder.encode([query], query=True)[0].tolist()
        collection = {
            "object": "regions_current",
            "ocr": "ocr_current",
            "asr": "asr_current",
            "dense": "frames_dense_current",
        }[modality]
        return self._query(
            collection,
            dense,
            modality=modality,
            limit=limit,
            video_id=video_id,
            object_slot=object_slot,
            lexical_text=query,
        )

    def frame_image_path(self, video_id: str, frame_idx: int) -> str | None:
        candidate = self.artifact_root / "dense_frames" / video_id / f"{frame_idx}.jpg"
        return str(candidate) if candidate.is_file() else None

    def neighbors(self, video_id: str, frame_idx: int, *, radius_s: float) -> Sequence[SearchHit]:
        del radius_s
        return []

    def ready(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def status(self) -> dict[str, object]:
        qdrant_ready = self.ready()
        aliases: set[str] = set()
        if qdrant_ready:
            try:
                aliases = {item.alias_name for item in self.client.get_aliases().aliases}
            except Exception:
                qdrant_ready = False
        collections = {
            name: f"{name}_current" in aliases
            for name in ("frames_sparse", "frames_dense", "regions", "ocr", "asr")
        }
        return {
            "qdrant_ready": qdrant_ready,
            "collections": collections,
            "models": {
                "siglip2_text": self.scene_encoder is not None,
                "e5_text": self.text_encoder is not None,
            },
        }
