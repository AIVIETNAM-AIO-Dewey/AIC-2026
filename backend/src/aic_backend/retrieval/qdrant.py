"""Qdrant-backed retrieval; imports the client lazily so unit tests stay light."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..ingest.sparse import sparse_vector
from .fuzzy import rerank_fuzzy_candidates
from .models import Evidence, FrameCandidate, OcrLine, SearchHit, StructuredOcr
from .ports import RetrievalRepository


def _structured_ocr(value: Any) -> StructuredOcr | None:
    if not isinstance(value, dict):
        return None
    lines = []
    for item in value.get("lines", []):
        polygon = item.get("polygon_xy")
        points = tuple((float(point[0]), float(point[1])) for point in polygon) if polygon else None
        lines.append(
            OcrLine(
                line_id=str(item["line_id"]),
                raw_text=str(item.get("raw_text", "")),
                normalized_text=str(item.get("normalized_text", "")),
                confidence=(
                    float(item["confidence"]) if item.get("confidence") is not None else None
                ),
                accepted=bool(item.get("accepted", True)),
                polygon_xy=points,
                polygon_clamped=bool(item.get("polygon_clamped", False)),
                reading_order=int(item.get("reading_order", 0)),
            )
        )
    return StructuredOcr(
        terminal_status=value["terminal_status"],
        full_text=str(value.get("full_text", "")),
        width=int(value["width"]),
        height=int(value["height"]),
        run_id=str(value["run_id"]),
        model_revisions=tuple(str(item) for item in value.get("model_revisions", [])),
        source_image_sha256=value.get("source_image_sha256"),
        lines=tuple(lines),
    )


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

    def _has_alias(self, alias: str) -> bool:
        try:
            return alias in {item.alias_name for item in self.client.get_aliases().aliases}
        except Exception:
            return False

    @staticmethod
    def _candidate(point: Any, modality: str, object_slot: int | None = None) -> FrameCandidate:
        payload = point.payload or {}
        structured = _structured_ocr(payload.get("ocr_frame"))
        return FrameCandidate(
            video_id=str(payload["video_id"]),
            frame_idx=int(payload["frame_idx"]),
            pts_time_s=float(payload["pts_time_s"]),
            keyframe_n=payload.get("keyframe_n"),
            score=float(point.score),
            modality=modality,  # type: ignore[arg-type]
            object_slot=object_slot,
            region_id=payload.get("region_id"),
            ocr=structured,
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
        vector: list[float] | None,
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
            if vector is None:
                return []
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
            lexical = models.SparseVector(indices=indexes, values=values)
            if vector is None:
                response = self.client.query_points(
                    collection_name=collection,
                    query=lexical,
                    using="lexical",
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
            else:
                response = self.client.query_points(
                    collection_name=collection,
                    prefetch=[
                        models.Prefetch(query=vector, using="dense", limit=limit * 2),
                        models.Prefetch(
                            query=lexical,
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
        candidates = [self._candidate(point, modality, object_slot) for point in points]
        if lexical_text is not None and modality == "ocr":
            return rerank_fuzzy_candidates(lexical_text, candidates, limit=limit)
        return candidates[:limit]

    def search_scene(
        self,
        query: str,
        *,
        limit: int,
        video_id: str | None = None,
        dense: bool = False,
    ) -> Sequence[FrameCandidate]:
        collection = "frames_dense_current" if dense else "frames_sparse_current"
        if self.scene_encoder is None or not self._has_alias(collection):
            return []
        return self._query(
            collection,
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
        collection = {
            "object": "regions_current",
            "ocr": "ocr_current",
            "asr": "asr_current",
            "dense": "frames_dense_current",
        }[modality]
        if not self._has_alias(collection):
            return []
        dense = (
            self.text_encoder.encode([query], query=True)[0].tolist()
            if self.text_encoder is not None
            else None
        )
        # Fetch a bounded pool for trigram/dense recall. Edit distance never scans
        # the full collection in application memory.
        candidate_limit = min(max(limit * 4, limit), 400) if modality == "ocr" else limit
        return self._query(
            collection,
            dense,
            modality=modality,
            limit=candidate_limit,
            video_id=video_id,
            object_slot=object_slot,
            lexical_text=query,
        )[:limit]

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
