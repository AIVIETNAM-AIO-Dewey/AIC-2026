"""Legacy-compatible local Qdrant helpers.

OCR/ASR retrieval implementations live in their canonical modality modules;
this module retains the small visual/DAM workbench adapter used by existing
routes while delegating text lookup to injected canonical services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..infrastructure.qdrant import QdrantHttpClient, base_frame

if TYPE_CHECKING:
    from ..infrastructure.metadata import FrameMetadataStore


EXPECTED_FRAMES = 247_956


class CpuQdrantSearch:
    def __init__(
        self,
        qdrant: QdrantHttpClient,
        encoders: Any,
        ocr: Any,
        metadata: FrameMetadataStore | None = None,
        asr_service: Any | None = None,
        ocr_service: Any | None = None,
    ) -> None:
        self.qdrant = qdrant
        self.encoders = encoders
        self.ocr = ocr
        self.metadata = metadata
        self.asr_service = asr_service
        self.ocr_service = ocr_service

    def validate(self) -> None:
        frames = self.qdrant.collection("aic_frames")
        dam = self.qdrant.collection("aic_dam_regions")
        if int(frames["points_count"]) != EXPECTED_FRAMES:
            raise ValueError("aic_frames count mismatch")
        if int(dam["points_count"]) != 681_355:
            raise ValueError("aic_dam_regions count mismatch")

    def search_siglip(self, query: str, top_k: int) -> list[dict[str, Any]]:
        vector = self.encoders.embed_siglip_text(query)
        return self.search_visual(vector, top_k=top_k)

    def search_visual(
        self,
        vector: np.ndarray,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        points = self.qdrant.query("aic_frames", "siglip2", vector, top_k)
        return self._frame_results(points)

    def search_visual_in_video(
        self,
        vector: np.ndarray,
        video_id: str,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        canonical = video_id.upper().replace("-", "_")
        points = self.qdrant.query(
            "aic_frames",
            "siglip2",
            vector,
            top_k,
            {"must": [{"key": "video_id", "match": {"value": canonical}}]},
        )
        return self._frame_results(points)

    def _frame_results(self, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for rank, point in enumerate(points, 1):
            frame = base_frame(
                point["payload"],
                score=point["score"],
                rank=rank,
                score_type="cosine",
            )
            frame["global_idx"] = int(point["id"])
            results.append(frame)
        if results:
            lookup_many = getattr(self.ocr, "lookup_many", None)
            try:
                texts = (
                    lookup_many(str(frame.get("frame_uid") or "") for frame in results)
                    if callable(lookup_many)
                    else {}
                )
            except Exception:
                texts = {}
            if not isinstance(texts, dict):
                texts = {}
            for frame in results:
                frame["ocr_text"] = texts.get(str(frame.get("frame_uid") or ""), "")
        return results

    def search_dam_queries(self, queries: list[str], top_k: int) -> list[dict[str, Any]]:
        if not queries:
            return []
        vectors = self.encoders.embed_bge_text(queries)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        return self.search_dam(
            [vectors[index] for index in range(len(queries))],
            queries,
            top_k=top_k,
        )

    def search_dam(
        self,
        vectors: list[np.ndarray],
        queries: list[str],
        *,
        top_k: int,
        match_threshold: float = 0.50,
    ) -> list[dict[str, Any]]:
        if not queries:
            return []
        region_limit = max(1_000, min(4_000, top_k * 20))
        subject_hits: list[dict[int, dict[str, Any]]] = []
        for vector in vectors:
            hits = self.qdrant.query("aic_dam_regions", "dam", vector, region_limit)
            best: dict[int, dict[str, Any]] = {}
            for hit in hits:
                parent = int(hit["payload"]["parent_point_id"])
                if parent not in best or float(hit["score"]) > float(best[parent]["score"]):
                    best[parent] = hit
            subject_hits.append(best)
        candidate_ids = set().union(*(set(items) for items in subject_hits))
        scored: list[tuple[float, int]] = []
        for parent in candidate_ids:
            scores = [
                float(items[parent]["score"]) if parent in items else -1.0 for items in subject_hits
            ]
            scored.append((sum(scores) / len(scores), parent))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = scored[:top_k]
        frame_payloads = self.qdrant.retrieve("aic_frames", [parent for _, parent in selected])
        results: list[dict[str, Any]] = []
        for rank, (score, parent) in enumerate(selected, 1):
            payload = frame_payloads[parent]
            frame = base_frame(
                payload, score=score, rank=rank, score_type="mean_best_region_cosine"
            )
            subject_scores = []
            matched_boxes = []
            descriptions = []
            for subject, items in zip(queries, subject_hits, strict=True):
                hit = items.get(parent)
                if hit is None:
                    subject_scores.append({"subject": subject, "cosine": -1.0})
                    continue
                region = hit["payload"]
                cosine = round(float(hit["score"]), 6)
                description = str(region.get("description_en", ""))
                descriptions.append(description)
                subject_scores.append({"subject": subject, "cosine": cosine})
                matched_boxes.append(
                    {
                        "query_subject": subject,
                        "region_id": region.get("region_id"),
                        "class_entity": region.get("class_entity", "Object"),
                        "bbox": region.get("bbox", []),
                        "score": cosine,
                        "caption": description,
                    }
                )
            frame["subject_scores"] = subject_scores
            frame["matched_boxes"] = matched_boxes
            frame["best_matching_objects"] = matched_boxes
            frame["dam_summary"] = " ".join(dict.fromkeys(descriptions))
            frame["dam_supported"] = all(
                float(item["cosine"]) >= match_threshold for item in subject_scores
            )
            frame["global_idx"] = parent
            results.append(frame)
        if results:
            lookup_many = getattr(self.ocr, "lookup_many", None)
            try:
                texts = (
                    lookup_many(str(frame.get("frame_uid") or "") for frame in results)
                    if callable(lookup_many)
                    else {}
                )
            except Exception:
                texts = {}
            if not isinstance(texts, dict):
                texts = {}
            for frame in results:
                frame["ocr_text"] = texts.get(str(frame.get("frame_uid") or ""), "")
        return results

    def search_ocr(self, keywords: list[str], *, top_k: int) -> list[dict[str, Any]]:
        if self.ocr_service is None:
            raise RuntimeError("OCR service is not ready")
        return self.ocr_service.execute_single(" ".join(keywords), top_k)

    def search_speech(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        """Delegate compatibility ASR search to the canonical Branch-3 service."""
        if self.asr_service is None:
            raise RuntimeError("ASR service is not ready")
        return self.asr_service.execute_single(query, top_k)

    def get_video_frame_count(self, video_id: str) -> int:
        if self.metadata is None:
            return 0
        return len(self.metadata.video_frames(video_id.upper().replace("-", "_")))
