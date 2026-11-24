"""Small typed HTTP client for the local Qdrant service."""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np


def base_frame(
    payload: dict[str, Any], *, score: float, rank: int, score_type: str
) -> dict[str, Any]:
    video_id = str(payload["video_id"])
    frame_idx = int(payload["frame_idx"])
    return {
        "rank": rank,
        "video_id": video_id,
        "keyframe_n": int(payload.get("keyframe_n", 1)),
        "frame_idx": frame_idx,
        "pts_time_s": float(payload.get("pts_time_s", 0.0)),
        "fps": float(payload.get("fps", 0.0)),
        "image_relpath": str(
            payload.get("image_relpath") or f"keyframes/{video_id}/{frame_idx:08d}.jpg"
        ),
        "submission_string": f"{video_id}, {frame_idx}",
        "score": round(float(score), 6),
        "score_type": score_type,
        "dam_summary": "",
        "asr_transcript": "",
        "ocr_text": "",
    }


class QdrantHttpClient:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=120.0)

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        response = self.client.post(f"{self.url}{path}", json=body)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "ok":
            raise RuntimeError(f"Qdrant request failed: {payload}")
        return payload["result"]

    def collection(self, name: str) -> dict[str, Any]:
        response = self.client.get(f"{self.url}/collections/{name}")
        response.raise_for_status()
        return response.json()["result"]

    def count(self, collection: str, query_filter: dict[str, Any] | None = None) -> int:
        body: dict[str, Any] = {"exact": True}
        if query_filter:
            body["filter"] = query_filter
        result = self._post(f"/collections/{collection}/points/count", body)
        return int(result.get("count", 0))

    def query(
        self,
        collection: str,
        vector_name: str,
        vector: np.ndarray,
        limit: int,
        query_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "query": np.asarray(vector, dtype=np.float32).tolist(),
            "using": vector_name,
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
            "params": {"hnsw_ef": 96, "quantization": {"rescore": True, "oversampling": 2.0}},
        }
        if query_filter:
            body["filter"] = query_filter
        result = self._post(f"/collections/{collection}/points/query", body)
        return list(result["points"])

    def query_by_id(
        self,
        collection: str,
        vector_name: str,
        point_id: int,
        limit: int,
        query_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Find neighbours of an indexed point without running a query encoder."""

        body: dict[str, Any] = {
            "query": int(point_id),
            "using": vector_name,
            "limit": int(limit),
            "with_payload": True,
            "with_vector": False,
            "params": {
                "hnsw_ef": 96,
                "quantization": {"rescore": True, "oversampling": 2.0},
            },
        }
        if query_filter:
            body["filter"] = query_filter
        result = self._post(f"/collections/{collection}/points/query", body)
        return list(result["points"])

    def find_frame_point(
        self,
        collection: str,
        video_id: str,
        frame_idx: int,
    ) -> dict[str, Any] | None:
        """Resolve one exact canonical frame to its collection-local point."""

        result = self._post(
            f"/collections/{collection}/points/scroll",
            {
                "filter": {
                    "must": [
                        {"key": "video_id", "match": {"value": str(video_id)}},
                        {"key": "frame_idx", "match": {"value": int(frame_idx)}},
                    ]
                },
                "limit": 2,
                "with_payload": True,
                "with_vector": False,
            },
        )
        points = list(result.get("points") or [])
        if not points:
            return None
        if len(points) != 1:
            raise RuntimeError(
                f"Canonical frame {video_id}:{frame_idx} maps to {len(points)} Qdrant points"
            )
        return dict(points[0])

    def retrieve(self, collection: str, ids: list[int]) -> dict[int, dict[str, Any]]:
        if not ids:
            return {}
        result = self._post(
            f"/collections/{collection}/points",
            {"ids": ids, "with_payload": True, "with_vector": False},
        )
        return {int(point["id"]): point["payload"] for point in result}

    def dam_for_frame(self, video_id: str, frame_idx: int) -> list[dict[str, Any]]:
        result = self._post(
            "/collections/aic_dam_regions/points/scroll",
            {
                "filter": {
                    "must": [
                        {"key": "video_id", "match": {"value": video_id}},
                        {"key": "frame_idx", "match": {"value": frame_idx}},
                    ]
                },
                "limit": 100,
                "with_payload": True,
                "with_vector": False,
            },
        )
        return [point["payload"] for point in result["points"]]

    def close(self) -> None:
        self.client.close()


__all__ = ["QdrantHttpClient", "base_frame"]
