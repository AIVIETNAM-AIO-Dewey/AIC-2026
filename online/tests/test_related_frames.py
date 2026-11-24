"""Tests for exact frame identity and encoder-free related-frame filling."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from online.src.retrieval.infrastructure.metadata import FrameMetadataStore
from online.src.submission.related import RelatedFrameSearch, fuse_related_pools


def _payload(uid: str, point_id: int) -> dict:
    video_id, raw_frame_idx = uid.split(":")
    frame_idx = int(raw_frame_idx)
    return {
        "id": point_id,
        "payload": {
            "point_id": point_id,
            "video_id": video_id,
            "frame_idx": frame_idx,
            "keyframe_n": frame_idx // 10,
            "pts_time_s": frame_idx / 25,
            "fps": 25.0,
            "frame_uid": uid,
            "image_relpath": f"keyframes/{video_id}/{frame_idx:08d}.jpg",
        },
    }


class _Metadata:
    def __init__(self, frames: list[dict]) -> None:
        self.frames = frames

    def video_frames(self, video_id: str):
        return tuple(frame for frame in self.frames if frame["video_id"] == video_id)


class _Qdrant:
    def __init__(self, pools: dict[str, list[dict]]) -> None:
        self.pools = pools
        self.calls: list[tuple[str, str, int, int]] = []

    def find_frame_point(self, collection: str, video_id: str, frame_idx: int):
        self.calls.append(("seed", collection, frame_idx, 0))
        return _payload(f"{video_id}:{frame_idx}", 1)

    def query_by_id(self, collection: str, vector_name: str, point_id: int, limit: int):
        self.calls.append((collection, vector_name, point_id, limit))
        return self.pools[vector_name]


class RelatedFrameTests(unittest.TestCase):
    def test_weighted_rrf_excludes_seed_deduplicates_and_is_deterministic(self) -> None:
        seed = _payload("L01_V001:10", 1)
        first = _payload("L01_V001:20", 2)
        second = _payload("L02_V001:30", 3)
        pools = {
            "siglip2": [seed, first, first, second],
            "metaclip2": [second, first],
            "beit3": [first, second],
        }
        results = fuse_related_pools(pools, seed_uid="L01_V001:10", limit=2)
        self.assertEqual([item["frame_uid"] for item in results], ["L01_V001:20", "L02_V001:30"])
        self.assertEqual([item["rank"] for item in results], [1, 2])
        self.assertTrue(all(item["frame_uid"] != "L01_V001:10" for item in results))
        self.assertEqual(results, fuse_related_pools(pools, seed_uid="L01_V001:10", limit=2))

    def test_service_uses_all_stored_vectors_and_never_needs_an_encoder(self) -> None:
        frames = [
            _payload("L01_V001:10", 1)["payload"],
            _payload("L01_V001:20", 2)["payload"],
            _payload("L02_V001:30", 3)["payload"],
        ]
        pools = {
            "siglip2": [_payload("L01_V001:10", 1), _payload("L01_V001:20", 2)],
            "metaclip2": [_payload("L02_V001:30", 3)],
            "beit3": [_payload("L01_V001:20", 2)],
        }
        qdrant = _Qdrant(pools)
        result = RelatedFrameSearch(qdrant, _Metadata(frames)).execute("L01_V001", 10, 2)
        self.assertFalse(result["query_pipeline_invoked"])
        self.assertEqual(result["seed"]["frame_uid"], "L01_V001:10")
        self.assertEqual(result["result_count"], 2)
        self.assertEqual(
            {(call[0], call[1]) for call in qdrant.calls[1:]},
            {
                ("aic_frames", "siglip2"),
                ("aic_frames", "metaclip2"),
                ("aic_beit3_frames", "beit3"),
            },
        )


class MetadataIdentityTests(unittest.TestCase):
    def test_unified_enrichment_cannot_overwrite_canonical_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_dir = root / "visual_embeddings" / "metaclip2" / "video_metadata"
            unified_dir = root / "unified_metadata"
            video_dir.mkdir(parents=True)
            unified_dir.mkdir()
            canonical = _payload("L01_V001:20", 42)["payload"]
            (video_dir / "L01_V001.jsonl").write_text(
                json.dumps(canonical) + "\n",
                encoding="utf-8",
            )
            (unified_dir / "L01_V001.jsonl").write_text(
                json.dumps(
                    {
                        **canonical,
                        "point_id": 2,
                        "frame_idx": 999,
                        "frame_uid": "L01_V001:999",
                        "image_relpath": "wrong.jpg",
                        "dam_summary_en": "usable enrichment",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            store = FrameMetadataStore(root, object())
            detail = store.detail("L01_V001", 2)
            self.assertIsNotNone(detail)
            self.assertEqual(detail["point_id"], 42)
            self.assertEqual(detail["frame_idx"], 20)
            self.assertEqual(detail["frame_uid"], "L01_V001:20")
            self.assertEqual(detail["image_relpath"], "keyframes/L01_V001/00000020.jpg")
            self.assertEqual(detail["dam_summary_en"], "usable enrichment")


if __name__ == "__main__":
    unittest.main()
