"""BM25 v2 must preserve canonical metadata for sparse-only frames."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from online.src.retrieval.branches.branch2 import sparse as sparse_module
from online.src.retrieval.branches.branch2.sparse import DamBm25Index


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SparseCanonicalMetadataTests(unittest.TestCase):
    def test_sparse_only_hit_uses_canonical_point_time_fps_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            state = Path(directory) / "state"
            dense = root / "dense_text_embeddings"
            canonical = root / "visual_embeddings" / "metaclip2"
            dense.mkdir(parents=True)
            canonical.mkdir(parents=True)
            frames = [
                {
                    "point_id": 1,
                    "frame_uid": "L01_V001:4",
                    "video_id": "L01_V001",
                    "frame_idx": 4,
                    "keyframe_n": 1,
                    "pts_time_s": 0.1333,
                    "fps": 30.0,
                    "image_relpath": "keyframes/L01_V001/00000004.jpg",
                },
                {
                    "point_id": 2,
                    "frame_uid": "L01_V001:31",
                    "video_id": "L01_V001",
                    "frame_idx": 31,
                    "keyframe_n": 2,
                    "pts_time_s": 1.0333,
                    "fps": 30.0,
                    "image_relpath": "keyframes/L01_V001/00000031.jpg",
                },
            ]
            frame_path = canonical / "keyframes_metadata.jsonl"
            frame_path.write_text(
                "".join(json.dumps(row) + "\n" for row in frames), encoding="utf-8"
            )
            dam_path = dense / "dam_metadata.jsonl"
            dam_path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in (
                        {
                            "video_id": "L01_V001",
                            "frame_idx": 4,
                            "region_id": "r1",
                            "class_entity": "sky",
                            "bbox": [0, 0, 1, 1],
                            "description_en": "blue sky",
                        },
                        {
                            "video_id": "L01_V001",
                            "frame_idx": 31,
                            "region_id": "r2",
                            "class_entity": "car",
                            "bbox": [0, 0, 1, 1],
                            "description_en": "red car",
                        },
                    )
                ),
                encoding="utf-8",
            )
            state.mkdir()
            (state / "branch2_dam_manifest.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "status": "ready",
                        "metadata_sha256": sha256(dam_path),
                        "frame_metadata_sha256": sha256(frame_path),
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(sparse_module, "EXPECTED_FRAMES", 2),
                patch.object(sparse_module, "EXPECTED_DAM_REGIONS", 2),
            ):
                DamBm25Index.prepare(root, state, sha256(dam_path), sha256(frame_path))
                results = DamBm25Index(root, state).search(["sky"] * 6, 2)
            frame = results["L01_V001:4"]
            self.assertEqual(frame["global_idx"], 1)
            self.assertEqual(frame["pts_time_s"], 0.1333)
            self.assertEqual(frame["fps"], 30.0)
            self.assertEqual(frame["image_relpath"], "keyframes/L01_V001/00000004.jpg")


if __name__ == "__main__":
    unittest.main()
