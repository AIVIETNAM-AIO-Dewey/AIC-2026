"""Unit test for Qdrant multi-vector indexing and retrieval."""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient

from online.src.index.qdrant_indexer import QdrantIndexer
from online.src.retrieval.embeddings import ModelRegistry


class TestQdrantIndexing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models = ModelRegistry()
        cls.test_db_path = Path("/tmp/qdrant_test_db")
        cls.indexer = QdrantIndexer(qdrant_db_path=str(cls.test_db_path), models=cls.models)
        cls.indexer.init_collections(force_recreate=True)

        # Base dataset paths
        cls.map_dir = Path("/Users/khoale/Downloads/AIC_HCM/map-keyframes")
        cls.scene_dir = Path("/Users/khoale/Downloads/[AIC2026] Scene Embeddings")
        cls.dam_dir = Path("/Users/khoale/Downloads/AIC_HCM/artifacts/dam_descriptions")
        cls.asr_dir = Path("/Users/khoale/Downloads/AIC_HCM/artifacts/asr_segments")

    def test_sample_video_indexing(self):
        """Index a single sample video (L21_V001) and verify vector search."""
        stats = self.indexer.index_all_videos(
            map_dir=self.map_dir,
            scene_dir=self.scene_dir,
            dam_dir=self.dam_dir,
            asr_dir=self.asr_dir,
            video_ids=["L21_V001"],
        )

        self.assertGreater(stats["keyframes"], 300)
        self.assertGreater(stats["dam_objects"], 500)

        # 1. Test Visual Vector Search (768-d)
        dummy_vis_vec = np.random.randn(768).astype(np.float32)
        dummy_vis_vec /= np.linalg.norm(dummy_vis_vec)

        vis_hits = self.indexer.client.query_points(
            collection_name=QdrantIndexer.KEYFRAME_COLLECTION,
            query=dummy_vis_vec.tolist(),
            using="visual",
            limit=5,
        ).points

        self.assertEqual(len(vis_hits), 5)
        self.assertEqual(vis_hits[0].payload["video_id"], "L21_V001")

        # 2. Test DAM Object Vector Search (1024-d)
        dummy_obj_vec = np.random.randn(1024).astype(np.float32)
        dummy_obj_vec /= np.linalg.norm(dummy_obj_vec)

        dam_hits = self.indexer.client.query_points(
            collection_name=QdrantIndexer.DAM_COLLECTION,
            query=dummy_obj_vec.tolist(),
            limit=5,
        ).points

        self.assertEqual(len(dam_hits), 5)
        self.assertIn("class_entity", dam_hits[0].payload)
        self.assertIn("bbox", dam_hits[0].payload)


if __name__ == "__main__":
    unittest.main()
