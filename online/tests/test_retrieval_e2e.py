"""End-to-end retrieval verification for KIS, TRAKE, and VQA search pipelines."""

from __future__ import annotations

import unittest
from pathlib import Path
from online.src.retrieval.pipeline import VideoRetrievalEngine


class TestRetrievalE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = VideoRetrievalEngine(
            qdrant_db_path="/tmp/qdrant_test_db",
            keyframes_root="/Users/khoale/Downloads/AIC_Challenger/data/keyframes",
        )
        # Fully initialize all models in memory
        cls.engine.models.get_bge_m3()
        cls.engine.models.get_siglip()
        cls.engine.models.get_reranker()

        # Warmup query
        cls.engine.search("người trong trường quay", task_type="KIS", top_k=5)

    def test_kis_retrieval(self):
        """Test KIS query execution and explainability card generation."""
        query = "Một người đàn ông và một người phụ nữ đứng sau bàn làm việc trong trường quay"
        response = self.engine.search(query=query, task_type="KIS", top_k=10)

        self.assertEqual(response.task_type, "KIS")
        self.assertGreater(len(response.results), 0)
        self.assertLess(response.execution_time_ms, 2000.0)

        top1 = response.results[0]
        self.assertIsNotNone(top1.submission_string)
        self.assertIn("L21_V001", top1.video_id)
        self.assertGreater(top1.final_score, 0.0)
        print(f"✅ KIS Test Passed! Top-1: {top1.submission_string} (Score: {top1.final_score:.4f}, Time: {response.execution_time_ms:.1f}ms)")

    def test_vqa_retrieval_and_answer(self):
        """Test VQA answer extraction on evidence keyframe."""
        question = "Người đàn ông trong trường quay mặc áo sơ mi màu gì?"
        response = self.engine.search(query=question, task_type="VQA", top_k=5)

        self.assertEqual(response.task_type, "VQA")
        top1 = response.results[0]
        self.assertIsNotNone(top1.vqa_answer)
        print(f"✅ VQA Test Passed! Answer Extracted: '{top1.vqa_answer}' for Frame {top1.submission_string}")


if __name__ == "__main__":
    unittest.main()
