"""Branch-2 score contracts that do not require model weights or Qdrant."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from online.src.retrieval.branches.branch2.contracts import normalize_weights
from online.src.retrieval.branches.branch2.dense import DamDenseRetriever, normalized_lse
from online.src.retrieval.branches.branch2.fusion import fuse_dense_sparse
from online.src.retrieval.branches.rerankers.beit3_cosine import Beit3CosineReranker


class FakeQdrant:
    def __init__(self, *, incomplete: bool = False) -> None:
        self.filters = []
        self.incomplete = incomplete

    def query(self, collection, vector_name, vector, limit, query_filter=None):
        self.filters.append(query_filter)
        points = [
            {"id": 1, "score": 0.1, "payload": {"frame_uid": "f1"}},
            {"id": 2, "score": 0.9, "payload": {"frame_uid": "f2"}},
        ]
        return points[:1] if self.incomplete else points[:limit]


class Branch2ContractTests(unittest.TestCase):
    def test_dam_health_exposes_ingestion_fingerprints(self) -> None:
        class MatrixHeader:
            shape = (681_355, 1024)
            dtype = np.dtype(np.float16)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = root / "dam_vectors.f16.npy"
            metadata_path = root / "dam_metadata.jsonl"
            manifest_path = root / "branch2_dam_manifest.json"
            matrix_path.write_bytes(b"matrix-fixture")
            metadata_path.write_bytes(b"metadata-fixture")
            matrix_stat = matrix_path.stat()
            metadata_stat = metadata_path.stat()
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": DamDenseRetriever.MANIFEST_SCHEMA_VERSION,
                        "status": "ready",
                        "passed": True,
                        "vector_count": 681_355,
                        "metadata_count": 681_355,
                        "model_id": "BAAI/bge-m3",
                        "pooling": "cls",
                        "normalization": "l2",
                        "l2_normalized": True,
                        "dimension": 1024,
                        "dtype": "float16",
                        "finite_verified": True,
                        "frame_mapping_verified": True,
                        "region_identity_verified": True,
                        "offline_identity": {
                            "revision_verified": False,
                            "evidence": "fixture manifest",
                        },
                        "matrix_size": matrix_stat.st_size,
                        "matrix_mtime_ns": matrix_stat.st_mtime_ns,
                        "metadata_size": metadata_stat.st_size,
                        "metadata_mtime_ns": metadata_stat.st_mtime_ns,
                        "matrix_sha256": "matrix-sha256",
                        "metadata_sha256": "metadata-sha256",
                        "frame_metadata_sha256": "frame-metadata-sha256",
                    }
                ),
                encoding="utf-8",
            )
            retriever = object.__new__(DamDenseRetriever)
            retriever.matrix_path = matrix_path
            retriever.metadata_path = metadata_path
            retriever.manifest_path = manifest_path
            retriever.temperature = 0.05

            with patch(
                "online.src.retrieval.branches.branch2.dense.np.load",
                return_value=MatrixHeader(),
            ):
                health = retriever.health()

        self.assertTrue(health["ready"])
        self.assertEqual(health["matrix_sha256"], "matrix-sha256")
        self.assertEqual(health["metadata_sha256"], "metadata-sha256")
        self.assertEqual(
            health["frame_metadata_sha256"], "frame-metadata-sha256"
        )

    def test_weights_are_normalized(self) -> None:
        self.assertEqual(normalize_weights({"dense": 7, "sparse": 3}, ("dense", "sparse")), {"dense": 0.7, "sparse": 0.3})

    def test_hybrid_fusion_preserves_missing_scores_and_order(self) -> None:
        dense = {
            "f1": {"frame_uid": "f1", "dense_raw": 0.9, "dense_rank": 1},
            "f2": {"frame_uid": "f2", "dense_raw": 0.1, "dense_rank": 2},
        }
        sparse = {"f2": {"frame_uid": "f2", "sparse_raw": 1 / 61, "sparse_rank": 1}}
        results = fuse_dense_sparse(dense, sparse, {"dense": 0.7, "sparse": 0.3}, 10)
        self.assertEqual({item["frame_uid"] for item in results}, {"f1", "f2"})
        self.assertEqual(results[0]["frame_uid"], "f1")
        self.assertEqual(results[0]["sparse_normalized"], 0.0)
        self.assertGreater(results[0]["dense_normalized"], 0.5)

    def test_invalid_weight_group_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_weights({"dense": 0, "sparse": 0}, ("dense", "sparse"))

    def test_normalized_lse_matches_definition(self) -> None:
        scores = np.asarray([[0.2, 0.4, 0.1], [0.5, 0.5, 0.5]], dtype=np.float32)
        actual = normalized_lse(scores, 0.05)
        expected = []
        for row in scores:
            expected.append(0.05 * np.log(np.exp(row / 0.05).sum()) - 0.05 * np.log(3))
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_hybrid_provenance_is_bound_to_each_frame(self) -> None:
        dense = {
            "f1": {"frame_uid": "f1", "dense_raw": 0.9, "dense_rank": 1, "dense_best_query_role": "entity"},
            "f2": {"frame_uid": "f2", "dense_raw": 0.1, "dense_rank": 2, "dense_best_query_role": "context"},
        }
        sparse = {
            "f1": {"frame_uid": "f1", "sparse_raw": 0.01, "sparse_rank": 2, "sparse_best_query_role": "keyword"},
            "f2": {"frame_uid": "f2", "sparse_raw": 0.02, "sparse_rank": 1, "sparse_best_query_role": "original"},
        }
        results = {item["frame_uid"]: item for item in fuse_dense_sparse(dense, sparse, {"dense": 0.7, "sparse": 0.3}, 10)}
        self.assertEqual(results["f1"]["hybrid_provenance"]["dense"]["best_query_role"], "entity")
        self.assertEqual(results["f2"]["hybrid_provenance"]["dense"]["best_query_role"], "context")

    def test_fusion_rejects_conflicting_canonical_identity(self) -> None:
        with self.assertRaises(ValueError):
            fuse_dense_sparse(
                {
                    "f1": {
                        "frame_uid": "f1",
                        "global_idx": 1,
                        "video_id": "L01_V001",
                        "frame_idx": 4,
                        "dense_raw": 0.9,
                        "dense_rank": 1,
                    }
                },
                {
                    "f1": {
                        "frame_uid": "f1",
                        "global_idx": 2,
                        "video_id": "L01_V001",
                        "frame_idx": 4,
                        "sparse_raw": 0.1,
                        "sparse_rank": 1,
                    }
                },
                {"dense": 0.7, "sparse": 0.3},
                10,
            )

    def test_beit_filter_schema_and_tail_order(self) -> None:
        qdrant = FakeQdrant()
        reranker = Beit3CosineReranker(qdrant, object(), {"f1": 1, "f2": 2, "f3": 3})
        candidates = [
            {"frame_uid": "f1", "hybrid_score": 0.9, "hybrid_rank": 1},
            {"frame_uid": "f2", "hybrid_score": 0.8, "hybrid_rank": 2},
            {"frame_uid": "f3", "hybrid_score": 0.7, "hybrid_rank": 3},
        ]
        output, info = reranker.rerank(
            candidates,
            ["q"] * 6,
            top_k=2,
            weights={"beit3": 0.4, "previous": 0.6},
            text_vectors=np.ones((6, 768), dtype=np.float32),
        )
        self.assertEqual(qdrant.filters, [{"must": [{"has_id": [1, 2]}]}] * 6)
        self.assertEqual(output[2]["frame_uid"], "f3")
        self.assertNotIn("beit3_raw_cosine", output[2])
        self.assertEqual(output[2]["hybrid_score"], candidates[2]["hybrid_score"])
        self.assertEqual(info["rerank_count"], 2)
        self.assertEqual(output[0]["rerank_formula"]["beit3_weight"], 0.4)

    def test_beit_missing_filtered_evidence_fails_closed(self) -> None:
        reranker = Beit3CosineReranker(FakeQdrant(incomplete=True), object(), {"f1": 1, "f2": 2})
        with self.assertRaises(RuntimeError):
            reranker.rerank(
                [
                    {"frame_uid": "f1", "hybrid_score": 0.9},
                    {"frame_uid": "f2", "hybrid_score": 0.8},
                ],
                ["q"] * 6,
                top_k=2,
                weights={"beit3": 0.4, "previous": 0.6},
                text_vectors=np.ones((6, 768), dtype=np.float32),
            )

    def test_beit_reranker_never_accepts_more_than_100_candidates(self) -> None:
        reranker = Beit3CosineReranker(FakeQdrant(), object(), {})
        with self.assertRaises(ValueError):
            reranker.rerank(
                [{"frame_uid": "f1", "hybrid_score": 0.9}],
                ["q"] * 6,
                top_k=101,
                weights={"beit3": 0.4, "previous": 0.6},
                text_vectors=np.ones((6, 768), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
