"""Branch-1 data-gate freshness tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from online.src.retrieval.branches.branch1.health import (
    DATA_GATE_SCHEMA_VERSION,
    _gate_artifacts_current,
    _collection_status,
    _gate_fingerprints,
    _gate_model_status,
)
from scripts.qdrant import prepare_branch1


class Branch1HealthGateTests(unittest.TestCase):
    def test_collection_readiness_uses_exact_count(self) -> None:
        class ApproximateCountQdrant:
            def collection(self, _collection: str) -> dict:
                return {
                    "status": "green",
                    "points_count": 248_116,
                    "config": {
                        "params": {
                            "vectors": {
                                "beit3": {"size": 768, "distance": "Cosine"}
                            }
                        }
                    },
                }

            def count(self, _collection: str, query_filter=None) -> int:
                return 247_956

        status = _collection_status(
            ApproximateCountQdrant(),
            "aic_beit3_frames",
            "beit3",
            768,
        )

        self.assertTrue(status["ready"])
        self.assertEqual(status["points_count"], 247_956)
        self.assertEqual(status["exact_points_count"], 247_956)
        self.assertEqual(status["approximate_points_count"], 248_116)

    def _fixture(self, root: Path) -> tuple[dict, tuple[Path, ...]]:
        metaclip = root / "visual_embeddings" / "metaclip2"
        beit3 = root / "visual_embeddings" / "beit3"
        scene = root / "scene_embeddings"
        for directory in (metaclip, beit3, scene):
            directory.mkdir(parents=True)
        paths = {
            "meta_matrix": metaclip / "keyframes_visual_vectors.f16.npy",
            "canonical": metaclip / "keyframes_metadata.jsonl",
            "beit_matrix": beit3 / "keyframes_visual_vectors.f16.npy",
            "beit_metadata": beit3 / "keyframes_metadata.jsonl",
            "siglip": scene / "L01_V001.safetensors",
        }
        for path in paths.values():
            path.write_bytes(b"fixture")

        def record(path: Path) -> dict[str, object]:
            stat = path.stat()
            return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": "fixture-hash"}

        gate = {
            "schema_version": DATA_GATE_SCHEMA_VERSION,
            "status": "ready",
            "passed": True,
            "canonical_metadata": record(paths["canonical"]),
            "metaclip2": {"matrix": record(paths["meta_matrix"])},
            "beit3": {"matrix": record(paths["beit_matrix"]), "metadata": record(paths["beit_metadata"])},
            "siglip2": {"shards": [{"path": "scene_embeddings/L01_V001.safetensors", **record(paths["siglip"])}]},
        }
        artifacts = (
            paths["meta_matrix"],
            paths["canonical"],
            paths["siglip"],
        )
        return gate, artifacts

    def test_preparer_shape_requires_beit_metadata_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate, _ = self._fixture(root)
            self.assertTrue(_gate_artifacts_current(gate, root))
            del gate["beit3"]["metadata"]
            self.assertFalse(_gate_artifacts_current(gate, root))

    def test_gate_fingerprints_are_bound_to_selected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate, artifacts = self._fixture(root)
            fingerprints = _gate_fingerprints(gate, root, artifacts)
            self.assertEqual(
                fingerprints["visual_embeddings/metaclip2/keyframes_visual_vectors.f16.npy"],
                "fixture-hash",
            )
            self.assertEqual(
                fingerprints["scene_embeddings/L01_V001.safetensors"],
                "fixture-hash",
            )

    def test_preparer_delegates_to_the_runtime_gate_builder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate, _ = self._fixture(root)
            with patch.object(
                prepare_branch1, "build_data_gate_report", return_value=gate
            ) as builder:
                result = prepare_branch1.validate_data(
                    root, root / "visual_embeddings" / "beit3"
                )
            self.assertIs(result, gate)
            self.assertEqual(result["beit3"]["metadata"]["sha256"], "fixture-hash")
            builder.assert_called_once_with(
                root, root / "visual_embeddings" / "beit3"
            )

    def test_model_gate_requires_an_explicit_offline_identity_record(self) -> None:
        gate = {
            "schema_version": DATA_GATE_SCHEMA_VERSION,
            "passed": True,
            "metaclip2": {
                "vector_count": 247_956,
                "dimension": 1024,
                "dtype": "float16",
                "finite_verified": True,
                "l2_normalized": True,
                "ordering_verified": True,
                "metadata_rows": 247_956,
                "index_rows": 247_956,
            },
        }
        self.assertFalse(_gate_model_status(gate, "metaclip2", 1024)["ready"])
        gate["metaclip2"]["offline_identity"] = {
            "evidence": "legacy manifest",
            "revision_verified": False,
        }
        self.assertTrue(_gate_model_status(gate, "metaclip2", 1024)["ready"])


if __name__ == "__main__":
    unittest.main()
