"""Branch-1 ranking and fusion contracts without loading model weights."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from online.src.retrieval.branches.branch1.service import (
    Branch1Search,
    PersistentQueryEmbeddingCache,
    QUERY_ROLES,
    aggregate_model_streams,
    fuse_model_candidates,
)


def point(point_id: int, frame_uid: str, score: float) -> dict:
    video_id, frame_idx = frame_uid.split(":")
    return {
        "id": point_id,
        "score": score,
        "payload": {
            "frame_uid": frame_uid,
            "video_id": video_id,
            "frame_idx": int(frame_idx),
            "keyframe_n": int(frame_idx),
        },
    }


class FakeEncoder:
    revisions = {"siglip2": "s", "metaclip2": "m", "beit3": "b"}

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.unloads = 0

    def encode(self, model_name: str, texts: list[str]):
        self.calls.append((model_name, texts))
        dimension = {"siglip2": 768, "metaclip2": 1024, "beit3": 768}[model_name]
        rows = 6 if model_name == "beit3" else 12
        return np.ones((rows, dimension), dtype=np.float32), [
            {"token_count": 1, "max_tokens": 64, "truncated": False} for _ in texts
        ]

    def unload(self) -> None:
        self.unloads += 1


class FakeQdrant:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def query(self, collection, vector_name, vector, limit):
        self.calls.append((collection, vector_name, limit))
        stream = (len(self.calls) - 1) % 6
        return [
            point(1, "L01_V001:1", 0.8 - stream * 0.01),
            point(2, "L01_V001:2", 0.5 + stream * 0.01),
        ]


class Branch1RankingTests(unittest.TestCase):
    def test_bilingual_streams_keep_language_provenance_and_max_score(self) -> None:
        streams = []
        stream_keys = tuple(
            f"{role}:{language}"
            for role in QUERY_ROLES
            for language in ("vi", "en")
        )
        for index, _stream in enumerate(stream_keys):
            streams.append([point(1, "L01_V001:1", 0.2 + index * 0.01)])
        result = aggregate_model_streams(stream_keys, streams)["L01_V001:1"]
        self.assertAlmostEqual(result["raw_score"], 0.31)
        self.assertEqual(result["best_query_role"], "keyword")
        self.assertEqual(result["best_query_language"], "en")
        self.assertEqual(set(result["query_scores"]), set(stream_keys))
        self.assertEqual(result["query_scores"]["original:vi"]["language"], "vi")
        self.assertEqual(result["query_scores"]["original:en"]["language"], "en")

    def test_max_cosine_and_best_query_provenance(self) -> None:
        streams = [
            [point(1, "L01_V001:1", 0.1 + index * 0.1)] for index in range(6)
        ]
        result = aggregate_model_streams(QUERY_ROLES, streams)["L01_V001:1"]
        self.assertAlmostEqual(result["raw_score"], 0.6)
        self.assertEqual(result["best_query_role"], "keyword")
        self.assertEqual(result["best_query_rank"], 1)
        self.assertEqual(set(result["query_scores"]), set(QUERY_ROLES))

    def test_near_zero_std_assigns_half(self) -> None:
        streams = [[point(1, "L01_V001:1", 0.4), point(2, "L01_V001:2", 0.4)]] * 6
        result = aggregate_model_streams(QUERY_ROLES, streams)
        self.assertEqual(result["L01_V001:1"]["normalized_score"], 0.5)
        self.assertEqual(result["L01_V001:2"]["normalized_score"], 0.5)

    def test_missing_model_score_is_zero_and_weights_are_applied(self) -> None:
        siglip = aggregate_model_streams(
            QUERY_ROLES, [[point(1, "L01_V001:1", 0.5)]] * 6
        )
        results = fuse_model_candidates(
            {"siglip2": siglip, "metaclip2": {}, "beit3": {}},
            {"siglip2": 0.45, "metaclip2": 0.30, "beit3": 0.25},
            10,
        )
        self.assertAlmostEqual(results[0]["final_score"], 0.225)
        self.assertEqual(results[0]["model_provenance"]["metaclip2"]["normalized_score"], 0)
        self.assertIsNone(results[0]["model_provenance"]["beit3"]["raw_cosine"])

    def test_execute_runs_thirty_streams_and_persists_cache(self) -> None:
        encoder = FakeEncoder()
        qdrant = FakeQdrant()
        bundle = {
            "schema_version": "branch1.query.v1",
            "queries": [
                {"role": role, "vi": f"vi {role}", "en": f"en {role}"} for role in QUERY_ROLES
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            cache = PersistentQueryEmbeddingCache(Path(directory) / "cache.sqlite3")
            service = Branch1Search(qdrant, encoder, cache)
            response = service.execute(
                bundle,
                {"siglip2": 0.45, "metaclip2": 0.30, "beit3": 0.25},
                2000,
                1500,
            )
            self.assertEqual(len(qdrant.calls), 30)
            self.assertEqual([name for name, _ in encoder.calls], ["siglip2", "metaclip2", "beit3"])
            self.assertEqual(response["result_count"], 2)
            service.execute(
                bundle,
                {"siglip2": 0.45, "metaclip2": 0.30, "beit3": 0.25},
                2000,
                1500,
            )
            self.assertEqual(len(encoder.calls), 3, "second search must use the persistent cache")
            self.assertEqual(len(qdrant.calls), 60)

    def test_execute_rejects_a_non_fixed_output_gate(self) -> None:
        encoder = FakeEncoder()
        qdrant = FakeQdrant()
        bundle = {
            "schema_version": "branch1.query.v1",
            "queries": [
                {"role": role, "vi": f"vi {role}", "en": f"en {role}"}
                for role in QUERY_ROLES
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            cache = PersistentQueryEmbeddingCache(Path(directory) / "cache.sqlite3")
            service = Branch1Search(qdrant, encoder, cache)
            with self.assertRaisesRegex(ValueError, "fixed at 1500"):
                service.execute(
                    bundle,
                    {"siglip2": 0.45, "metaclip2": 0.30, "beit3": 0.25},
                    2000,
                    1499,
                )


if __name__ == "__main__":
    unittest.main()
