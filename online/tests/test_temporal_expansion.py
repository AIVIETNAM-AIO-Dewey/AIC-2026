"""Tests for bounded multi-path ordered SigLIP search."""

from __future__ import annotations

import unittest

import numpy as np

from online.src.retrieval.modalities.workbench import IndependentModalitySearch


def _unit(index: int) -> np.ndarray:
    vector = np.zeros(8, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _hit(
    video_id: str,
    frame_idx: int,
    keyframe_n: int,
    pts_time_s: float,
    rank: int,
    score: float,
    global_idx: int,
) -> dict:
    return {
        "video_id": video_id,
        "frame_idx": frame_idx,
        "keyframe_n": keyframe_n,
        "pts_time_s": pts_time_s,
        "rank": rank,
        "score": score,
        "global_idx": global_idx,
        "score_type": "cosine",
        "submission_string": f"{video_id}, {frame_idx}",
    }


class _Registry:
    _queries = {"event one": 0, "event two": 1}

    def embed_siglip_text(self, text: str) -> np.ndarray:
        return _unit(self._queries[text])


class _Searcher:
    def __init__(self, pools: list[list[dict]]) -> None:
        self.pools = pools

    def search_visual(self, query_vector: np.ndarray, top_k: int = 100) -> list[dict]:
        index = int(np.argmax(query_vector))
        return [dict(item) for item in self.pools[index][:top_k]]


class TemporalExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = [
            {"order": 1, "description": "one", "global_scene_en": "event one"},
            {"order": 2, "description": "two", "global_scene_en": "event two"},
        ]
        self.pools = [
            [
                _hit("VIDEO", 10, 1, 1.0, 1, 0.90, 0),
                _hit("VIDEO", 20, 2, 2.0, 2, 0.80, 1),
            ],
            [
                _hit("VIDEO", 30, 3, 3.0, 1, 0.95, 2),
                _hit("VIDEO", 40, 4, 4.0, 2, 0.85, 3),
            ],
        ]

    def _search(self, **kwargs) -> dict:
        search = IndependentModalitySearch(
            searcher=_Searcher(self.pools),
            registry=_Registry(),
        )
        return search.search_temporal_intersection(
            events=self.events,
            top_k_per_event=kwargs.pop("top_k_per_event", 100),
            top_k_sequences=kwargs.pop("top_k_sequences", 20),
            max_gap_seconds=None,
            **kwargs,
        )

    def test_default_retains_one_best_path_per_video(self) -> None:
        result = self._search()
        self.assertEqual(result["paths_per_video"], 1)
        self.assertEqual(result["path_search_mode"], "legacy_exact_single_path")
        self.assertEqual(result["monotonic_video_count"], 1)
        self.assertEqual(result["ordered_sequence_count"], 1)
        self.assertEqual(result["sequences"][0]["matched_frames"], [10, 30])
        self.assertEqual(result["reserve_sequences"], [])

    def test_multi_path_mode_has_separate_return_and_reserve_counts(self) -> None:
        result = self._search(
            paths_per_video=3,
            top_k_sequences=1,
            sequence_reservoir_size=3,
        )
        self.assertEqual(result["monotonic_video_count"], 1)
        self.assertEqual(result["computed_sequence_count"], 3)
        self.assertEqual(result["result_count"], 1)
        self.assertEqual(result["sequence_reservoir_count"], 3)
        self.assertEqual(result["sequences"][0]["matched_frames"], [10, 30])
        self.assertEqual(
            [sequence["matched_frames"] for sequence in result["reserve_sequences"]],
            [[10, 40], [20, 30]],
        )
        signatures = {
            tuple(sequence["matched_frames"])
            for sequence in [*result["sequences"], *result["reserve_sequences"]]
        }
        self.assertEqual(len(signatures), 3)
        self.assertTrue(result["path_beam_applied"])

    def test_larger_event_pool_automatically_uses_bounded_search(self) -> None:
        result = self._search(top_k_per_event=301)
        self.assertEqual(result["path_search_mode"], "bounded_diverse_beam")
        self.assertEqual(result["sequences"][0]["matched_frames"], [10, 30])

    def test_multi_path_diversity_can_require_different_frames_for_every_event(self) -> None:
        result = self._search(
            paths_per_video=3,
            top_k_sequences=2,
            sequence_reservoir_size=3,
            path_diversity_min_events=2,
        )
        self.assertEqual(result["computed_sequence_count"], 2)
        self.assertEqual(
            [sequence["matched_frames"] for sequence in result["sequences"]],
            [[10, 30], [20, 40]],
        )

    def test_temporal_expansion_limits_are_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k_per_event cannot exceed"):
            self._search(top_k_per_event=1_001)
        with self.assertRaisesRegex(ValueError, "paths_per_video must be between"):
            self._search(paths_per_video=11)
        with self.assertRaisesRegex(ValueError, "cannot be smaller"):
            self._search(top_k_sequences=2, sequence_reservoir_size=1)


if __name__ == "__main__":
    unittest.main()
