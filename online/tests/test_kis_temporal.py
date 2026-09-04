"""Contract tests for full-KIS ordered event orchestration."""

from __future__ import annotations

import unittest

from online.src.retrieval.branches.branch1.contracts import QUERY_ROLES
from online.src.retrieval.branches.final_fusion.temporal import (
    KisTemporalFusionSearch,
    focus_event_query_bundle,
)


def query_bundle() -> dict[str, object]:
    return {
        "schema_version": "branch1.query.v1",
        "queries": [
            {"role": role, "vi": f"parent vi {role}", "en": f"parent en {role}"}
            for role in QUERY_ROLES
        ],
    }


def frame(video_id: str, frame_idx: int, rank: int, score: float) -> dict[str, object]:
    return {
        "frame_uid": f"{video_id}:{frame_idx}",
        "point_id": frame_idx,
        "global_idx": frame_idx,
        "video_id": video_id,
        "frame_idx": frame_idx,
        "keyframe_n": rank,
        "pts_time_s": frame_idx / 10.0,
        "fps": 10.0,
        "image_relpath": f"keyframes/{video_id}/{frame_idx:08d}.jpg",
        "submission_string": f"{video_id}, {frame_idx}",
        "rank": rank,
        "score": score,
        "final_score": score,
        "score_type": "beit3_coco_cosine_blend",
    }


class RecordingFusion:
    def __init__(self) -> None:
        self.bundles: list[dict[str, object]] = []
        self.weights: dict[str, float] | None = None
        self.health_already_checked = False

    def execute_batch(self, bundles, weights, *, _health_already_checked=False):
        self.bundles = bundles
        self.weights = weights
        self.health_already_checked = _health_already_checked
        return [
            {
                "fusion_applied": True,
                "final_top_k": 150,
                "branch_pool_counts": {
                    "branch1": 1500,
                    "branch2": 500,
                    "ocr": 500,
                    "asr": 500,
                },
                "timing": {"total_ms": 10.0},
                "results": [
                    frame("L01_V001", 10, 1, 0.90),
                    frame("L01_V002", 30, 2, 0.70),
                ],
            },
            {
                "fusion_applied": True,
                "final_top_k": 150,
                "branch_pool_counts": {
                    "branch1": 1500,
                    "branch2": 500,
                    "ocr": 500,
                    "asr": 500,
                },
                "timing": {"total_ms": 11.0},
                "results": [
                    frame("L01_V001", 20, 1, 0.80),
                    frame("L01_V002", 20, 2, 0.95),
                ],
            },
        ]


class KisTemporalTests(unittest.TestCase):
    def test_event_focus_keeps_six_roles_and_places_event_before_context(self) -> None:
        focused = focus_event_query_bundle(
            query_bundle(),
            {"description": "event", "vi": "su kien", "en": "event"},
        )
        by_role = {item["role"]: item for item in focused["queries"]}
        self.assertEqual(set(by_role), set(QUERY_ROLES))
        self.assertEqual(by_role["original"]["en"], "event")
        self.assertEqual(by_role["action"]["vi"], "su kien")
        self.assertTrue(by_role["entity"]["en"].startswith("event. Context: "))
        self.assertIn("parent en entity", by_role["entity"]["en"])

    def test_ordered_focus_inherits_only_shared_context(self) -> None:
        focused = focus_event_query_bundle(
            query_bundle(),
            {"description": "event", "vi": "su kien", "en": "event"},
            shared_context_only=True,
        )
        by_role = {item["role"]: item for item in focused["queries"]}
        self.assertEqual(by_role["original"]["en"], "event")
        self.assertEqual(by_role["action"]["en"], "event")
        self.assertIn("parent en context", by_role["entity"]["en"])
        self.assertIn("parent en context", by_role["keyword"]["en"])
        self.assertNotIn("parent en entity", by_role["entity"]["en"])
        self.assertNotIn("parent en keyword", by_role["keyword"]["en"])

    def test_every_event_runs_full_kis_before_strict_temporal_ordering(self) -> None:
        fusion = RecordingFusion()
        events = [
            {"order": 1, "description": "first", "vi": "dau", "en": "first"},
            {"order": 2, "description": "second", "vi": "sau", "en": "second"},
        ]
        result = KisTemporalFusionSearch(fusion).execute(
            query_bundle=query_bundle(),
            events=events,
            branch_weights={"branch1": 0.4, "branch2": 0.3, "ocr": 0.15, "asr": 0.15},
            top_k_sequences=10,
            max_gap_seconds=30.0,
            task_type="KIS",
            health_already_checked=True,
        )

        self.assertEqual(len(fusion.bundles), 2)
        self.assertTrue(fusion.health_already_checked)
        first_bundle = {item["role"]: item for item in fusion.bundles[0]["queries"]}
        self.assertIn("parent en context", first_bundle["entity"]["en"])
        self.assertNotIn("parent en entity", first_bundle["entity"]["en"])
        self.assertEqual(result["operation"], "ordered_kis_fusion")
        self.assertTrue(result["event_fusion_applied"])
        self.assertTrue(result["cross_modal_fusion_applied"])
        self.assertTrue(result["reranking_applied"])
        self.assertEqual(result["frame_index_base"], 0)
        self.assertEqual(result["ordering_fields"], ["frame_idx", "pts_time_s"])
        self.assertTrue(result["complete_sequence_required"])
        self.assertEqual(result["intersection_video_count"], 2)
        self.assertEqual(result["ordered_sequence_count"], 1)
        sequence = result["sequences"][0]
        self.assertEqual(sequence["video_id"], "L01_V001")
        self.assertEqual(sequence["matched_frames"], [10, 20])
        self.assertEqual(sequence["sequence_score"], 0.8)
        self.assertEqual(
            [event["retrieval_modality"] for event in sequence["matched_events"]],
            ["kis_fusion", "kis_fusion"],
        )

    def test_invalid_bundle_type_fails_with_contract_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "branch1.query.v1"):
            focus_event_query_bundle([], {"description": "event"})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
