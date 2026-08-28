"""Tests for canonical KIS/VQA/TRAKE submission preparation."""

from __future__ import annotations

import unittest

from online.src.submission.core import (
    SubmissionValidationError,
    build_submission,
    prepare_submission,
    validate_frame_reference,
)


def _frame(video_id: str, frame_idx: int, keyframe_n: int) -> dict:
    return {
        "video_id": video_id,
        "frame_idx": frame_idx,
        "keyframe_n": keyframe_n,
        "pts_time_s": float(keyframe_n),
        "image_relpath": f"keyframes/{video_id}/{keyframe_n:06d}.jpg",
    }


class CanonicalStore:
    def __init__(self, videos: dict[str, list[dict]]) -> None:
        self.videos = videos
        self.by_identity = {
            (str(frame["video_id"]), int(frame["frame_idx"])): frame
            for frames in videos.values()
            for frame in frames
        }

    def frame(self, video_id: str, frame_idx: int) -> dict | None:
        value = self.by_identity.get((video_id, frame_idx))
        return dict(value) if value is not None else None

    def video(self, video_id: str) -> list[dict]:
        return [dict(frame) for frame in self.videos.get(video_id, [])]


class SubmissionCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = CanonicalStore(
            {
                "A": [_frame("A", index * 10, index) for index in range(1, 121)],
                "B": [_frame("B", index * 10, index) for index in range(1, 6)],
            }
        )

    def test_reference_uses_canonical_values_not_spoofed_client_metadata(self) -> None:
        result = validate_frame_reference(
            {
                "video_id": "A",
                "frame_idx": 20,
                "keyframe_n": 999,
                "pts_time_s": 999.0,
                "modality": "siglip",
            },
            self.store.frame,
        )
        self.assertEqual(result["keyframe_n"], 2)
        self.assertEqual(result["pts_time_s"], 2.0)
        self.assertEqual(result["modality"], "siglip")

    def test_kis_keeps_manual_then_reservoir_then_canonical_neighbors(self) -> None:
        result = build_submission(
            "KIS",
            manual_items=[{"video_id": "A", "frame_idx": 30}],
            candidate_items=[
                {"video_id": "A", "frame_idx": 50, "modality": "dam", "rank": 1},
                {"video_id": "A", "frame_idx": 30},
                {"video_id": "UNKNOWN", "frame_idx": 1},
            ],
            frame_lookup=self.store.frame,
            video_frames_lookup=self.store.video,
            target_rows=5,
        )
        self.assertTrue(result["complete"])
        self.assertEqual(
            [row["frame_idx"] for row in result["rows"]],
            [30, 50, 20, 40, 10],
        )
        self.assertTrue(result["rows"][0]["manual"])
        self.assertEqual(result["rows"][1]["selection_origin"], "active_query_reservoir")
        self.assertEqual(result["rows"][1]["modality"], "dam")
        self.assertTrue(all(row["auto_filled"] for row in result["rows"][2:]))
        self.assertEqual(result["auto_filled_row_count"], 3)
        self.assertEqual(len(result["warnings"]), 1)

    def test_default_completion_produces_100_unique_verified_rows(self) -> None:
        result = build_submission(
            "KIS",
            manual_items=[{"video_id": "A", "frame_idx": 600}],
            candidate_items=[],
            frame_lookup=self.store.frame,
            video_frames_lookup=self.store.video,
        )
        self.assertTrue(result["complete"])
        self.assertEqual(result["row_count"], 100)
        self.assertEqual(result["rows"][0]["frame_idx"], 600)
        identities = {(row["video_id"], row["frame_idx"]) for row in result["rows"]}
        self.assertEqual(len(identities), 100)
        self.assertTrue(all(self.store.frame(*identity) is not None for identity in identities))

    def test_invalid_manual_frame_is_a_blocking_error(self) -> None:
        with self.assertRaisesRegex(SubmissionValidationError, "Invalid manual frame"):
            build_submission(
                "KIS",
                manual_items=[{"video_id": "A", "frame_idx": 999_999}],
                candidate_items=[],
                frame_lookup=self.store.frame,
            )

    def test_vqa_requires_human_answer_and_preserves_it_at_query_level(self) -> None:
        invalid = prepare_submission(
            {
                "task_type": "VQA",
                "query_id": "query-1",
                "manual_selections": [{"video_id": "A", "frame_idx": 10}],
                "target_rows": 1,
            },
            frame_lookup=self.store.frame,
        )
        self.assertFalse(invalid["ok"])
        self.assertIn("human-provided VQA answer", invalid["errors"][0])

        valid = prepare_submission(
            {
                "task_type": "VQA",
                "mode": "siglip",
                "query_id": "query-1",
                "manual_selections": [{"video_id": "A", "frame_idx": 10}],
                "target_rows": 1,
                "vqa_answer": "  câu trả lời của người dùng  ",
            },
            frame_lookup=self.store.frame,
        )
        self.assertTrue(valid["ok"])
        self.assertEqual(valid["mode"], "siglip")
        self.assertEqual(valid["query_id"], "query-1")
        self.assertEqual(valid["vqa_answer"], "câu trả lời của người dùng")
        self.assertEqual(valid["rows"][0]["query_id"], "query-1")
        self.assertEqual(valid["rows"][0]["mode"], "siglip")
        self.assertEqual(valid["rows"][0]["answer"], "câu trả lời của người dùng")
        self.assertFalse(valid["official_csv"]["has_header"])
        self.assertEqual(
            valid["rows"][0]["csv_line"],
            "A,10,câu trả lời của người dùng",
        )

    def test_vqa_enforces_organizer_answer_limit_and_quotes_special_characters(self) -> None:
        too_long = prepare_submission(
            {
                "task_type": "QA",
                "manual_selections": [{"video_id": "A", "frame_idx": 10}],
                "target_rows": 1,
                "vqa_answer": "x" * 101,
            },
            frame_lookup=self.store.frame,
        )
        self.assertFalse(too_long["ok"])
        self.assertIn("at most 100 characters", too_long["errors"][0])

        quoted = build_submission(
            "QA",
            manual_items=[{"video_id": "A", "frame_idx": 10}],
            candidate_items=[],
            frame_lookup=self.store.frame,
            target_rows=1,
            vqa_answer='Có 3 người, nói "Xin chào"',
        )
        self.assertEqual(quoted["task_type"], "VQA")
        self.assertEqual(
            quoted["rows"][0]["csv_line"],
            'A,10,"Có 3 người, nói ""Xin chào"""',
        )
        self.assertEqual(quoted["official_csv"]["content"], quoted["rows"][0]["csv_line"])

    def test_trake_uses_manual_then_candidates_then_canonical_temporal_fill(self) -> None:
        result = prepare_submission(
            {
                "task_type": "TRAKE",
                "mode": "ordered_siglip",
                "query_id": "trake-1",
                "event_count": 2,
                "target_rows": 3,
                "manual_sequences": [
                    [
                        {"video_id": "A", "frame_idx": 10},
                        {"video_id": "A", "frame_idx": 30},
                    ]
                ],
                "candidate_sequences": [
                    {
                        "sequence_score": 0.83,
                        "rank": 4,
                        "matched_events": [
                            {"video_id": "A", "frame_idx": 20},
                            {"video_id": "A", "frame_idx": 40},
                        ]
                    },
                    [
                        {"video_id": "A", "frame_idx": 20},
                        {"video_id": "B", "frame_idx": 40},
                    ],
                ],
            },
            frame_lookup=self.store.frame,
            video_frames_lookup=self.store.video,
        )
        self.assertTrue(result["ok"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["row_count"], 3)
        self.assertEqual(result["missing_rows"], 0)
        self.assertEqual(result["auto_filled_row_count"], 1)
        self.assertEqual(result["rows"][0]["matched_frames"], [10, 30])
        self.assertEqual(result["rows"][1]["matched_frames"], [20, 40])
        self.assertEqual(result["rows"][1]["sequence_score"], 0.83)
        self.assertEqual(result["rows"][1]["rank"], 4)
        self.assertEqual(
            result["rows"][2]["selection_origin"],
            "canonical_temporal_neighbor",
        )
        self.assertTrue(all(len(row["events"]) == 2 for row in result["rows"]))
        self.assertTrue(all(row["video_id"] == "A" for row in result["rows"]))
        self.assertTrue(
            all(
                row["matched_frames"][0] < row["matched_frames"][1]
                for row in result["rows"]
            )
        )
        self.assertIn("same video", result["warnings"][0])
        self.assertEqual(result["rows"][0]["csv_line"], "A,10,30")
        self.assertFalse(result["official_csv"]["has_header"])

    def test_trake_deterministically_fills_100_unique_canonical_sequences(self) -> None:
        payload = {
            "task_type": "TRAKE",
            "event_count": 3,
            "target_rows": 100,
            "manual_sequences": [
                [
                    {"video_id": "A", "frame_idx": 300},
                    {"video_id": "A", "frame_idx": 600},
                    {"video_id": "A", "frame_idx": 900},
                ]
            ],
            "candidate_sequences": [
                [
                    {"video_id": "A", "frame_idx": 310},
                    {"video_id": "A", "frame_idx": 610},
                    {"video_id": "A", "frame_idx": 910},
                ]
            ],
        }
        first = prepare_submission(
            payload,
            frame_lookup=self.store.frame,
            video_frames_lookup=self.store.video,
        )
        second = prepare_submission(
            payload,
            frame_lookup=self.store.frame,
            video_frames_lookup=self.store.video,
        )

        self.assertTrue(first["ok"])
        self.assertTrue(first["complete"])
        self.assertEqual(first["row_count"], 100)
        self.assertEqual(first["rows"][0]["matched_frames"], [300, 600, 900])
        self.assertEqual(first["rows"][1]["matched_frames"], [310, 610, 910])
        identities = {
            (row["video_id"], tuple(row["matched_frames"])) for row in first["rows"]
        }
        self.assertEqual(len(identities), 100)
        for row in first["rows"]:
            self.assertEqual(len(row["matched_frames"]), 3)
            self.assertEqual(row["matched_frames"], sorted(row["matched_frames"]))
            self.assertTrue(
                all(self.store.frame(row["video_id"], frame_idx) for frame_idx in row["matched_frames"])
            )
        self.assertEqual(
            [row["matched_frames"] for row in first["rows"]],
            [row["matched_frames"] for row in second["rows"]],
        )
        self.assertEqual(len(first["official_csv"]["content"].split("\r\n")), 100)

    def test_trake_stops_without_duplicates_when_canonical_timeline_is_too_small(self) -> None:
        result = build_submission(
            "TRAKE",
            manual_items=[
                [
                    {"video_id": "B", "frame_idx": 10},
                    {"video_id": "B", "frame_idx": 50},
                ]
            ],
            candidate_items=[],
            frame_lookup=self.store.frame,
            video_frames_lookup=self.store.video,
            event_count=2,
            target_rows=100,
        )
        identities = [tuple(row["matched_frames"]) for row in result["rows"]]
        self.assertFalse(result["complete"])
        self.assertTrue(result["valid_for_download"])
        self.assertTrue(result["official_csv"]["valid"])
        self.assertLess(result["row_count"], 100)
        self.assertEqual(len(identities), len(set(identities)))
        self.assertTrue(any("No duplicate" in warning for warning in result["warnings"]))

    def test_trake_normalizes_complete_event_order_labels_before_time_validation(self) -> None:
        result = build_submission(
            "TRAKE",
            manual_items=[
                [
                    {"video_id": "A", "frame_idx": 30, "event_order": 2},
                    {"video_id": "A", "frame_idx": 10, "event_order": 1},
                ]
            ],
            candidate_items=[],
            frame_lookup=self.store.frame,
            event_count=2,
            target_rows=1,
        )
        self.assertEqual(result["rows"][0]["matched_frames"], [10, 30])
        self.assertEqual(
            [event["event_order"] for event in result["rows"][0]["events"]],
            [1, 2],
        )

    def test_trake_rejects_partial_duplicate_or_gapped_event_order_labels(self) -> None:
        invalid_sequences = [
            [
                {"video_id": "A", "frame_idx": 10, "event_order": 1},
                {"video_id": "A", "frame_idx": 30},
            ],
            [
                {"video_id": "A", "frame_idx": 10, "event_order": 1},
                {"video_id": "A", "frame_idx": 30, "event_order": 1},
            ],
            [
                {"video_id": "A", "frame_idx": 10, "event_order": 1},
                {"video_id": "A", "frame_idx": 30, "event_order": 3},
            ],
        ]
        for sequence in invalid_sequences:
            with self.subTest(sequence=sequence):
                result = prepare_submission(
                    {
                        "task_type": "TRAKE",
                        "event_count": 2,
                        "target_rows": 1,
                        "manual_sequences": [sequence],
                    },
                    frame_lookup=self.store.frame,
                )
                self.assertFalse(result["ok"])
                self.assertFalse(result["valid_for_download"])
                self.assertFalse(result["official_csv"]["valid"])
                self.assertTrue(
                    "event_order" in result["errors"][0],
                    result["errors"],
                )

    def test_all_submission_types_reject_more_than_100_rows(self) -> None:
        with self.assertRaisesRegex(SubmissionValidationError, "from 1 to 100"):
            build_submission(
                "KIS",
                manual_items=[],
                candidate_items=[],
                frame_lookup=self.store.frame,
                target_rows=101,
            )

    def test_trake_rejects_non_monotonic_manual_sequence(self) -> None:
        result = prepare_submission(
            {
                "task_type": "TRAKE",
                "event_count": 2,
                "manual_sequences": [
                    [
                        {"video_id": "A", "frame_idx": 30},
                        {"video_id": "A", "frame_idx": 20},
                    ]
                ],
            },
            frame_lookup=self.store.frame,
        )
        self.assertFalse(result["ok"])
        self.assertIn("strictly increasing", result["errors"][0])
        self.assertEqual(result["rows"], [])


if __name__ == "__main__":
    unittest.main()
