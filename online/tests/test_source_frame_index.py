"""Boundary and identity tests for exact source-frame submission indexing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from online.src.submission.frame_index import SourceFrameIndex


class _Metadata:
    def __init__(self, frames: list[dict]) -> None:
        self.frames = frames

    def video_frames(self, video_id: str) -> tuple[dict, ...]:
        if video_id != "L25_V001":
            return ()
        return tuple(dict(frame) for frame in self.frames)


class SourceFrameIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        media_root = Path(self.temporary_directory.name)
        (media_root / "L25_V001.json").write_text(
            json.dumps({"length": 10}),
            encoding="utf-8",
        )
        self.frames = [
            {
                "video_id": "L25_V001",
                "keyframe_n": 1,
                "frame_idx": 17,
                "pts_time_s": 0.68,
                "fps": 25.0,
                "image_relpath": "keyframes/L25_V001/00000017.jpg",
            },
            {
                "video_id": "L25_V001",
                "keyframe_n": 2,
                "frame_idx": 51,
                "pts_time_s": 2.04,
                "fps": 25.0,
                "image_relpath": "keyframes/L25_V001/00000051.jpg",
            },
            {
                "video_id": "L25_V001",
                "keyframe_n": 3,
                "frame_idx": 87,
                "pts_time_s": 3.48,
                "fps": 25.0,
                "image_relpath": "keyframes/L25_V001/00000087.jpg",
            },
        ]
        self.index = SourceFrameIndex(_Metadata(self.frames), media_root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_timeline_declares_zero_based_bounds_without_keyframe_substitution(self) -> None:
        timeline = self.index.timeline("l25-v001")
        self.assertIsNotNone(timeline)
        assert timeline is not None
        self.assertEqual(timeline["frame_index_base"], 0)
        self.assertEqual(timeline["max_frame_idx"], 249)
        self.assertEqual(timeline["frame_count"], 250)
        self.assertEqual(timeline["keyframe_count"], 3)
        self.assertEqual(timeline["timing_method"], "exact-anchor-piecewise-linear-v1")

    def test_exact_index_preserves_organizer_identity_and_timestamp(self) -> None:
        frame = self.index.resolve("L25_V001", 51)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame["frame_idx"], 51)
        self.assertEqual(frame["frame_uid"], "L25_V001:51")
        self.assertEqual(frame["pts_time_s"], 2.04)
        self.assertEqual(frame["keyframe_n"], 2)
        self.assertTrue(frame["indexed_keyframe"])
        self.assertEqual(frame["validation"], "canonical")
        self.assertEqual(frame["preview_frame_idx"], 51)

    def test_arbitrary_index_is_not_replaced_by_its_preview_keyframe(self) -> None:
        frame = self.index.resolve("L25_V001", 34)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame["frame_idx"], 34)
        self.assertEqual(frame["frame_uid"], "L25_V001:34")
        self.assertAlmostEqual(frame["pts_time_s"], 1.36)
        self.assertIsNone(frame["keyframe_n"])
        self.assertFalse(frame["indexed_keyframe"])
        self.assertEqual(frame["validation"], "source_timeline")
        self.assertEqual(frame["image_relpath"], "")
        # 34 is exactly halfway between anchors 17 and 51; ties go earlier.
        self.assertEqual(frame["preview_frame_idx"], 17)
        self.assertEqual(frame["related_seed_frame_idx"], 17)

    def test_first_last_and_out_of_range_boundaries_are_unambiguous(self) -> None:
        first = self.index.resolve("L25_V001", 0)
        last = self.index.resolve("L25_V001", 249)
        self.assertEqual(first["frame_idx"], 0)
        self.assertEqual(first["pts_time_s"], 0.0)
        self.assertEqual(last["frame_idx"], 249)
        self.assertAlmostEqual(last["pts_time_s"], 9.96)
        self.assertIsNone(self.index.resolve("L25_V001", -1))
        self.assertIsNone(self.index.resolve("L25_V001", 250))
        self.assertIsNone(self.index.resolve("UNKNOWN", 0))

    def test_invalid_or_non_monotonic_organizer_timeline_fails_closed(self) -> None:
        malformed = [dict(self.frames[0]), {**self.frames[1], "frame_idx": 17}]
        index = SourceFrameIndex(_Metadata(malformed), Path(self.temporary_directory.name))
        with self.assertRaisesRegex(ValueError, "not strictly ordered"):
            index.timeline("L25_V001")


if __name__ == "__main__":
    unittest.main()
