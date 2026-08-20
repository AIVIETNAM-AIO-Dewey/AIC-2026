"""Unit tests for official AIC 2026 Submission formatting and ZIP packaging."""

import unittest
import tempfile
import zipfile
from pathlib import Path

from online.src.submission.export_submission import (
    clean_answer_for_csv,
    format_kis_row,
    format_qa_row,
    format_trake_row,
    export_query_csv,
    package_submission_zip,
)
from online.src.contracts.query import SearchResponse, SearchResult


class TestSubmissionFormat(unittest.TestCase):

    def test_kis_row_format(self):
        row = format_kis_row("L01_V028.mp4", 25300)
        self.assertEqual(row, "L01_V028, 25300")

    def test_qa_row_format(self):
        # Plain answer
        row1 = format_qa_row("L01_V028", 3450, "5")
        self.assertEqual(row1, 'L01_V028, 3450, "5"')

        # Answer with comma and quotes
        row2 = format_qa_row("L04_V012", 4100, 'Anh ấy nói "Tuyệt vời"')
        self.assertEqual(row2, 'L04_V012, 4100, "Anh ấy nói ""Tuyệt vời"""')

        # Length truncation test
        long_ans = "a" * 150
        row3 = format_qa_row("L01_V001", 100, long_ans)
        self.assertTrue(len(clean_answer_for_csv(long_ans)) <= 100)

    def test_trake_row_format(self):
        row = format_trake_row("L10_V001", [1200, 1850, 2100, 2450])
        self.assertEqual(row, "L10_V001, 1200, 1850, 2100, 2450")

    def test_export_and_packaging_zip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create mock response
            resp = SearchResponse(
                task_type="KIS",
                original_query="test query",
                results=[
                    SearchResult(rank=1, video_id="L00_V000", keyframe_n=1, frame_idx=1234),
                    SearchResult(rank=2, video_id="L00_V055", keyframe_n=5, frame_idx=5555),
                    SearchResult(rank=3, video_id="L01_V028", keyframe_n=25, frame_idx=25300),
                ]
            )

            csv_file = tmp_path / "query-1-kis.csv"
            export_query_csv(csv_file, resp)

            # Check CSV content (No header, exact lines)
            content = csv_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(content), 3)
            self.assertEqual(content[0], "L00_V000, 1234")
            self.assertEqual(content[1], "L00_V055, 5555")
            self.assertEqual(content[2], "L01_V028, 25300")

            # Check ZIP packaging
            zip_file = tmp_path / "submission.zip"
            package_submission_zip([csv_file], zip_file)
            self.assertTrue(zip_file.exists())

            with zipfile.ZipFile(zip_file, "r") as zf:
                namelist = zf.namelist()
                self.assertIn("submission/query-1-kis.csv", namelist)


if __name__ == "__main__":
    unittest.main()
