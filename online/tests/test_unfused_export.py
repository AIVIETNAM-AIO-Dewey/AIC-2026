"""Unit tests for Un-Fused Multi-Channel 100-Image Exporter (Dry-Run)."""

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from online.scripts.export_unfused_channels import run_unfused_multi_channel_export, StandaloneVectorSearchEngine


class TestUnfusedExportDryRun(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_standalone_search_channels_return_100_items(self):
        searcher = StandaloneVectorSearchEngine(is_dry_run=True)
        import numpy as np

        # Test SigLIP2
        siglip_res = searcher.search_siglip2(np.random.randn(768), top_k=100)
        self.assertEqual(len(siglip_res), 100)
        self.assertEqual(siglip_res[0]["rank"], 1)
        self.assertTrue(siglip_res[0]["score"] >= siglip_res[-1]["score"])

        # Test DAM
        dam_res = searcher.search_dam([np.random.randn(1024)], ["car"], top_k=100)
        self.assertEqual(len(dam_res), 100)
        self.assertEqual(dam_res[0]["rank"], 1)

        # Test ASR
        asr_res = searcher.search_asr(np.random.randn(1024), top_k=100)
        self.assertEqual(len(asr_res), 100)

        # Test OCR
        ocr_res = searcher.search_ocr(["vietnam", "thoi"], top_k=100)
        self.assertEqual(len(ocr_res), 100)

    def test_full_unfused_export_dryrun(self):
        report = run_unfused_multi_channel_export(
            query_text="Người lái xe ô tô màu đỏ",
            output_base_dir=self.temp_dir,
            dry_run=True,
            top_k=100,
        )

        channels = ["siglip2", "dam", "asr", "ocr"]
        for ch in channels:
            self.assertIn(ch, report["channels"])
            ch_data = report["channels"][ch]
            self.assertEqual(ch_data["count"], 100)
            self.assertEqual(ch_data["images_exported"], 100)

            # Validate CSV file exists and has exactly 100 rows
            csv_path = Path(ch_data["csv_path"])
            self.assertTrue(csv_path.exists())
            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
                self.assertEqual(len(rows), 100)
                # Check format: VIDEO_ID, FRAME_IDX
                self.assertTrue(rows[0][0].startswith("L01_V"))
                self.assertTrue(int(rows[0][1]) >= 0)

            # Validate 100 images exported in folder
            images_dir = Path(self.temp_dir) / "unfused_export" / ch / "top100_images"
            self.assertTrue(images_dir.exists())
            exported_images = list(images_dir.glob("*.jpg"))
            self.assertEqual(len(exported_images), 100)


if __name__ == "__main__":
    unittest.main()
