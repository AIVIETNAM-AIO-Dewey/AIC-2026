"""Unit tests for Stage 3 OCR extraction and Stage 4 multi-modal metadata fusion."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from aic2026.contracts import FrameRef
from aic2026.ocr.ocr_backend import OcrResult, OcrSpan
from scripts.build_unified_frame_metadata import main as fusion_main
from scripts.run_ocr_extraction import main as ocr_main


def test_ocr_extraction_and_manifest(tmp_path: Path) -> None:
    # 1. Create a mock keyframe image
    img_dir = tmp_path / "keyframes" / "L21_V001"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "00000411.jpg"
    Image.new("RGB", (1280, 720), color="white").save(img_path)

    # 2. Create frame manifest
    manifest_path = tmp_path / "frame_manifests" / "L21_V001.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame_ref = FrameRef(
        video_id="L21_V001",
        frame_uid="L21_V001:411",
        keyframe_n=1,
        frame_idx=411,
        pts_time_s=13.7,
        fps=30.0,
        frame_relpath="keyframes/L21_V001/00000411.jpg",
        width=1280,
        height=720,
    )
    manifest_path.write_text(json.dumps(frame_ref.model_dump()) + "\n", encoding="utf-8")

    # 3. Mock OCR reader
    mock_span = OcrSpan(
        line_id="line-0000",
        raw_text="HTV9 HD 06:30:24",
        normalized_text="HTV9 HD 06:30:24",
        confidence=0.95,
        polygon_xy=[(800.0, 50.0), (1000.0, 50.0), (1000.0, 90.0), (800.0, 90.0)],
        normalized_polygon_xy=[(0.625, 0.069), (0.781, 0.069), (0.781, 0.125), (0.625, 0.125)],
        source_order=0,
        reading_order=0,
    )
    mock_ocr_result = OcrResult(full_text="HTV9 HD 06:30:24", spans=[mock_span])

    mock_reader = MagicMock()
    mock_reader.extract.return_value = mock_ocr_result

    output_path = tmp_path / "ocr" / "transcripts" / "L21_V001.jsonl"

    with patch("aic2026.ocr.OcrReader.create", return_value=mock_reader):
        code = ocr_main(
            [
                "--video-id", "L21_V001",
                "--frame-manifest", str(manifest_path),
                "--data-root", str(tmp_path),
                "--output", str(output_path),
            ]
        )
        assert code == 0

    assert output_path.is_file()
    with output_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 1
    assert records[0]["video_id"] == "L21_V001"
    assert records[0]["full_text"] == "HTV9 HD 06:30:24"
    assert len(records[0]["spans"]) == 1


def test_build_unified_frame_metadata_fusion(tmp_path: Path) -> None:
    # 1. Create Frame Manifest
    manifest_path = tmp_path / "frame_manifests" / "L21_V001.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame_ref = FrameRef(
        video_id="L21_V001",
        frame_uid="L21_V001:411",
        keyframe_n=1,
        frame_idx=411,
        pts_time_s=13.7,
        fps=30.0,
        frame_relpath="keyframes/L21_V001/00000411.jpg",
        width=1280,
        height=720,
    )
    manifest_path.write_text(json.dumps(frame_ref.model_dump()) + "\n", encoding="utf-8")

    # 2. Create DAM description
    desc_path = tmp_path / "object_description" / "descriptions" / "L21_V001.jsonl"
    desc_path.parent.mkdir(parents=True, exist_ok=True)
    desc_entry = {
        "video_id": "L21_V001",
        "frame_uid": "L21_V001:411",
        "keyframe_n": 1,
        "frame_idx": 411,
        "regions": [
            {
                "region_id": "L21_V001:411:01",
                "detector": {"class_entity": "road", "score": 0.85},
                "bbox_xyxy_px": [100, 150, 800, 500],
                "caption": {"description_en": "A cracked asphalt road next to a canal."},
            }
        ],
    }
    desc_path.write_text(json.dumps(desc_entry) + "\n", encoding="utf-8")

    # 3. Create OCR transcript
    ocr_path = tmp_path / "ocr" / "transcripts" / "L21_V001.jsonl"
    ocr_path.parent.mkdir(parents=True, exist_ok=True)
    ocr_entry = {
        "video_id": "L21_V001",
        "frame_uid": "L21_V001:411",
        "full_text": "TIN NÓNG 24H",
        "spans": [{"raw_text": "TIN NÓNG 24H", "normalized_text": "TIN NÓNG 24H", "confidence": 0.98}],
    }
    ocr_path.write_text(json.dumps(ocr_entry) + "\n", encoding="utf-8")

    # 4. Run Fusion
    unified_path = tmp_path / "unified_metadata" / "L21_V001.jsonl"
    code = fusion_main(
        [
            "--video-id", "L21_V001",
            "--frame-manifest", str(manifest_path),
            "--descriptions", str(desc_path),
            "--ocr-transcripts", str(ocr_path),
            "--output", str(unified_path),
        ]
    )
    assert code == 0
    assert unified_path.is_file()

    with unified_path.open("r", encoding="utf-8") as f:
        unified_rows = [json.loads(line) for line in f]
    assert len(unified_rows) == 1
    row = unified_rows[0]

    assert row["video_id"] == "L21_V001"
    assert row["frame_uid"] == "L21_V001:411"
    assert row["dam_summary_en"] == "A cracked asphalt road next to a canal."
    assert row["num_objects"] == 1
    assert row["ocr_text"] == "TIN NÓNG 24H"
    assert row["image_relpath"] == "keyframes/L21_V001/00000411.jpg"
