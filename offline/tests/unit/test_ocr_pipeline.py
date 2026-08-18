from __future__ import annotations

from pathlib import Path

from aic2026.common import iter_jsonl, write_jsonl_atomic
from aic2026.contracts import FrameRef, OcrFrameRecord, OcrText
from aic2026.ocr import extract_ocr_frames
from PIL import Image


def _line(text: str = "SỐ 15") -> OcrText:
    return OcrText(
        line_id="line-0000",
        raw_text=text,
        normalized_text=text.lower(),
        confidence=0.9,
        confidence_semantics="engine_native_score",
        accepted=True,
        polygon_raw_xy=[(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)],
        polygon_xy=[(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)],
        normalized_polygon_xy=[(0.0, 0.0), (2 / 3, 0.0), (2 / 3, 0.625), (0.0, 0.625)],
        source_order=0,
        reading_order=0,
    )


class FakeReader:
    def extract(self, image: Image.Image, *, image_path: Path | None = None) -> list[OcrText]:
        assert image.size == (16, 9)
        if image_path and image_path.name == "2.jpg":
            raise RuntimeError("fixture inference failure")
        return [_line()]


def _ref(number: int) -> FrameRef:
    return FrameRef(
        video_id="L21_V011",
        frame_uid=f"L21_V011:{24924 + number}",
        keyframe_n=261 + number,
        frame_idx=24924 + number,
        pts_time_s=996.0 + number,
        fps=25.0,
        frame_relpath=f"keyframes/L21_V011/{number}.jpg",
        width=16,
        height=9,
    )


def test_ocr_emits_one_terminal_record_per_frame(tmp_path: Path) -> None:
    frame_dir = tmp_path / "keyframes" / "L21_V011"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (16, 9)).save(frame_dir / "1.jpg")
    Image.new("RGB", (16, 9)).save(frame_dir / "2.jpg")
    manifest = tmp_path / "frames.jsonl"
    write_jsonl_atomic(manifest, [_ref(1), _ref(2)])
    output = tmp_path / "ocr.jsonl"

    counts = extract_ocr_frames(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=output,
        run_id="ocr-test",
        reader=FakeReader(),
    )

    records = [OcrFrameRecord.model_validate(row) for row in iter_jsonl(output)]
    assert counts == {
        "frames": 2,
        "success": 1,
        "empty": 0,
        "error": 1,
        "spans": 1,
        "accepted_spans": 1,
        "rejected_spans": 0,
        "frames_without_text": 1,
    }
    assert [record.terminal_status for record in records] == ["success", "error"]
    assert records[0].full_text == "số 15"
    assert records[1].error is not None
    assert records[1].error.code == "ocr_inference_error"


class EmptyReader:
    def extract(self, image: Image.Image, *, image_path: Path | None = None) -> list[OcrText]:
        return []


def test_ocr_empty_is_explicit_terminal_record(tmp_path: Path) -> None:
    frame_dir = tmp_path / "keyframes" / "L21_V011"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (16, 9)).save(frame_dir / "1.jpg")
    manifest = tmp_path / "frames.jsonl"
    write_jsonl_atomic(manifest, [_ref(1)])
    output = tmp_path / "ocr.jsonl"

    extract_ocr_frames(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=output,
        run_id="ocr-test",
        reader=EmptyReader(),
    )

    record = OcrFrameRecord.model_validate(next(iter_jsonl(output)))
    assert record.terminal_status == "empty"
    assert record.texts == []
    assert record.full_text == ""
