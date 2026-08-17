from __future__ import annotations

from pathlib import Path

from aic2026.common import iter_jsonl, write_jsonl_atomic
from aic2026.contracts import FrameRef, OcrFrameRecord, OcrText
from aic2026.ocr import extract_ocr_frames
from PIL import Image


class FakeReader:
    def extract(self, image: Image.Image) -> list[OcrText]:
        assert image.size == (16, 9)
        return [
            OcrText(
                raw_text="SỐ 15",
                normalized_text="số 15",
                confidence=0.9,
                polygon_xy=[(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)],
            )
        ]


def test_ocr_uses_canonical_frame_identity(tmp_path: Path) -> None:
    frame_path = tmp_path / "keyframes" / "L21_V011" / "262.jpg"
    frame_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 9)).save(frame_path)
    manifest = tmp_path / "frames.jsonl"
    write_jsonl_atomic(
        manifest,
        [
            FrameRef(
                video_id="L21_V011",
                frame_uid="L21_V011:24925",
                keyframe_n=262,
                frame_idx=24925,
                pts_time_s=997.0,
                fps=25.0,
                frame_relpath="keyframes/L21_V011/262.jpg",
                width=16,
                height=9,
            )
        ],
    )
    output = tmp_path / "ocr.jsonl"
    counts = extract_ocr_frames(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=output,
        run_id="ocr-test",
        reader=FakeReader(),
    )
    record = OcrFrameRecord.model_validate(next(iter_jsonl(output)))
    assert counts == {"frames": 1, "spans": 1, "frames_without_text": 0}
    assert (record.video_id, record.frame_idx) == ("L21_V011", 24925)
