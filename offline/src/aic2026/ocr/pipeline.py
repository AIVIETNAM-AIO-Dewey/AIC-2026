"""Small EasyOCR adapter; model import is delayed until an offline run."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from PIL import Image

from aic2026.common import iter_jsonl, write_jsonl_atomic
from aic2026.contracts import FrameRef, OcrFrameRecord
from aic2026.contracts.ocr import OcrText


class OcrReader(Protocol):
    def extract(self, image: Image.Image) -> list[OcrText]: ...


def normalize_vietnamese_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


class EasyOcrReader:
    def __init__(self, reader: object) -> None:
        self.reader = reader

    @classmethod
    def create(cls, *, gpu: bool) -> EasyOcrReader:
        try:
            import easyocr
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Install the easyocr profile before running OCR") from error
        return cls(easyocr.Reader(["vi", "en"], gpu=gpu, verbose=False))

    def extract(self, image: Image.Image) -> list[OcrText]:
        rows = self.reader.readtext(image, detail=1, paragraph=False)
        return [
            OcrText(
                raw_text=str(text),
                normalized_text=normalize_vietnamese_text(str(text)),
                confidence=float(confidence),
                polygon_xy=[(float(point[0]), float(point[1])) for point in polygon],
            )
            for polygon, text, confidence in rows
            if str(text).strip()
        ]


def extract_ocr_frames(
    *,
    frame_manifest: Path,
    data_root: Path,
    output: Path,
    run_id: str,
    reader: OcrReader,
    limit: int | None = None,
) -> dict[str, int]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    refs = [FrameRef.model_validate(raw) for raw in iter_jsonl(frame_manifest)]
    if limit is not None:
        refs = refs[:limit]
    if not refs:
        raise ValueError("frame manifest is empty")
    root = data_root.resolve()
    records: list[OcrFrameRecord] = []
    spans = 0
    frames_without_text = 0
    for ref in refs:
        if ref.keyframe_n is None:
            raise ValueError("OCR requires organizer keyframes, not dense frames")
        path = (root / ref.frame_relpath).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Frame path escapes data root: {path}") from error
        with Image.open(path) as source:
            image = source.convert("RGB")
        texts = reader.extract(image)
        spans += len(texts)
        frames_without_text += int(not texts)
        records.append(OcrFrameRecord(**ref.model_dump(), run_id=run_id, texts=texts))
    write_jsonl_atomic(output, records)
    return {"frames": len(records), "spans": spans, "frames_without_text": frames_without_text}
