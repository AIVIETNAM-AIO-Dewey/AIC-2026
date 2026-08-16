"""Small EasyOCR adapter; model import is delayed until an offline run."""

from __future__ import annotations

import re

from PIL import Image

from aic2026.contracts.ocr import OcrText


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
