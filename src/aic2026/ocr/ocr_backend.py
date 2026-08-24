"""Multi-backend OCR Engine supporting EasyOCR and PaddleOCR with Vietnamese text normalization."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .text_normalizer import normalize_vietnamese_text


@dataclass(frozen=True, slots=True)
class OcrSpan:
    line_id: str
    raw_text: str
    normalized_text: str
    confidence: float
    polygon_xy: list[tuple[float, float]]
    normalized_polygon_xy: list[tuple[float, float]]
    source_order: int
    reading_order: int


@dataclass(frozen=True, slots=True)
class OcrResult:
    full_text: str
    spans: list[OcrSpan] = field(default_factory=list)


class OcrReader:
    """Unified OCR interface."""

    def __init__(self, backend_type: str, reader: Any, threshold: float = 0.30) -> None:
        self.backend_type = backend_type
        self.reader = reader
        self.threshold = threshold

    @classmethod
    def create(
        cls,
        backend: str = "auto",
        device: str = "cuda",
        threshold: float = 0.30,
        languages: list[str] | None = None,
    ) -> OcrReader:
        languages = languages or ["vi", "en"]
        use_gpu = device.startswith("cuda")

        if backend in ("auto", "easyocr"):
            try:
                import easyocr

                reader = easyocr.Reader(languages, gpu=use_gpu)
                return cls("easyocr", reader, threshold=threshold)
            except ImportError:
                if backend == "easyocr":
                    raise

        if backend in ("auto", "paddleocr"):
            try:
                os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
                from paddleocr import PaddleOCR

                reader = PaddleOCR(use_angle_cls=False, lang="vi", use_gpu=use_gpu, show_log=False)
                return cls("paddleocr", reader, threshold=threshold)
            except ImportError:
                if backend == "paddleocr":
                    raise

        raise RuntimeError("No OCR backend is available. Please install easyocr (pip install easyocr) or paddleocr.")

    def extract(self, image: Image.Image, image_path: Path | None = None) -> OcrResult:
        width, height = image.size
        spans: list[OcrSpan] = []

        if self.backend_type == "easyocr":
            # EasyOCR can accept numpy array directly
            img_np = np.array(image.convert("RGB"))
            raw_results = self.reader.readtext(img_np, detail=1, paragraph=False)
            for idx, (polygon, text, conf) in enumerate(raw_results):
                raw_text = str(text).strip()
                normalized = normalize_vietnamese_text(raw_text)
                if not normalized or float(conf) < self.threshold:
                    continue

                poly_pts = [(float(pt[0]), float(pt[1])) for pt in polygon]
                clamped_pts = [
                    (min(max(x, 0.0), width - 1.0), min(max(y, 0.0), height - 1.0))
                    for x, y in poly_pts
                ]
                norm_pts = [(x / max(1.0, width - 1.0), y / max(1.0, height - 1.0)) for x, y in clamped_pts]

                spans.append(
                    OcrSpan(
                        line_id=f"line-{idx:04d}",
                        raw_text=raw_text,
                        normalized_text=normalized,
                        confidence=float(conf),
                        polygon_xy=clamped_pts,
                        normalized_polygon_xy=norm_pts,
                        source_order=idx,
                        reading_order=idx,
                    )
                )

        elif self.backend_type == "paddleocr":
            path_str = str(image_path) if image_path else ""
            if not path_str or not Path(path_str).exists():
                img_np = np.array(image.convert("RGB"))
                raw_results = self.reader.ocr(img_np, cls=False)
            else:
                raw_results = self.reader.ocr(path_str, cls=False)

            if raw_results and raw_results[0]:
                for idx, line in enumerate(raw_results[0]):
                    polygon, (text, conf) = line
                    raw_text = str(text).strip()
                    normalized = normalize_vietnamese_text(raw_text)
                    if not normalized or float(conf) < self.threshold:
                        continue

                    poly_pts = [(float(pt[0]), float(pt[1])) for pt in polygon]
                    clamped_pts = [
                        (min(max(x, 0.0), width - 1.0), min(max(y, 0.0), height - 1.0))
                        for x, y in poly_pts
                    ]
                    norm_pts = [(x / max(1.0, width - 1.0), y / max(1.0, height - 1.0)) for x, y in clamped_pts]

                    spans.append(
                        OcrSpan(
                            line_id=f"line-{idx:04d}",
                            raw_text=raw_text,
                            normalized_text=normalized,
                            confidence=float(conf),
                            polygon_xy=clamped_pts,
                            normalized_polygon_xy=norm_pts,
                            source_order=idx,
                            reading_order=idx,
                        )
                    )

        # Sort spans by reading order (top-to-bottom, left-to-right)
        spans_sorted = sorted(
            spans,
            key=lambda s: (s.normalized_polygon_xy[0][1] if s.normalized_polygon_xy else 0.0, s.normalized_polygon_xy[0][0] if s.normalized_polygon_xy else 0.0),
        )
        full_text = " ".join(s.normalized_text for s in spans_sorted)
        return OcrResult(full_text=full_text, spans=spans_sorted)
