"""OCR artifact generation shared by EasyOCR and pinned PP-OCRv6."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Protocol

from PIL import Image

from aic2026.common import iter_jsonl, write_jsonl_atomic
from aic2026.contracts import FrameRef, OcrError, OcrFrameRecord
from aic2026.contracts.ocr import OcrText


class OcrReader(Protocol):
    def extract(self, image: Image.Image, *, image_path: Path | None = None) -> list[OcrText]: ...


def normalize_vietnamese_text(text: str) -> str:
    canonical = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", canonical.strip()).lower()


def _safe_error(error: BaseException) -> str:
    value = " ".join(str(error).replace("\r", " ").replace("\n", " ").split())
    return (value or type(error).__name__)[:500]


class EasyOcrReader:
    """Compatibility reader retained for existing EasyOCR runs."""

    def __init__(self, reader: object) -> None:
        self.reader = reader

    @classmethod
    def create(cls, *, gpu: bool) -> EasyOcrReader:
        try:
            import easyocr
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Install the easyocr profile before running OCR") from error
        return cls(easyocr.Reader(["vi", "en"], gpu=gpu, verbose=False))

    def extract(self, image: Image.Image, *, image_path: Path | None = None) -> list[OcrText]:
        del image_path
        rows = self.reader.readtext(image, detail=1, paragraph=False)
        texts: list[OcrText] = []
        for index, (polygon, text, confidence) in enumerate(rows):
            raw_text = str(text)
            normalized_text = normalize_vietnamese_text(raw_text)
            if not normalized_text:
                continue
            raw_points = [(float(point[0]), float(point[1])) for point in polygon]
            points = [
                (
                    min(max(x, 0.0), image.width - 1.0),
                    min(max(y, 0.0), image.height - 1.0),
                )
                for x, y in raw_points
            ]
            normalized = [(x / (image.width - 1), y / (image.height - 1)) for x, y in points]
            texts.append(
                OcrText(
                    line_id=f"line-{index:04d}",
                    raw_text=raw_text,
                    normalized_text=normalized_text,
                    confidence=float(confidence),
                    confidence_semantics="engine_native_score",
                    accepted=True,
                    polygon_raw_xy=raw_points,
                    polygon_xy=points,
                    normalized_polygon_xy=normalized,
                    polygon_clamped=points != raw_points,
                    source_order=index,
                    reading_order=index,
                )
            )
        return texts


def extract_ocr_frames(
    *,
    frame_manifest: Path,
    data_root: Path,
    output: Path,
    run_id: str,
    reader: OcrReader,
    limit: int | None = None,
) -> dict[str, int]:
    """Write one schema-valid terminal record for every selected frame."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    refs = [FrameRef.model_validate(raw) for raw in iter_jsonl(frame_manifest)]
    if limit is not None:
        refs = refs[:limit]
    if not refs:
        raise ValueError("frame manifest is empty")
    if any(ref.keyframe_n is None for ref in refs):
        raise ValueError("OCR requires organizer keyframes, not dense frames")

    root = data_root.resolve()
    records: list[OcrFrameRecord] = []
    counters = {
        "frames": 0,
        "success": 0,
        "empty": 0,
        "error": 0,
        "spans": 0,
        "accepted_spans": 0,
        "rejected_spans": 0,
        "frames_without_text": 0,
    }
    for ref in refs:
        path = (root / ref.frame_relpath).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Frame path escapes data root: {path}") from error

        terminal_status = "error"
        full_text = ""
        texts: list[OcrText] = []
        record_error: OcrError | None = None
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
            if image.size != (ref.width, ref.height):
                raise ValueError(
                    f"image dimensions {image.size} differ from manifest {(ref.width, ref.height)}"
                )
            texts = reader.extract(image, image_path=path)
            accepted = sorted(
                (line for line in texts if line.accepted), key=lambda line: line.reading_order
            )
            if accepted:
                terminal_status = "success"
                full_text = " ".join(line.normalized_text for line in accepted)
            else:
                terminal_status = "empty"
                texts = []
        except Exception as error:  # one bad frame must not erase coverage for the run
            terminal_status = "error"
            texts = []
            record_error = OcrError(
                code="source_image_error" if isinstance(error, OSError) else "ocr_inference_error",
                message=f"{type(error).__name__}: {_safe_error(error)}",
            )

        record = OcrFrameRecord(
            **ref.model_dump(),
            run_id=run_id,
            terminal_status=terminal_status,
            full_text=full_text,
            texts=texts,
            error=record_error,
        )
        records.append(record)
        counters["frames"] += 1
        counters[terminal_status] += 1
        counters["spans"] += len(texts)
        counters["accepted_spans"] += sum(line.accepted for line in texts)
        counters["rejected_spans"] += sum(not line.accepted for line in texts)
        counters["frames_without_text"] += int(terminal_status != "success")

    write_jsonl_atomic(output, records)
    return counters
