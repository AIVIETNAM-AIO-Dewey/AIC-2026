"""OCR artifact generation shared by EasyOCR and pinned PP-OCRv6."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Protocol

from PIL import Image

from aic2026.common import iter_jsonl, sha256_file
from aic2026.contracts import FrameRef, OcrError, OcrFrameRecord
from aic2026.contracts.ocr import OcrText


class OcrReader(Protocol):
    def extract(self, image: Image.Image, *, image_path: Path | None = None) -> list[OcrText]: ...


class SourceImageIdentityError(RuntimeError):
    """The current source bytes no longer match the frame manifest."""


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
    resume: bool = False,
) -> dict[str, int]:
    """Write one durable terminal record per frame and resume only a valid prefix."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    refs = [FrameRef.model_validate(raw) for raw in iter_jsonl(frame_manifest)]
    if limit is not None:
        refs = refs[:limit]
    if not refs:
        raise ValueError("frame manifest is empty")
    if any(ref.keyframe_n is None for ref in refs):
        raise ValueError("OCR requires organizer keyframes, not dense frames")
    if any(ref.source_image_sha256 is None for ref in refs):
        raise ValueError("OCR frame manifest must include source_image_sha256 for every frame")

    root = data_root.resolve()
    paths: list[Path] = []
    for ref in refs:
        path = (root / ref.frame_relpath).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Frame path escapes data root: {path}") from error
        paths.append(path)

    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists():
        if partial.exists():
            raise ValueError("Both final and partial OCR artifacts exist")
        records = [OcrFrameRecord.model_validate(raw) for raw in iter_jsonl(output)]
        _validate_resume_prefix(records, refs, run_id=run_id)
        if len(records) != len(refs):
            raise ValueError("Final OCR artifact is not complete")
        return _count_records(records)

    completed: list[OcrFrameRecord] = []
    if partial.exists():
        if not resume:
            raise FileExistsError(f"Partial OCR artifact exists; use --resume: {partial}")
        completed = [OcrFrameRecord.model_validate(raw) for raw in iter_jsonl(partial)]
        _validate_resume_prefix(completed, refs, run_id=run_id)
        for ref, path in zip(refs[: len(completed)], paths[: len(completed)], strict=True):
            try:
                actual_sha256 = sha256_file(path)
            except OSError as error:
                message = f"Completed OCR source image is unavailable: {ref.frame_uid}"
                raise ValueError(message) from error
            if actual_sha256 != ref.source_image_sha256:
                raise ValueError(f"Completed OCR source image checksum drift: {ref.frame_uid}")

    counters = _count_records(completed)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if completed else "w"
    with partial.open(mode, encoding="utf-8", newline="\n") as stream:
        for ref, path in zip(refs[len(completed) :], paths[len(completed) :], strict=True):
            record = _extract_frame(ref, path=path, run_id=run_id, reader=reader)
            stream.write(
                json.dumps(
                    record.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
                )
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            _increment_counters(counters, record)

    os.replace(partial, output)
    directory_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return counters


def _extract_frame(ref: FrameRef, *, path: Path, run_id: str, reader: OcrReader) -> OcrFrameRecord:
    terminal_status = "error"
    full_text = ""
    texts: list[OcrText] = []
    record_error: OcrError | None = None
    try:
        if sha256_file(path) != ref.source_image_sha256:
            raise SourceImageIdentityError("source image checksum differs from manifest")
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
    except Exception as error:  # one bad frame remains an explicit terminal record
        terminal_status = "error"
        texts = []
        record_error = OcrError(
            code=(
                "source_image_error"
                if isinstance(error, OSError)
                else "source_image_identity_error"
                if isinstance(error, SourceImageIdentityError)
                else "ocr_inference_error"
            ),
            message=f"{type(error).__name__}: {_safe_error(error)}",
        )
    return OcrFrameRecord(
        **ref.model_dump(),
        run_id=run_id,
        terminal_status=terminal_status,
        full_text=full_text,
        texts=texts,
        error=record_error,
    )


def _validate_resume_prefix(
    records: list[OcrFrameRecord], refs: list[FrameRef], *, run_id: str
) -> None:
    if len(records) > len(refs):
        raise ValueError("Partial OCR artifact contains extra frames")
    for index, (record, ref) in enumerate(zip(records, refs, strict=False)):
        if record.run_id != run_id:
            raise ValueError(f"Partial OCR run_id mismatch at record {index}")
        expected = ref.model_dump()
        actual = {key: getattr(record, key) for key in expected}
        if actual != expected:
            raise ValueError(f"Partial OCR frame identity mismatch at record {index}")


def _increment_counters(counters: dict[str, int], record: OcrFrameRecord) -> None:
    texts = record.texts
    counters["frames"] += 1
    counters[record.terminal_status] += 1
    counters["spans"] += len(texts)
    counters["accepted_spans"] += sum(line.accepted for line in texts)
    counters["rejected_spans"] += sum(not line.accepted for line in texts)
    counters["frames_without_text"] += int(record.terminal_status != "success")


def _count_records(records: list[OcrFrameRecord]) -> dict[str, int]:
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
    for record in records:
        _increment_counters(counters, record)
    return counters
