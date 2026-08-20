"""Small, model-agnostic evaluation loop for local OCR recognizers.

This module is intentionally independent from OCR Phase 1.  It consumes immutable
crop manifests, records one terminal result per crop, and supports prefix resume.
Optional model packages are imported only by their adapters.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import quote

from PIL import Image
from pydantic import Field, model_validator

from aic2026.common import iter_jsonl, sha256_file, write_jsonl_atomic
from aic2026.contracts.models import StrictModel
from aic2026.contracts.paths import require_safe_relative_path

SHA256_PATTERN = r"^[0-9a-f]{64}$"


def normalize_transcript(text: str) -> str:
    """Normalize serialization noise without changing spelling or letter case."""

    return " ".join(unicodedata.normalize("NFC", text).split())


class RecognitionEvalSample(StrictModel):
    schema_version: Literal["aic26.ocr_recognition_eval_sample.v1"] = (
        "aic26.ocr_recognition_eval_sample.v1"
    )
    sample_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    crop_relpath: str = Field(min_length=1)
    crop_sha256: str = Field(pattern=SHA256_PATTERN)
    reference_transcript_nfc: str = Field(min_length=1)

    @model_validator(mode="after")
    def canonical_fields(self) -> RecognitionEvalSample:
        require_safe_relative_path(self.crop_relpath, field_name="crop_relpath")
        if self.reference_transcript_nfc != normalize_transcript(self.reference_transcript_nfc):
            raise ValueError("reference transcript must be trimmed, whitespace-canonical NFC")
        return self


class RecognitionPrediction(StrictModel):
    transcript_raw: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class LocalRecognitionResult(StrictModel):
    schema_version: Literal["aic26.ocr_local_recognition_result.v1"] = (
        "aic26.ocr_local_recognition_result.v1"
    )
    sample_id: str = Field(min_length=1)
    crop_sha256: str = Field(pattern=SHA256_PATTERN)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    status: Literal["ok", "empty", "error"]
    transcript_raw: str | None = None
    transcript_nfc: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    latency_ms: float = Field(ge=0)
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def status_payload(self) -> LocalRecognitionResult:
        if not math.isfinite(self.latency_ms):
            raise ValueError("latency must be finite")
        if self.status == "ok":
            if not self.transcript_raw or not self.transcript_nfc:
                raise ValueError("successful recognition requires non-empty text")
            if self.transcript_nfc != normalize_transcript(self.transcript_raw):
                raise ValueError("normalized transcript differs from raw transcript")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful recognition cannot contain an error")
        elif self.status == "empty":
            if self.transcript_raw != "" or self.transcript_nfc != "":
                raise ValueError("empty recognition must contain canonical empty strings")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("empty recognition cannot contain an error")
        else:
            if self.transcript_raw is not None or self.transcript_nfc is not None:
                raise ValueError("failed recognition cannot contain text")
            if not self.error_type or not self.error_message:
                raise ValueError("failed recognition requires structured error details")
            if self.confidence is not None:
                raise ValueError("failed recognition cannot claim confidence")
        return self


class CropRecognizer(Protocol):
    model_id: str
    model_revision: str

    def predict(self, image: Image.Image) -> RecognitionPrediction: ...


class BatchCropRecognizer(CropRecognizer, Protocol):
    def predict_batch(self, images: Sequence[Image.Image]) -> list[RecognitionPrediction]: ...


class VietOcrRecognizer:
    """Lazy adapter around the official ``vietocr`` Predictor API."""

    model_id = "pbcquoc/vietocr-vgg-seq2seq"

    def __init__(self, predictor: object, *, model_revision: str) -> None:
        self._predictor = predictor
        self.model_revision = model_revision

    @classmethod
    def create(
        cls,
        *,
        config_path: Path,
        weights_path: Path,
        device: str,
        expected_weights_sha256: str,
    ) -> VietOcrRecognizer:
        if sha256_file(weights_path) != expected_weights_sha256:
            raise ValueError("VietOCR weights SHA-256 does not match the expected revision")
        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as error:  # pragma: no cover - optional runtime profile
            raise RuntimeError("Install offline/requirements/vietocr.txt first") from error
        config = Cfg.load_config_from_file(str(config_path))
        config["weights"] = str(weights_path)
        config["device"] = device
        cnn_config = config.get("cnn")
        if not isinstance(cnn_config, dict):
            raise ValueError("VietOCR config must contain a cnn mapping")
        # vietocr 0.3.13 defaults this argument to True, which makes torchvision
        # download VGG ImageNet weights before Predictor strictly loads our full
        # local OCR checkpoint.  The initialization is both unused and hostile to
        # deterministic/offline inference, so always disable it here.
        cnn_config["pretrained"] = False
        config.setdefault("predictor", {})["beamsearch"] = False
        return cls(Predictor(config), model_revision=expected_weights_sha256)

    def predict(self, image: Image.Image) -> RecognitionPrediction:
        raw, probability = self._predictor.predict(image, return_prob=True)  # type: ignore[attr-defined]
        confidence = _normalize_confidence(probability)
        return RecognitionPrediction(transcript_raw=str(raw), confidence=confidence)

    def predict_batch(self, images: Sequence[Image.Image]) -> list[RecognitionPrediction]:
        """Run the official VietOCR 0.3.13 width-bucketed batch API."""

        if not images:
            return []
        raw_texts, probabilities = self._predictor.predict_batch(  # type: ignore[attr-defined]
            list(images), return_prob=True
        )
        if len(raw_texts) != len(images) or len(probabilities) != len(images):
            raise RuntimeError("VietOCR batch output length differs from its input")
        return [
            RecognitionPrediction(
                transcript_raw=str(raw),
                confidence=_normalize_confidence(probability),
            )
            for raw, probability in zip(raw_texts, probabilities, strict=True)
        ]


def _normalize_confidence(probability: object) -> float | None:
    if probability is None:
        return None
    if hasattr(probability, "item"):
        probability = probability.item()  # type: ignore[union-attr]
    return min(max(float(probability), 0.0), 1.0)


def export_l23_verified_evaluation(
    *, state_db: Path, annotation_root: Path, output: Path
) -> dict[str, int | str]:
    """Export verified L23 labels without mutating the annotation state store."""

    if output.exists() or output.with_suffix(output.suffix + ".tmp").exists():
        raise FileExistsError(f"fresh evaluation output is required: {output}")
    state_db = state_db.resolve()
    annotation_root = annotation_root.resolve()
    uri = f"file:{quote(str(state_db))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            """
            SELECT b.annotation_id, b.source_id, b.crop_relpath, b.crop_sha256,
                   d.transcript_nfc
            FROM base_annotations AS b
            JOIN decisions AS d USING(annotation_id)
            WHERE d.status = 'verified'
            ORDER BY b.annotation_id
            """
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError("annotation state contains no verified records")

    samples: list[RecognitionEvalSample] = []
    seen: set[str] = set()
    for annotation_id, source_id, crop_relpath, crop_sha256, transcript_nfc in rows:
        sample = RecognitionEvalSample(
            sample_id=annotation_id,
            video_id=source_id,
            crop_relpath=crop_relpath,
            crop_sha256=crop_sha256,
            reference_transcript_nfc=transcript_nfc,
        )
        if sample.sample_id in seen:
            raise ValueError("verified annotation IDs must be unique")
        seen.add(sample.sample_id)
        crop_path = (annotation_root / sample.crop_relpath).resolve()
        try:
            crop_path.relative_to(annotation_root)
        except ValueError as error:
            raise ValueError("evaluation crop path escapes annotation root") from error
        if not crop_path.is_file() or sha256_file(crop_path) != sample.crop_sha256:
            raise ValueError(f"evaluation crop identity mismatch: {sample.sample_id}")
        with Image.open(crop_path) as image:
            image.verify()
        samples.append(sample)
    write_jsonl_atomic(output, samples)
    return {
        "samples": len(samples),
        "videos": len({sample.video_id for sample in samples}),
        "manifest_sha256": sha256_file(output),
    }


def _load_eval_samples(manifest: Path, crop_root: Path) -> list[RecognitionEvalSample]:
    samples = [RecognitionEvalSample.model_validate(row) for row in iter_jsonl(manifest)]
    if not samples:
        raise ValueError("recognition evaluation manifest is empty")
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("recognition evaluation sample IDs must be unique")
    root = crop_root.resolve()
    for sample in samples:
        path = (root / sample.crop_relpath).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("recognition crop path escapes crop root") from error
        if not path.is_file() or sha256_file(path) != sample.crop_sha256:
            raise ValueError(f"recognition crop identity mismatch: {sample.sample_id}")
    return samples


def _load_result_prefix(
    path: Path,
    samples: list[RecognitionEvalSample],
    *,
    model_id: str,
    model_revision: str,
) -> list[LocalRecognitionResult]:
    records = [LocalRecognitionResult.model_validate(row) for row in iter_jsonl(path)]
    if len(records) > len(samples):
        raise ValueError("recognition result contains extra records")
    for index, (record, sample) in enumerate(zip(records, samples, strict=False)):
        if (
            record.sample_id != sample.sample_id
            or record.crop_sha256 != sample.crop_sha256
            or record.model_id != model_id
            or record.model_revision != model_revision
        ):
            raise ValueError(f"recognition resume identity mismatch at record {index}")
    return records


def run_local_recognition_evaluation(
    *,
    manifest: Path,
    crop_root: Path,
    output: Path,
    recognizer: CropRecognizer,
    resume: bool = False,
) -> dict[str, int]:
    """Run one recognizer with durable per-record progress and exact prefix resume."""

    manifest_sha256 = sha256_file(manifest)
    samples = _load_eval_samples(manifest, crop_root)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists():
        if partial.exists():
            raise ValueError("both final and partial recognition outputs exist")
        records = _load_result_prefix(
            output,
            samples,
            model_id=recognizer.model_id,
            model_revision=recognizer.model_revision,
        )
        if len(records) != len(samples):
            raise ValueError("final recognition output is incomplete")
        return _result_counts(records)

    completed: list[LocalRecognitionResult] = []
    if partial.exists():
        if not resume:
            raise FileExistsError(f"partial recognition output exists; use --resume: {partial}")
        completed = _load_result_prefix(
            partial,
            samples,
            model_id=recognizer.model_id,
            model_revision=recognizer.model_revision,
        )
    elif resume:
        raise FileNotFoundError("cannot resume recognition without a partial output")

    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if completed else "w"
    root = crop_root.resolve()
    with partial.open(mode, encoding="utf-8", newline="\n") as stream:
        for sample in samples[len(completed) :]:
            path = (root / sample.crop_relpath).resolve()
            if sha256_file(path) != sample.crop_sha256:
                raise ValueError(f"recognition crop changed before inference: {sample.sample_id}")
            started = time.perf_counter()
            try:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                prediction = recognizer.predict(image)
                transcript_nfc = normalize_transcript(prediction.transcript_raw)
                status: Literal["ok", "empty", "error"] = "ok" if transcript_nfc else "empty"
                record = LocalRecognitionResult(
                    sample_id=sample.sample_id,
                    crop_sha256=sample.crop_sha256,
                    model_id=recognizer.model_id,
                    model_revision=recognizer.model_revision,
                    status=status,
                    transcript_raw=prediction.transcript_raw if status == "ok" else "",
                    transcript_nfc=transcript_nfc,
                    confidence=prediction.confidence,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            except Exception as error:
                message = " ".join(str(error).replace("\r", " ").replace("\n", " ").split())
                record = LocalRecognitionResult(
                    sample_id=sample.sample_id,
                    crop_sha256=sample.crop_sha256,
                    model_id=recognizer.model_id,
                    model_revision=recognizer.model_revision,
                    status="error",
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error_type=type(error).__name__,
                    error_message=(message or type(error).__name__)[:500],
                )
            stream.write(
                json.dumps(
                    record.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
            completed.append(record)
    os.replace(partial, output)
    directory_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if sha256_file(manifest) != manifest_sha256:
        raise ValueError("evaluation manifest changed during recognition")
    return _result_counts(completed)


def _result_counts(records: list[LocalRecognitionResult]) -> dict[str, int]:
    return {
        "records": len(records),
        "ok": sum(record.status == "ok" for record in records),
        "empty": sum(record.status == "empty" for record in records),
        "error": sum(record.status == "error" for record in records),
    }


def _edit_alignment(reference: str, prediction: str) -> tuple[int, list[tuple[str, str | None]]]:
    matrix = [[0] * (len(prediction) + 1) for _ in range(len(reference) + 1)]
    for row in range(len(reference) + 1):
        matrix[row][0] = row
    for column in range(len(prediction) + 1):
        matrix[0][column] = column
    for row, reference_char in enumerate(reference, start=1):
        for column, prediction_char in enumerate(prediction, start=1):
            matrix[row][column] = min(
                matrix[row - 1][column] + 1,
                matrix[row][column - 1] + 1,
                matrix[row - 1][column - 1] + (reference_char != prediction_char),
            )

    pairs: list[tuple[str, str | None]] = []
    row, column = len(reference), len(prediction)
    while row or column:
        if row and column:
            cost = reference[row - 1] != prediction[column - 1]
            if matrix[row][column] == matrix[row - 1][column - 1] + cost:
                pairs.append((reference[row - 1], prediction[column - 1]))
                row -= 1
                column -= 1
                continue
        if row and matrix[row][column] == matrix[row - 1][column] + 1:
            pairs.append((reference[row - 1], None))
            row -= 1
            continue
        column -= 1  # insertion: it consumes no reference character
    pairs.reverse()
    return matrix[-1][-1], pairs


VIETNAMESE_DIACRITICS = frozenset(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "ÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)


def evaluate_local_recognition(
    *, manifest: Path, results: Path, minimum_exact_match: float = 0.0
) -> dict[str, object]:
    samples = [RecognitionEvalSample.model_validate(row) for row in iter_jsonl(manifest)]
    records = [LocalRecognitionResult.model_validate(row) for row in iter_jsonl(results)]
    if len(samples) != len(records):
        raise ValueError("recognition evaluation requires one result per sample")
    total_edits = 0
    total_reference_characters = 0
    exact = 0
    casefold_exact = 0
    diacritic_total = 0
    diacritic_matches = 0
    for index, (sample, record) in enumerate(zip(samples, records, strict=True)):
        if sample.sample_id != record.sample_id or sample.crop_sha256 != record.crop_sha256:
            raise ValueError(f"recognition evaluation identity mismatch at record {index}")
        prediction = record.transcript_nfc or ""
        reference = sample.reference_transcript_nfc
        exact += prediction == reference
        casefold_exact += prediction.casefold() == reference.casefold()
        distance, aligned_pairs = _edit_alignment(reference, prediction)
        total_edits += distance
        total_reference_characters += len(reference)
        for character, predicted_character in aligned_pairs:
            if character in VIETNAMESE_DIACRITICS:
                diacritic_total += 1
                diacritic_matches += predicted_character == character
    character_error_rate = total_edits / total_reference_characters
    exact_match = exact / len(samples)
    successful_records = sum(record.status == "ok" for record in records)
    report: dict[str, object] = {
        "schema_version": "aic26.ocr_local_recognition_evaluation.v1",
        "samples": len(samples),
        "model_id": records[0].model_id,
        "model_revision": records[0].model_revision,
        "manifest_sha256": sha256_file(manifest),
        "results_sha256": sha256_file(results),
        "exact_match": exact_match,
        "casefold_exact_match": casefold_exact / len(samples),
        "character_error_rate": character_error_rate,
        "character_accuracy": max(0.0, 1.0 - character_error_rate),
        "vietnamese_diacritic_recall_conservative": (
            diacritic_matches / diacritic_total if diacritic_total else None
        ),
        "errors": sum(record.status == "error" for record in records),
        "empty": sum(record.status == "empty" for record in records),
        "successful_records": successful_records,
        "minimum_exact_match": minimum_exact_match,
        "passed": exact_match >= minimum_exact_match
        and successful_records > 0
        and all(record.status != "error" for record in records),
    }
    return report
