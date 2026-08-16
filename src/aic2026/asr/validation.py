"""Post-hoc validation of ASR pipeline artifacts.

Reads ``<video_id>.jsonl`` files and validates every line against the
``aic26.asr_segments.v1`` Pydantic contract, plus cross-checks against
the companion ``.manifest.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from aic2026.contracts.asr import AsrSegmentRecord, AsrVideoManifest

logger = logging.getLogger(__name__)


class ValidationResult:
    """Collects validation results for a single JSONL artifact."""

    def __init__(self, video_id: str) -> None:
        self.video_id = video_id
        self.total_lines = 0
        self.valid_records = 0
        self.errors: list[dict] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0 and self.total_lines > 0

    def add_error(self, line_num: int, message: str) -> None:
        self.errors.append({"line": line_num, "error": message})

    def summary(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        return (
            f"[{status}] {self.video_id}: "
            f"{self.valid_records}/{self.total_lines} valid records, "
            f"{len(self.errors)} errors"
        )


def validate_jsonl(jsonl_path: str | Path) -> ValidationResult:
    """Validate all records in an ASR JSONL artifact.

    Parameters
    ----------
    jsonl_path:
        Path to a ``<video_id>.jsonl`` file.

    Returns
    -------
    ValidationResult
        Summary of validation with per-line errors.
    """
    jsonl_path = Path(jsonl_path)
    video_id = jsonl_path.stem
    result = ValidationResult(video_id)

    if not jsonl_path.exists():
        result.add_error(0, f"File not found: {jsonl_path}")
        return result

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            result.total_lines += 1
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                result.add_error(line_num, f"Invalid JSON: {exc}")
                continue

            try:
                record = AsrSegmentRecord.model_validate(data)
                result.valid_records += 1
            except ValidationError as exc:
                result.add_error(line_num, str(exc))
                continue

            # Cross-check: video_id consistency
            if record.video_id != video_id:
                result.add_error(
                    line_num,
                    f"video_id mismatch: record has '{record.video_id}' "
                    f"but file is '{video_id}.jsonl'",
                )

    return result


def validate_manifest(
    manifest_path: str | Path,
    jsonl_result: ValidationResult | None = None,
) -> list[str]:
    """Validate a manifest file and optionally cross-check with JSONL results.

    Returns a list of error messages (empty = valid).
    """
    manifest_path = Path(manifest_path)
    errors: list[str] = []

    if not manifest_path.exists():
        return [f"Manifest not found: {manifest_path}"]

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        manifest = AsrVideoManifest.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        return [f"Invalid manifest: {exc}"]

    # Cross-check with JSONL if provided
    if jsonl_result is not None:
        if manifest.segment_count != jsonl_result.valid_records:
            errors.append(
                f"Manifest claims {manifest.segment_count} segments "
                f"but JSONL has {jsonl_result.valid_records} valid records"
            )

    return errors
