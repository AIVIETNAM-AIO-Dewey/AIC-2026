"""Strict quality, execution-attestation, and negative-fixture contracts for OCR Phase 1."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from aic2026.common import sha256_file
from aic2026.contracts import FrameRef, OcrDetectionFrameRecord, QuadGeometry
from aic2026.contracts.models import StrictModel

from .frame_snapshot import CanonicalFrameError, decode_canonical_frame
from .tracking import natural_key, polygon_iou

InstanceStratum = Literal[
    "positive_text",
    "horizontal",
    "perspective",
    "clipped_edge",
    "near_vertical",
]
FrameStratum = Literal["multi_box"]
INSTANCE_STRATA: frozenset[str] = frozenset(
    {"positive_text", "horizontal", "perspective", "clipped_edge", "near_vertical"}
)
FRAME_STRATA: frozenset[str] = frozenset({"multi_box"})
REQUIRED_STRATA: frozenset[str] = frozenset({*INSTANCE_STRATA, *FRAME_STRATA})


@dataclass(frozen=True, slots=True)
class DetectionQualityConfig:
    version: str = "aic26.ocr_detection_quality.v1"
    minimum_labeled_frames: int = 100
    minimum_non_ignored_instances: int = 200
    minimum_frames_per_stratum: int = 15
    matching_iou_threshold: float = 0.50
    minimum_overall_recall: float = 0.95
    minimum_stratum_recall: float = 0.90
    minimum_overall_precision: float = 0.50

    def __post_init__(self) -> None:
        integer_values = (
            self.minimum_labeled_frames,
            self.minimum_non_ignored_instances,
            self.minimum_frames_per_stratum,
        )
        float_values = (
            self.matching_iou_threshold,
            self.minimum_overall_recall,
            self.minimum_stratum_recall,
            self.minimum_overall_precision,
        )
        if any(type(value) is not int for value in integer_values) or any(
            type(value) is not float for value in float_values
        ):
            raise ValueError("OCR detection quality policy requires exact JSON numeric types")
        expected = {
            "version": "aic26.ocr_detection_quality.v1",
            "minimum_labeled_frames": 100,
            "minimum_non_ignored_instances": 200,
            "minimum_frames_per_stratum": 15,
            "matching_iou_threshold": 0.50,
            "minimum_overall_recall": 0.95,
            "minimum_stratum_recall": 0.90,
            "minimum_overall_precision": 0.50,
        }
        if asdict(self) != expected:
            raise ValueError("OCR detection quality policy differs from locked v1 thresholds")

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GroundTruthInstance(StrictModel):
    instance_id: str = Field(min_length=1)
    polygon_xy: QuadGeometry
    ignore: bool
    strata: tuple[InstanceStratum, ...]

    @field_validator("strata", mode="before")
    @classmethod
    def accept_json_strata(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_instance_strata(self) -> GroundTruthInstance:
        if len(self.strata) != len(set(self.strata)):
            raise ValueError("duplicate stratum in ground-truth instance")
        if self.ignore and self.strata:
            raise ValueError("ignored ground-truth instances cannot contribute strata")
        if not self.ignore and "positive_text" not in self.strata:
            raise ValueError("non-ignored ground-truth instances require positive_text")
        return self


class GroundTruthFrame(StrictModel):
    frame_uid: str = Field(min_length=3)
    strata: tuple[FrameStratum, ...]
    instances: tuple[GroundTruthInstance, ...]

    @field_validator("strata", "instances", mode="before")
    @classmethod
    def accept_json_arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def unique_local_identity(self) -> GroundTruthFrame:
        if len(self.strata) != len(set(self.strata)):
            raise ValueError("duplicate stratum in ground-truth frame")
        instance_ids = [item.instance_id for item in self.instances]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("duplicate instance_id in ground-truth frame")
        non_ignored = sum(not item.ignore for item in self.instances)
        if ("multi_box" in self.strata) is not (non_ignored >= 2):
            raise ValueError("multi_box frame stratum must exactly reflect two or more instances")
        return self


class ExecutionAttestation(StrictModel):
    schema_version: Literal["aic26.ocr_phase1.execution_attestation.v1"]
    provider: Literal["kaggle"]
    notebook_kernel_identifier: str = Field(min_length=1)
    notebook_version_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    detector_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    detector_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_runtime_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    internet_enabled: Literal[False]
    accelerator_device: Literal["cpu"]
    created_at: datetime
    approver: str = Field(min_length=1)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_timezone(self) -> ExecutionAttestation:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("execution attestation created_at must include a timezone")
        payload = json.dumps(
            self.model_dump(mode="json", exclude={"payload_sha256"}),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != self.payload_sha256:
            raise ValueError("execution attestation payload checksum mismatch")
        return self


NegativeErrorCode = Literal[
    "source_checksum_drift",
    "unsupported_source_mode",
    "corrupt_source_image",
    "canonical_dimension_mismatch",
]


class NegativeFixture(FrameRef):
    fixture_id: str = Field(min_length=1)
    expected_error_code: NegativeErrorCode
    expected_reason: str = Field(min_length=1)


class NegativeSuiteReceipt(StrictModel):
    schema_version: Literal["aic26.ocr_phase1.negative_suite_receipt.v1"]
    status: Literal["completed"]
    negative_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_count: int = Field(ge=1)
    results_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _manifest_snapshot(path: Path, model: type[StrictModel]) -> tuple[str, list[Any]]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"manifest is unavailable: {path}") from error
    digest = hashlib.sha256(payload).hexdigest()
    records: list[Any] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"invalid manifest record at line {line_number}: {path}") from error
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return digest, records


def load_ground_truth_manifest(
    path: Path,
    frames_by_uid: Mapping[str, FrameRef],
    *,
    config: DetectionQualityConfig,
) -> tuple[str, list[GroundTruthFrame]]:
    digest, raw_records = _manifest_snapshot(path, GroundTruthFrame)
    records = list(raw_records)
    frame_uids = [item.frame_uid for item in records]
    if len(frame_uids) != len(set(frame_uids)):
        raise ValueError("duplicate frame_uid in ground-truth manifest")
    if len(records) < config.minimum_labeled_frames:
        raise ValueError("ground-truth subset has fewer than 100 fully labeled frames")

    global_instance_ids: set[str] = set()
    non_ignored_instances = 0
    frames_per_stratum = {name: set() for name in REQUIRED_STRATA}
    for record in records:
        frame = frames_by_uid.get(record.frame_uid)
        if frame is None:
            raise ValueError(f"unknown ground-truth frame_uid: {record.frame_uid}")
        for name in record.strata:
            frames_per_stratum[name].add(record.frame_uid)
        for instance in record.instances:
            if instance.instance_id in global_instance_ids:
                raise ValueError(f"duplicate global instance_id: {instance.instance_id}")
            global_instance_ids.add(instance.instance_id)
            if not instance.ignore:
                non_ignored_instances += 1
                for name in instance.strata:
                    frames_per_stratum[name].add(record.frame_uid)
            for x, y in instance.polygon_xy.points:
                if not 0 <= x <= frame.width - 1 or not 0 <= y <= frame.height - 1:
                    raise ValueError(
                        f"ground-truth polygon is outside frame bounds: {instance.instance_id}"
                    )
    if non_ignored_instances < config.minimum_non_ignored_instances:
        raise ValueError("ground-truth subset has fewer than 200 non-ignored text instances")
    missing = {
        name: len(frame_uids_for_name)
        for name, frame_uids_for_name in frames_per_stratum.items()
        if len(frame_uids_for_name) < config.minimum_frames_per_stratum
    }
    if missing:
        raise ValueError(f"ground-truth strata coverage is below 15 frames: {missing}")
    if sha256_file(path) != digest:
        raise ValueError("ground-truth manifest changed during validation")
    return digest, sorted(records, key=lambda item: natural_key(item.frame_uid))


def _maximum_cardinality_iou_matching(
    predictions: Sequence[tuple[str, QuadGeometry]],
    ground_truth: Sequence[GroundTruthInstance],
    *,
    threshold: float,
) -> list[tuple[int, int, float]]:
    adjacency: dict[int, list[tuple[int, float]]] = {}
    for ground_index, instance in enumerate(ground_truth):
        candidates: list[tuple[int, float]] = []
        for prediction_index, (_prediction_id, polygon) in enumerate(predictions):
            overlap = polygon_iou(polygon.points, instance.polygon_xy.points)
            if overlap >= threshold:
                candidates.append((prediction_index, overlap))
        adjacency[ground_index] = sorted(
            candidates,
            key=lambda item: (
                -round(item[1] * 1_000_000_000_000),
                natural_key(predictions[item[0]][0]),
            ),
        )

    prediction_owner: dict[int, int] = {}

    def augment(ground_index: int, visited: set[int]) -> bool:
        for prediction_index, _overlap in adjacency[ground_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            owner = prediction_owner.get(prediction_index)
            if owner is None or augment(owner, visited):
                prediction_owner[prediction_index] = ground_index
                return True
        return False

    ordered_ground = sorted(
        range(len(ground_truth)), key=lambda index: natural_key(ground_truth[index].instance_id)
    )
    for ground_index in ordered_ground:
        augment(ground_index, set())
    matches = []
    for prediction_index, ground_index in prediction_owner.items():
        overlap = next(
            value for candidate, value in adjacency[ground_index] if candidate == prediction_index
        )
        matches.append((prediction_index, ground_index, overlap))
    return sorted(matches, key=lambda item: natural_key(ground_truth[item[1]].instance_id))


def _iou_histogram(values: Sequence[float]) -> dict[str, int]:
    labels = ("0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00", "1.00")
    histogram = {label: 0 for label in labels}
    for value in values:
        if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12):
            histogram["1.00"] += 1
        else:
            index = min(max(int((value - 0.5) * 10), 0), 4)
            histogram[labels[index]] += 1
    return histogram


def evaluate_detection_quality(
    records: Sequence[OcrDetectionFrameRecord],
    ground_truth: Sequence[GroundTruthFrame],
    *,
    config: DetectionQualityConfig,
) -> dict[str, Any]:
    records_by_uid = {record.frame_uid: record for record in records}
    if len(records_by_uid) != len(records):
        raise ValueError("duplicate frame_uid in prediction records")
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    ignored_predictions = 0
    matched_ious: list[float] = []
    strata_counts = {name: [0, 0] for name in REQUIRED_STRATA}

    for labeled_frame in ground_truth:
        prediction_record = records_by_uid.get(labeled_frame.frame_uid)
        if prediction_record is None:
            raise ValueError(f"prediction artifact lacks labeled frame: {labeled_frame.frame_uid}")
        predictions = [
            (detection.detection_id, detection.polygon_xy)
            for detection in sorted(
                prediction_record.detections,
                key=lambda item: (item.source_order, natural_key(item.detection_id)),
            )
        ]
        evaluated = [instance for instance in labeled_frame.instances if not instance.ignore]
        ignored = [instance for instance in labeled_frame.instances if instance.ignore]
        matches = _maximum_cardinality_iou_matching(
            predictions,
            evaluated,
            threshold=config.matching_iou_threshold,
        )
        matched_prediction_indices = {item[0] for item in matches}
        matched_ground_indices = {item[1] for item in matches}
        frame_tp = len(matches)
        frame_fn = len(evaluated) - frame_tp
        frame_ignored_predictions = 0
        for prediction_index, (_prediction_id, polygon) in enumerate(predictions):
            if prediction_index in matched_prediction_indices:
                continue
            if any(
                polygon_iou(polygon.points, instance.polygon_xy.points)
                >= config.matching_iou_threshold
                for instance in ignored
            ):
                frame_ignored_predictions += 1
        frame_fp = len(predictions) - frame_tp - frame_ignored_predictions
        true_positives += frame_tp
        false_positives += frame_fp
        false_negatives += frame_fn
        ignored_predictions += frame_ignored_predictions
        matched_ious.extend(item[2] for item in matches)
        for ground_index, instance in enumerate(evaluated):
            for name in instance.strata:
                strata_counts[name][0] += int(ground_index in matched_ground_indices)
                strata_counts[name][1] += 1
        # multi_box is intentionally frame-level: its recall is measured over
        # every non-ignored text instance in frames containing 2+ instances.
        for name in labeled_frame.strata:
            strata_counts[name][0] += frame_tp
            strata_counts[name][1] += len(evaluated)

    precision = true_positives / (true_positives + false_positives) if true_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    recall_by_stratum = {
        name: matched / total if total else 0.0
        for name, (matched, total) in sorted(strata_counts.items())
    }
    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "ignored_predictions": ignored_predictions,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "recall_by_stratum": recall_by_stratum,
        "matched_iou_histogram": _iou_histogram(matched_ious),
        "matched_iou_values": sorted(matched_ious),
        "labeled_frames": len(ground_truth),
        "ground_truth_instances": sum(
            not instance.ignore for frame in ground_truth for instance in frame.instances
        ),
        "ignored_ground_truth_instances": sum(
            instance.ignore for frame in ground_truth for instance in frame.instances
        ),
    }


def enforce_quality_thresholds(metrics: Mapping[str, Any], config: DetectionQualityConfig) -> None:
    if metrics["recall"] < config.minimum_overall_recall:
        raise ValueError("overall detector recall is below locked threshold")
    if metrics["precision"] < config.minimum_overall_precision:
        raise ValueError("overall detector precision is below locked threshold")
    below = {
        name: value
        for name, value in metrics["recall_by_stratum"].items()
        if value < config.minimum_stratum_recall
    }
    if below:
        raise ValueError(f"detector recall is below locked per-stratum threshold: {below}")


def load_and_verify_execution_attestation(
    path: Path,
    *,
    expected_config_sha256: str,
    expected_detector_revision: str,
    expected_detector_tree_sha256: str,
    expected_runtime_identity_sha256: str,
    expected_source_commit_sha: str,
) -> tuple[str, ExecutionAttestation]:
    if re.fullmatch(r"[0-9a-f]{40}", expected_source_commit_sha) is None:
        raise ValueError("expected source commit SHA must be exactly 40 lowercase hex characters")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError("execution attestation is unavailable") from error
    digest = hashlib.sha256(payload).hexdigest()
    try:
        attestation = ExecutionAttestation.model_validate_json(payload)
    except ValueError as error:
        raise ValueError("execution attestation has invalid strict schema") from error
    actual = (
        attestation.config_sha256,
        attestation.detector_revision,
        attestation.detector_tree_sha256,
        attestation.environment_runtime_identity_sha256,
        attestation.notebook_version_commit_sha,
    )
    expected = (
        expected_config_sha256,
        expected_detector_revision,
        expected_detector_tree_sha256,
        expected_runtime_identity_sha256,
        expected_source_commit_sha,
    )
    if actual != expected:
        raise ValueError("execution attestation identity/config/commit mismatch")
    if sha256_file(path) != digest:
        raise ValueError("execution attestation changed during verification")
    return digest, attestation


def verify_negative_fixture_suite(
    manifest_path: Path,
    data_root: Path,
    *,
    config_sha256: str,
) -> tuple[NegativeSuiteReceipt, tuple[tuple[Path, str], ...]]:
    manifest_hash, raw_fixtures = _manifest_snapshot(manifest_path, NegativeFixture)
    fixtures = list(raw_fixtures)
    fixture_ids = [fixture.fixture_id for fixture in fixtures]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("duplicate fixture_id in negative manifest")
    root = data_root.resolve()
    baseline: list[tuple[Path, str]] = [(manifest_path.resolve(), manifest_hash)]
    results: list[dict[str, str]] = []
    for fixture in sorted(fixtures, key=lambda item: natural_key(item.fixture_id)):
        source = (root / fixture.frame_relpath).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"negative fixture path escapes data root: {fixture.fixture_id}"
            ) from error
        baseline.append((source, sha256_file(source)))
        try:
            decode_canonical_frame(fixture, source)
        except CanonicalFrameError as error:
            if error.code != fixture.expected_error_code or str(error) != fixture.expected_reason:
                raise ValueError(
                    f"negative fixture rejection drift: {fixture.fixture_id}"
                ) from error
            results.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "error_code": error.code,
                    "reason": str(error),
                }
            )
        else:
            raise ValueError(f"negative fixture was unexpectedly accepted: {fixture.fixture_id}")
    results_payload = json.dumps(results, sort_keys=True, separators=(",", ":")).encode("utf-8")
    receipt = NegativeSuiteReceipt(
        schema_version="aic26.ocr_phase1.negative_suite_receipt.v1",
        status="completed",
        negative_manifest_sha256=manifest_hash,
        config_sha256=config_sha256,
        fixture_count=len(fixtures),
        results_sha256=hashlib.sha256(results_payload).hexdigest(),
    )
    for path, expected_hash in baseline:
        if sha256_file(path) != expected_hash:
            raise ValueError(f"negative fixture input changed during verification: {path}")
    return receipt, tuple(baseline)


def verify_negative_suite_receipt(path: Path, expected: NegativeSuiteReceipt) -> str:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError("negative-suite receipt is unavailable") from error
    digest = hashlib.sha256(payload).hexdigest()
    try:
        actual = NegativeSuiteReceipt.model_validate_json(payload)
    except ValueError as error:
        raise ValueError("negative-suite receipt has invalid strict schema") from error
    if actual != expected:
        raise ValueError("negative-suite receipt identity/results mismatch")
    if sha256_file(path) != digest:
        raise ValueError("negative-suite receipt changed during verification")
    return digest


def verify_file_unchanged(path: Path, expected_sha256: str, *, label: str) -> None:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} changed during real-model gate")
