"""Crash-safe Phase 1 representative recognition with a local OCR model.

The runner deliberately owns no detection, tracking, fallback, or consensus logic.
It consumes a single verified Phase 1 shard and emits one model result for every
representative crop, preserving the representative artifact's byte order.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

from PIL import Image
from pydantic import Field, model_validator

from aic2026.common import atomic_write_json, iter_jsonl, sha256_file
from aic2026.contracts import (
    CropProvenance,
    FrameRef,
    OcrPhase1Receipt,
    OcrTrajectoryRecord,
    RepresentativeCropBinding,
)
from aic2026.contracts.models import StrictModel
from aic2026.ocr.frame_snapshot import decode_canonical_frame
from aic2026.ocr.phase1 import Phase1Identity, receipt_path_for, verify_linked_artifacts
from aic2026.ocr.tracking import TrackingConfig

from .local_recognition import CropRecognizer, RecognitionPrediction

SHA256_PATTERN = r"^[0-9a-f]{64}$"
SOURCE_COMMIT_PATTERN = r"^[0-9a-f]{7,64}$"
FaultInjector = Callable[[str], None]
Clock = Callable[[], float]
DEFAULT_RECOGNITION_BATCH_SIZE = 32
DEFAULT_FRAME_CACHE_CAPACITY = 8
DEFAULT_FRAME_CACHE_MAX_BYTES = 256 * 1024 * 1024
MAX_RECOGNITION_BATCH_SIZE = 256
MAX_FRAME_CACHE_CAPACITY = 256
MAX_FRAME_CACHE_BYTES = 2 * 1024 * 1024 * 1024
_EXECUTION_POLICY_SCHEMA = "aic26.ocr_phase2_execution_policy.v1"


def _bounded_integer(name: str, value: int, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer inside [1, {maximum}]")
    return value


def recognition_execution_policy(
    *, batch_size: int, frame_cache_capacity: int, frame_cache_max_bytes: int
) -> dict[str, object]:
    """Return the exact Phase 2 batching/cache policy bound into every receipt."""

    return {
        "schema_version": _EXECUTION_POLICY_SCHEMA,
        "batch_size": _bounded_integer(
            "batch_size", batch_size, maximum=MAX_RECOGNITION_BATCH_SIZE
        ),
        "frame_cache_algorithm": "canonical_rgb_lru.v1",
        "frame_cache_capacity": _bounded_integer(
            "frame_cache_capacity",
            frame_cache_capacity,
            maximum=MAX_FRAME_CACHE_CAPACITY,
        ),
        "frame_cache_max_bytes": _bounded_integer(
            "frame_cache_max_bytes",
            frame_cache_max_bytes,
            maximum=MAX_FRAME_CACHE_BYTES,
        ),
        "crop_reconstruction": "exact_phase1_png_sha_plus_direct_rgb.v1",
        "batch_failure_policy": "write_nothing_before_complete_batch.v1",
        "output_order": "representative_artifact_byte_order.v1",
    }


def recognition_execution_policy_sha256(
    *, batch_size: int, frame_cache_capacity: int, frame_cache_max_bytes: int
) -> str:
    return _canonical_hash(
        recognition_execution_policy(
            batch_size=batch_size,
            frame_cache_capacity=frame_cache_capacity,
            frame_cache_max_bytes=frame_cache_max_bytes,
        )
    )


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_vocab_sha256(config_path: Path) -> str:
    """Hash the exact ordered vocabulary independently from the YAML bytes."""

    try:
        import yaml
    except ImportError as error:  # pragma: no cover - platform dependency
        raise RuntimeError("PyYAML is required to inspect the VietOCR config") from error
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("vocab"), str):
        raise ValueError("VietOCR config must contain a string vocab")
    return hashlib.sha256(payload["vocab"].encode("utf-8")).hexdigest()


def local_runtime_identity(device: str) -> tuple[dict[str, str], dict[str, str], str]:
    """Return explicit package and runtime evidence for a production receipt."""

    package_names = (
        "vietocr",
        "torch",
        "torchvision",
        "numpy",
        "PyYAML",
        "pydantic",
        "Pillow",
    )
    packages: dict[str, str] = {}
    for name in package_names:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required Phase 2 package is not installed: {name}") from error
    try:
        import torch
    except ImportError as error:  # pragma: no cover - required production dependency
        raise RuntimeError("PyTorch is required to attest the Phase 2 runtime") from error

    def boolean(value: object) -> str:
        return "true" if bool(value) else "false"

    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
        "deterministic_algorithms": boolean(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": boolean(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": boolean(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": boolean(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": boolean(torch.backends.cudnn.allow_tf32),
    }
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        device_index = (
            torch.cuda.current_device() if torch_device.index is None else torch_device.index
        )
        major, minor = torch.cuda.get_device_capability(device_index)
        runtime["cuda_device_index"] = str(device_index)
        runtime["cuda_device_name"] = torch.cuda.get_device_name(device_index)
        runtime["cuda_compute_capability"] = f"{major}.{minor}"
    else:
        runtime["cuda_device_index"] = "not_applicable"
        runtime["cuda_device_name"] = "not_applicable"
        runtime["cuda_compute_capability"] = "not_applicable"
    return packages, runtime, _canonical_hash({"packages": packages, "runtime": runtime})


class RepresentativeRecognitionResult(StrictModel):
    schema_version: Literal["aic26.ocr_representative_recognition.v1"] = (
        "aic26.ocr_representative_recognition.v1"
    )
    binding: RepresentativeCropBinding
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=SHA256_PATTERN)
    status: Literal["ok", "empty", "error"]
    transcript_raw: str | None = None
    transcript_nfc: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    latency_ms: float = Field(ge=0)
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> RepresentativeRecognitionResult:
        if not math.isfinite(self.latency_ms):
            raise ValueError("latency_ms must be finite")
        if self.status in {"ok", "empty"}:
            expected = "" if self.status == "empty" else None
            if self.transcript_raw is None or self.transcript_nfc is None:
                raise ValueError("successful inference requires raw and NFC transcripts")
            if self.transcript_nfc != unicodedata.normalize("NFC", self.transcript_raw):
                raise ValueError("transcript_nfc must be exact Unicode NFC of transcript_raw")
            if expected is not None and (
                self.transcript_raw != expected or self.transcript_nfc != expected
            ):
                raise ValueError("empty result requires canonical empty transcripts")
            if self.status == "ok" and not self.transcript_nfc:
                raise ValueError("ok result requires non-empty text")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful inference cannot contain an error")
        else:
            if self.transcript_raw is not None or self.transcript_nfc is not None:
                raise ValueError("error result cannot contain a transcript")
            if self.confidence is not None or not self.error_type or not self.error_message:
                raise ValueError("error result requires error details and no confidence")
        return self


class RepresentativeRecognitionReceipt(StrictModel):
    schema_version: Literal["aic26.ocr_representative_recognition_receipt.v2"] = (
        "aic26.ocr_representative_recognition_receipt.v2"
    )
    status: Literal["running", "completed"]
    run_id: str = Field(min_length=1)
    source_commit_sha: str = Field(pattern=SOURCE_COMMIT_PATTERN)
    phase1_config_sha256: str = Field(pattern=SHA256_PATTERN)
    phase1_tracking_identity_mode: Literal["current", "legacy_without_commit_interval"]
    phase1_tracking_config_sha256: str = Field(pattern=SHA256_PATTERN)
    phase1_resource_limits_sha256: str = Field(pattern=SHA256_PATTERN)
    frame_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    detections_sha256: str = Field(pattern=SHA256_PATTERN)
    trajectories_sha256: str = Field(pattern=SHA256_PATTERN)
    representatives_sha256: str = Field(pattern=SHA256_PATTERN)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=SHA256_PATTERN)
    model_weights_sha256: str = Field(pattern=SHA256_PATTERN)
    model_config_sha256: str = Field(pattern=SHA256_PATTERN)
    model_vocab_sha256: str = Field(pattern=SHA256_PATTERN)
    package_versions: dict[str, str] = Field(min_length=1)
    runtime: dict[str, str] = Field(min_length=1)
    runtime_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    total_representatives: int = Field(ge=0)
    commit_interval_records: int = Field(ge=1)
    batch_size: int = Field(ge=1, le=MAX_RECOGNITION_BATCH_SIZE)
    frame_cache_capacity: int = Field(ge=1, le=MAX_FRAME_CACHE_CAPACITY)
    frame_cache_max_bytes: int = Field(ge=1, le=MAX_FRAME_CACHE_BYTES)
    recognition_execution_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    committed_records: int = Field(ge=0)
    committed_bytes: int = Field(ge=0)
    committed_sha256: str = Field(pattern=SHA256_PATTERN)
    output_sha256: str = Field(pattern=SHA256_PATTERN)
    inference_batches: int = Field(ge=0)
    frame_cache_hits: int = Field(ge=0)
    frame_cache_misses: int = Field(ge=0)
    frame_cache_evictions: int = Field(ge=0)
    frame_cache_oversized: int = Field(ge=0)
    trust_boundary: Literal["integrity_metadata_not_a_signature"] = (
        "integrity_metadata_not_a_signature"
    )

    @model_validator(mode="after")
    def validate_commit(self) -> RepresentativeRecognitionReceipt:
        if self.model_revision != self.model_weights_sha256:
            raise ValueError("model revision must equal the pinned weights SHA-256")
        expected_runtime = _canonical_hash(
            {"packages": self.package_versions, "runtime": self.runtime}
        )
        if self.runtime_identity_sha256 != expected_runtime:
            raise ValueError("runtime identity SHA-256 is inconsistent")
        expected_policy = recognition_execution_policy_sha256(
            batch_size=self.batch_size,
            frame_cache_capacity=self.frame_cache_capacity,
            frame_cache_max_bytes=self.frame_cache_max_bytes,
        )
        if self.recognition_execution_policy_sha256 != expected_policy:
            raise ValueError("recognition execution policy SHA-256 is inconsistent")
        if self.committed_records > self.total_representatives:
            raise ValueError("receipt commits more records than the representative input")
        if self.output_sha256 != self.committed_sha256:
            raise ValueError("output and committed SHA-256 must agree")
        if self.status == "completed" and self.committed_records != self.total_representatives:
            raise ValueError("completed receipt must commit every representative")
        if self.frame_cache_hits + self.frame_cache_misses != self.committed_records:
            raise ValueError("cache hit/miss counts must cover every committed representative")
        if self.frame_cache_oversized > self.frame_cache_misses:
            raise ValueError("oversized cache misses cannot exceed total cache misses")
        if self.frame_cache_evictions > self.frame_cache_misses:
            raise ValueError("cache evictions cannot exceed total cache misses")
        if self.inference_batches > self.committed_records:
            raise ValueError("inference batch count cannot exceed committed records")
        return self


class RepresentativeInferenceError(RuntimeError):
    """A per-record model failure left the stage resumable, not completed."""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_receipt(path: Path, receipt: RepresentativeRecognitionReceipt) -> None:
    atomic_write_json(path, receipt.model_dump(mode="json"))
    _fsync_directory(path.parent)


def _record_bytes(record: RepresentativeRecognitionResult) -> bytes:
    return (
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _parse_prefix(
    payload: bytes,
    representatives: list[RepresentativeCropBinding],
    *,
    model_id: str,
    model_revision: str,
) -> list[RepresentativeRecognitionResult]:
    if payload and not payload.endswith(b"\n"):
        raise ValueError("committed recognition prefix is missing its final newline")
    records: list[RepresentativeRecognitionResult] = []
    for index, line in enumerate(payload.splitlines()):
        try:
            record = RepresentativeRecognitionResult.model_validate_json(line)
        except Exception as error:
            raise ValueError(f"invalid committed recognition record {index}") from error
        if index >= len(representatives) or record.binding != representatives[index]:
            raise ValueError(f"recognition binding/order mismatch at record {index}")
        if record.model_id != model_id or record.model_revision != model_revision:
            raise ValueError(f"recognition model identity mismatch at record {index}")
        if record.status == "error":
            raise ValueError("committed recognition prefix contains an inference error")
        if _record_bytes(record) != line + b"\n":
            raise ValueError(f"recognition record {index} is not canonically serialized")
        records.append(record)
    return records


def _identity_fields(
    *,
    run_id: str,
    source_commit_sha: str,
    phase1_config_sha256: str,
    phase1_tracking_identity_mode: str,
    phase1_tracking_config_sha256: str,
    phase1_resource_limits_sha256: str,
    input_hashes: Mapping[str, str],
    recognizer: CropRecognizer,
    model_weights_sha256: str,
    model_config_sha256: str,
    vocab_sha256: str,
    package_versions: Mapping[str, str],
    runtime: Mapping[str, str],
    runtime_identity_sha256: str,
    total_representatives: int,
    commit_interval_records: int,
    batch_size: int,
    frame_cache_capacity: int,
    frame_cache_max_bytes: int,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "source_commit_sha": source_commit_sha,
        "phase1_config_sha256": phase1_config_sha256,
        "phase1_tracking_identity_mode": phase1_tracking_identity_mode,
        "phase1_tracking_config_sha256": phase1_tracking_config_sha256,
        "phase1_resource_limits_sha256": phase1_resource_limits_sha256,
        "frame_manifest_sha256": input_hashes["frame_manifest"],
        "detections_sha256": input_hashes["detections"],
        "trajectories_sha256": input_hashes["trajectories"],
        "representatives_sha256": input_hashes["representatives"],
        "model_id": recognizer.model_id,
        "model_revision": recognizer.model_revision,
        "model_weights_sha256": model_weights_sha256,
        "model_config_sha256": model_config_sha256,
        "model_vocab_sha256": vocab_sha256,
        "package_versions": dict(package_versions),
        "runtime": dict(runtime),
        "runtime_identity_sha256": runtime_identity_sha256,
        "total_representatives": total_representatives,
        "commit_interval_records": commit_interval_records,
        "batch_size": batch_size,
        "frame_cache_capacity": frame_cache_capacity,
        "frame_cache_max_bytes": frame_cache_max_bytes,
        "recognition_execution_policy_sha256": recognition_execution_policy_sha256(
            batch_size=batch_size,
            frame_cache_capacity=frame_cache_capacity,
            frame_cache_max_bytes=frame_cache_max_bytes,
        ),
    }


def _receipt_matches_identity(
    receipt: RepresentativeRecognitionReceipt, identity_fields: Mapping[str, object]
) -> None:
    actual = receipt.model_dump(mode="json")
    for field, expected in identity_fields.items():
        if actual[field] != expected:
            raise ValueError(f"recognition receipt identity drift: {field}")


def _load_manifest_frames(path: Path) -> dict[str, FrameRef]:
    frames = [FrameRef.model_validate(row) for row in iter_jsonl(path)]
    if not frames:
        raise ValueError("frame manifest is empty")
    by_uid = {frame.frame_uid: frame for frame in frames}
    if len(by_uid) != len(frames):
        raise ValueError("frame manifest contains duplicate frame_uid values")
    return by_uid


def _materialize_binding_frame(binding: RepresentativeCropBinding, frame: FrameRef) -> FrameRef:
    """Bind a lazy Phase 1 manifest row to its verified representative SHA."""

    expected = (
        frame.video_id,
        frame.frame_idx,
        frame.frame_relpath,
        frame.width,
        frame.height,
    )
    actual = (
        binding.video_id,
        binding.frame_idx,
        binding.frame_relpath,
        binding.source_width,
        binding.source_height,
    )
    if actual != expected:
        raise ValueError(f"representative/frame manifest provenance drift: {binding.frame_uid}")
    if (
        frame.source_image_sha256 is not None
        and frame.source_image_sha256 != binding.source_image_sha256
    ):
        raise ValueError(f"representative/frame manifest provenance drift: {binding.frame_uid}")
    if frame.source_image_sha256 is None:
        return frame.model_copy(update={"source_image_sha256": binding.source_image_sha256})
    return frame


_RESOURCE_LIMIT_FIELDS = (
    "maximum_active_trajectories",
    "maximum_candidate_edges_per_frame",
    "maximum_candidate_edges_per_component",
    "maximum_candidate_evaluations_per_frame",
    "maximum_detections_per_frame",
    "maximum_detections_per_shard",
    "maximum_frames_per_shard",
    "detection_receipt_commit_interval_frames",
)
_LEGACY_OMITTED_FIELD = "detection_receipt_commit_interval_frames"


class _LegacyTrackingIdentityView:
    """Current tracking semantics with the one historical identity projection."""

    def __init__(self, current: TrackingConfig, tracking_sha: str, resource_sha: str) -> None:
        self._current = current
        self.sha256 = tracking_sha
        self.resource_limits_sha256 = resource_sha

    def __getattr__(self, name: str) -> object:
        return getattr(self._current, name)


def _tracking_identity_hashes(config: TrackingConfig) -> tuple[str, str, str, str]:
    values = asdict(config)
    current_tracking = _canonical_hash(values)
    current_resources = _canonical_hash({field: values[field] for field in _RESOURCE_LIMIT_FIELDS})
    if current_tracking != config.sha256 or current_resources != config.resource_limits_sha256:
        raise ValueError("Phase 2 tracking identity projection is stale")
    legacy_values = dict(values)
    legacy_values.pop(_LEGACY_OMITTED_FIELD)
    legacy_resources = {
        field: values[field] for field in _RESOURCE_LIMIT_FIELDS if field != _LEGACY_OMITTED_FIELD
    }
    return (
        current_tracking,
        current_resources,
        _canonical_hash(legacy_values),
        _canonical_hash(legacy_resources),
    )


def _select_tracking_identity_view(
    *,
    tracking_config: TrackingConfig,
    detections: Path,
    trajectories: Path,
    representatives: Path,
) -> tuple[TrackingConfig, Literal["current", "legacy_without_commit_interval"]]:
    """Select current identity or the sole accepted pre-cadence legacy identity."""

    current_tracking, current_resources, legacy_tracking, legacy_resources = (
        _tracking_identity_hashes(tracking_config)
    )
    receipts = [
        OcrPhase1Receipt.model_validate_json(receipt_path_for(path).read_text(encoding="utf-8"))
        for path in (detections, trajectories, representatives)
    ]
    trajectory_records = [
        RepresentativeCropBinding.model_validate(row) for row in iter_jsonl(representatives)
    ]
    tracked_records = [OcrTrajectoryRecord.model_validate(row) for row in iter_jsonl(trajectories)]
    resource_markers = {receipt.resource_limits_sha256 for receipt in receipts}
    tracking_markers = {
        *(record.tracking_config_sha256 for record in tracked_records),
        *(record.tracking_config_sha256 for record in trajectory_records),
    }
    if resource_markers == {current_resources} and tracking_markers <= {current_tracking}:
        return tracking_config, "current"
    if resource_markers == {legacy_resources} and tracking_markers <= {legacy_tracking}:
        view = _LegacyTrackingIdentityView(
            tracking_config,
            tracking_sha=legacy_tracking,
            resource_sha=legacy_resources,
        )
        return cast(TrackingConfig, view), "legacy_without_commit_interval"
    raise ValueError(
        "Phase 1 tracking identity is neither current nor the exact legacy view "
        f"omitting only {_LEGACY_OMITTED_FIELD}"
    )


def _validate_binding_frame(binding: RepresentativeCropBinding, frame: FrameRef) -> None:
    expected = (
        frame.video_id,
        frame.frame_idx,
        frame.frame_relpath,
        frame.source_image_sha256,
        frame.width,
        frame.height,
    )
    actual = (
        binding.video_id,
        binding.frame_idx,
        binding.frame_relpath,
        binding.source_image_sha256,
        binding.source_width,
        binding.source_height,
    )
    if actual != expected:
        raise ValueError(f"representative/frame manifest provenance drift: {binding.frame_uid}")


def _safe_source_path(data_root: Path, frame: FrameRef) -> Path:
    root = data_root.resolve()
    path = (root / frame.frame_relpath).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("representative source path escapes data root") from error
    return path


def _reverify_release_inputs(
    *,
    paths: Mapping[str, Path],
    input_hashes: Mapping[str, str],
    model_config: Path,
    model_config_hash: str,
    model_weights: Path,
    model_weights_hash: str,
    vocab_sha256: str,
    frames_by_uid: Mapping[str, FrameRef],
    representative_records: Sequence[RepresentativeCropBinding],
    data_root: Path,
) -> None:
    """Recheck every mutable trust-boundary input immediately before release."""

    if any(sha256_file(path) != input_hashes[name] for name, path in paths.items()):
        raise ValueError("Phase 1 input changed during representative recognition")
    if (
        sha256_file(model_config) != model_config_hash
        or model_vocab_sha256(model_config) != vocab_sha256
        or sha256_file(model_weights) != model_weights_hash
    ):
        raise ValueError("recognizer model files changed during representative recognition")
    for frame_uid in {binding.frame_uid for binding in representative_records}:
        frame = frames_by_uid[frame_uid]
        if sha256_file(_safe_source_path(data_root, frame)) != frame.source_image_sha256:
            raise ValueError(f"source image changed during recognition: {frame_uid}")


@dataclass(slots=True)
class _CachedCanonicalFrame:
    image: Image.Image
    canonical_image_sha256: str
    size_bytes: int
    resident: bool


class _CanonicalFrameLruCache:
    """Deterministic, entry- and byte-bounded cache of canonical RGB frames."""

    def __init__(self, *, capacity: int, maximum_bytes: int) -> None:
        self.capacity = _bounded_integer(
            "frame_cache_capacity", capacity, maximum=MAX_FRAME_CACHE_CAPACITY
        )
        self.maximum_bytes = _bounded_integer(
            "frame_cache_max_bytes", maximum_bytes, maximum=MAX_FRAME_CACHE_BYTES
        )
        self._entries: OrderedDict[str, _CachedCanonicalFrame] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.oversized = 0

    @property
    def resident_entries(self) -> int:
        return len(self._entries)

    @property
    def resident_bytes(self) -> int:
        return self._bytes

    def get(self, frame: FrameRef, source_path: Path) -> _CachedCanonicalFrame:
        cached = self._entries.pop(frame.frame_uid, None)
        if cached is not None:
            self._entries[frame.frame_uid] = cached
            self.hits += 1
            return cached

        self.misses += 1
        snapshot = decode_canonical_frame(frame, source_path)
        entry = _CachedCanonicalFrame(
            image=snapshot.image,
            canonical_image_sha256=snapshot.canonical_image_sha256,
            size_bytes=snapshot.image.width * snapshot.image.height * 3,
            resident=False,
        )
        if entry.size_bytes > self.maximum_bytes:
            self.oversized += 1
            return entry
        while self._entries and (
            len(self._entries) >= self.capacity
            or self._bytes + entry.size_bytes > self.maximum_bytes
        ):
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted.size_bytes
            evicted.image.close()
            self.evictions += 1
        self._entries[frame.frame_uid] = entry
        self._bytes += entry.size_bytes
        entry.resident = True
        return entry

    def close(self) -> None:
        for entry in self._entries.values():
            entry.image.close()
        self._entries.clear()
        self._bytes = 0


def _reconstruct_crop_image(
    image: Image.Image, provenance: CropProvenance
) -> tuple[Image.Image, bytes]:
    """Reconstruct one exact Phase 1 PNG and retain its RGB image for inference."""

    tl, tr, br, bl = provenance.padded_polygon_xy.points
    crop = image.convert("RGB").transform(
        (provenance.perspective_width, provenance.perspective_height),
        Image.Transform.QUAD,
        (*tl, *bl, *br, *tr),
        resample=Image.Resampling.BICUBIC,
    )
    if provenance.rotation_quadrants_ccw:
        crop = crop.transpose(
            Image.Transpose.ROTATE_270
            if provenance.rotation_quadrants_ccw == 3
            else Image.Transpose.ROTATE_90
        )
    if (crop.width, crop.height) != (provenance.output_width, provenance.output_height):
        crop = crop.resize(
            (provenance.output_width, provenance.output_height),
            resample=Image.Resampling.BICUBIC,
        )
    buffer = io.BytesIO()
    crop.save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=provenance.png_compress_level,
    )
    return crop, buffer.getvalue()


def _prepare_binding_image(
    binding: RepresentativeCropBinding,
    frame: FrameRef,
    *,
    data_root: Path,
    frame_cache: _CanonicalFrameLruCache,
) -> Image.Image:
    _validate_binding_frame(binding, frame)
    source_path = _safe_source_path(data_root, frame)
    cached = frame_cache.get(frame, source_path)
    if cached.canonical_image_sha256 != binding.canonical_image_sha256:
        raise ValueError(f"canonical image checksum drift: {binding.frame_uid}")
    try:
        crop, crop_payload = _reconstruct_crop_image(cached.image, binding.crop)
    finally:
        if not cached.resident:
            cached.image.close()
    if hashlib.sha256(crop_payload).hexdigest() != binding.crop.png_sha256:
        crop.close()
        raise ValueError(f"reconstructed crop checksum drift: {binding.detection_id}")
    return crop


def _predict_images(
    recognizer: CropRecognizer, images: Sequence[Image.Image]
) -> list[RecognitionPrediction]:
    method = getattr(recognizer, "predict_batch", None)
    if callable(method):
        predictions = list(method(images))
    else:
        predictions = [recognizer.predict(image) for image in images]
    if len(predictions) != len(images):
        raise RuntimeError("recognizer batch output length differs from its input")
    if any(not isinstance(prediction, RecognitionPrediction) for prediction in predictions):
        raise TypeError("recognizer returned a non-RecognitionPrediction batch item")
    return predictions


def _result_from_prediction(
    *,
    binding: RepresentativeCropBinding,
    recognizer: CropRecognizer,
    prediction: RecognitionPrediction,
    latency_ms: float,
) -> RepresentativeRecognitionResult:
    raw = prediction.transcript_raw
    normalized = unicodedata.normalize("NFC", raw)
    status: Literal["ok", "empty"] = "ok" if normalized else "empty"
    return RepresentativeRecognitionResult(
        binding=binding,
        model_id=recognizer.model_id,
        model_revision=recognizer.model_revision,
        status=status,
        transcript_raw=raw if status == "ok" else "",
        transcript_nfc=normalized if status == "ok" else "",
        confidence=prediction.confidence,
        latency_ms=max(0.0, latency_ms),
    )


def run_representative_recognition(
    *,
    frame_manifest: Path,
    data_root: Path,
    detections: Path,
    trajectories: Path,
    representatives: Path,
    output: Path,
    run_id: str,
    phase1_config_sha256: str,
    phase1_identity: Phase1Identity,
    tracking_config: TrackingConfig,
    recognizer: CropRecognizer,
    model_config: Path,
    model_weights: Path,
    expected_model_weights_sha256: str,
    source_commit_sha: str,
    package_versions: Mapping[str, str],
    runtime: Mapping[str, str],
    runtime_identity_sha256: str,
    commit_interval_records: int = 32,
    batch_size: int = DEFAULT_RECOGNITION_BATCH_SIZE,
    frame_cache_capacity: int = DEFAULT_FRAME_CACHE_CAPACITY,
    frame_cache_max_bytes: int = DEFAULT_FRAME_CACHE_MAX_BYTES,
    resume: bool = False,
    fault_injector: FaultInjector | None = None,
    clock: Clock = perf_counter,
) -> dict[str, int | str]:
    """Recognize one verified Phase 1 shard with receipt-authenticated resume."""

    if commit_interval_records < 1:
        raise ValueError("commit interval must be positive")
    recognition_execution_policy(
        batch_size=batch_size,
        frame_cache_capacity=frame_cache_capacity,
        frame_cache_max_bytes=frame_cache_max_bytes,
    )
    if not isinstance(source_commit_sha, str) or not (
        7 <= len(source_commit_sha) <= 64
        and all(character in "0123456789abcdef" for character in source_commit_sha)
    ):
        raise ValueError("source commit must be 7..64 lowercase hexadecimal characters")
    paths = {
        "frame_manifest": frame_manifest.resolve(),
        "detections": detections.resolve(),
        "trajectories": trajectories.resolve(),
        "representatives": representatives.resolve(),
    }
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    verified_tracking_config, tracking_identity_mode = _select_tracking_identity_view(
        tracking_config=tracking_config,
        detections=paths["detections"],
        trajectories=paths["trajectories"],
        representatives=paths["representatives"],
    )
    verify_linked_artifacts(
        detections=paths["detections"],
        trajectories=paths["trajectories"],
        representatives=paths["representatives"],
        expected_run_id=run_id,
        expected_config_sha256=phase1_config_sha256,
        expected_identity=phase1_identity,
        tracking_config=verified_tracking_config,
    )
    if any(sha256_file(path) != input_hashes[name] for name, path in paths.items()):
        raise ValueError("Phase 1 input changed during linked-artifact verification")
    phase1_detection_receipt = OcrPhase1Receipt.model_validate_json(
        receipt_path_for(paths["detections"]).read_text(encoding="utf-8")
    )
    if phase1_detection_receipt.input_artifact_sha256 != input_hashes["frame_manifest"]:
        raise ValueError("frame manifest is not the shard manifest bound by Phase 1")

    frames_by_uid = _load_manifest_frames(paths["frame_manifest"])
    representative_records = [
        RepresentativeCropBinding.model_validate(row)
        for row in iter_jsonl(paths["representatives"])
    ]
    for binding in representative_records:
        frame = frames_by_uid.get(binding.frame_uid)
        if frame is None:
            raise ValueError(f"representative references an unknown frame: {binding.frame_uid}")
        frame = _materialize_binding_frame(binding, frame)
        frames_by_uid[binding.frame_uid] = frame
        _validate_binding_frame(binding, frame)

    model_config_hash = sha256_file(model_config)
    model_weights_hash = sha256_file(model_weights)
    if model_weights_hash != expected_model_weights_sha256:
        raise ValueError("model weights SHA-256 differs from the expected pin")
    if recognizer.model_revision != model_weights_hash:
        raise ValueError("recognizer revision differs from the model weights SHA-256")
    vocab_sha = model_vocab_sha256(model_config)
    expected_runtime_hash = _canonical_hash(
        {"packages": dict(package_versions), "runtime": dict(runtime)}
    )
    if runtime_identity_sha256 != expected_runtime_hash:
        raise ValueError("provided runtime identity SHA-256 is inconsistent")
    identity_fields = _identity_fields(
        run_id=run_id,
        source_commit_sha=source_commit_sha,
        phase1_config_sha256=phase1_config_sha256,
        phase1_tracking_identity_mode=tracking_identity_mode,
        phase1_tracking_config_sha256=verified_tracking_config.sha256,
        phase1_resource_limits_sha256=verified_tracking_config.resource_limits_sha256,
        input_hashes=input_hashes,
        recognizer=recognizer,
        model_weights_sha256=model_weights_hash,
        model_config_sha256=model_config_hash,
        vocab_sha256=vocab_sha,
        package_versions=package_versions,
        runtime=runtime,
        runtime_identity_sha256=runtime_identity_sha256,
        total_representatives=len(representative_records),
        commit_interval_records=commit_interval_records,
        batch_size=batch_size,
        frame_cache_capacity=frame_cache_capacity,
        frame_cache_max_bytes=frame_cache_max_bytes,
    )

    output = output.resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt: RepresentativeRecognitionReceipt | None = None
    if receipt_path.is_file():
        receipt = RepresentativeRecognitionReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        )
        _receipt_matches_identity(receipt, identity_fields)

    # Recover the narrow crash window after final output rename but before the
    # completed receipt rename.  The running receipt still authenticates all bytes.
    if output.exists():
        if partial.exists():
            raise ValueError("both final and partial representative outputs exist")
        if receipt is None:
            raise ValueError("final representative output has no receipt")
        payload = output.read_bytes()
        if (
            len(payload) != receipt.committed_bytes
            or hashlib.sha256(payload).hexdigest() != receipt.committed_sha256
        ):
            raise ValueError("final representative output differs from its receipt")
        records = _parse_prefix(
            payload,
            representative_records,
            model_id=recognizer.model_id,
            model_revision=recognizer.model_revision,
        )
        if len(records) != len(representative_records):
            raise ValueError("final representative output is incomplete")
        _reverify_release_inputs(
            paths=paths,
            input_hashes=input_hashes,
            model_config=model_config,
            model_config_hash=model_config_hash,
            model_weights=model_weights,
            model_weights_hash=model_weights_hash,
            vocab_sha256=vocab_sha,
            frames_by_uid=frames_by_uid,
            representative_records=representative_records,
            data_root=data_root,
        )
        if receipt.status == "running":
            completed_receipt = RepresentativeRecognitionReceipt(
                **identity_fields,
                status="completed",
                committed_records=len(records),
                committed_bytes=len(payload),
                committed_sha256=hashlib.sha256(payload).hexdigest(),
                output_sha256=hashlib.sha256(payload).hexdigest(),
                inference_batches=receipt.inference_batches,
                frame_cache_hits=receipt.frame_cache_hits,
                frame_cache_misses=receipt.frame_cache_misses,
                frame_cache_evictions=receipt.frame_cache_evictions,
                frame_cache_oversized=receipt.frame_cache_oversized,
            )
            _write_receipt(receipt_path, completed_receipt)
        return {
            "records": len(records),
            "ok": sum(record.status == "ok" for record in records),
            "empty": sum(record.status == "empty" for record in records),
            "output_sha256": hashlib.sha256(payload).hexdigest(),
            "inference_batches": receipt.inference_batches,
            "frame_cache_hits": receipt.frame_cache_hits,
            "frame_cache_misses": receipt.frame_cache_misses,
            "frame_cache_evictions": receipt.frame_cache_evictions,
            "frame_cache_oversized": receipt.frame_cache_oversized,
        }

    if receipt is None:
        if partial.exists():
            raise ValueError("partial representative output has no running receipt")
        partial.touch(exist_ok=False)
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        empty_hash = hashlib.sha256(b"").hexdigest()
        receipt = RepresentativeRecognitionReceipt(
            **identity_fields,
            status="running",
            committed_records=0,
            committed_bytes=0,
            committed_sha256=empty_hash,
            output_sha256=empty_hash,
            inference_batches=0,
            frame_cache_hits=0,
            frame_cache_misses=0,
            frame_cache_evictions=0,
            frame_cache_oversized=0,
        )
        _write_receipt(receipt_path, receipt)
    else:
        if not resume:
            raise FileExistsError("running recognition exists; pass resume=True")
        if receipt.status != "running" or not partial.is_file():
            raise ValueError("running receipt requires a partial representative output")

    payload = partial.read_bytes()
    if len(payload) < receipt.committed_bytes:
        raise ValueError("partial representative output is shorter than committed bytes")
    committed_payload = payload[: receipt.committed_bytes]
    if hashlib.sha256(committed_payload).hexdigest() != receipt.committed_sha256:
        raise ValueError("committed representative prefix checksum mismatch")
    completed = _parse_prefix(
        committed_payload,
        representative_records,
        model_id=recognizer.model_id,
        model_revision=recognizer.model_revision,
    )
    if len(completed) != receipt.committed_records:
        raise ValueError("receipt committed record count mismatch")
    if len(payload) != receipt.committed_bytes:
        with partial.open("r+b") as stream:
            stream.truncate(receipt.committed_bytes)
            stream.flush()
            os.fsync(stream.fileno())

    base_inference_batches = receipt.inference_batches
    base_frame_cache_hits = receipt.frame_cache_hits
    base_frame_cache_misses = receipt.frame_cache_misses
    base_frame_cache_evictions = receipt.frame_cache_evictions
    base_frame_cache_oversized = receipt.frame_cache_oversized
    inference_batches = 0

    frame_cache = _CanonicalFrameLruCache(
        capacity=frame_cache_capacity,
        maximum_bytes=frame_cache_max_bytes,
    )
    try:
        with partial.open("ab") as stream:
            while len(completed) < len(representative_records):
                until_commit = commit_interval_records - (len(completed) % commit_interval_records)
                current_batch_size = min(
                    batch_size,
                    until_commit,
                    len(representative_records) - len(completed),
                )
                bindings = representative_records[
                    len(completed) : len(completed) + current_batch_size
                ]
                images: list[Image.Image] = []
                try:
                    for binding in bindings:
                        images.append(
                            _prepare_binding_image(
                                binding,
                                frames_by_uid[binding.frame_uid],
                                data_root=data_root,
                                frame_cache=frame_cache,
                            )
                        )
                    started = clock()
                    try:
                        predictions = _predict_images(recognizer, images)
                    except Exception as error:
                        message = " ".join(str(error).replace("\r", " ").replace("\n", " ").split())
                        first = bindings[0]
                        raise RepresentativeInferenceError(
                            f"recognition failed for batch at {first.trajectory_id} rank "
                            f"{first.representative_rank}: {type(error).__name__}: "
                            f"{(message or type(error).__name__)[:500]}"
                        ) from error
                    elapsed_per_record_ms = max(
                        0.0, (clock() - started) * 1000 / current_batch_size
                    )
                    inference_batches += 1
                    batch_records = [
                        _result_from_prediction(
                            binding=binding,
                            recognizer=recognizer,
                            prediction=prediction,
                            latency_ms=elapsed_per_record_ms,
                        )
                        for binding, prediction in zip(bindings, predictions, strict=True)
                    ]
                    batch_payload = b"".join(_record_bytes(record) for record in batch_records)
                finally:
                    for image in images:
                        image.close()

                # Nothing from a failed/incomplete batch reaches the append-only prefix.
                stream.write(batch_payload)
                stream.flush()
                completed.extend(batch_records)
                if len(completed) % commit_interval_records == 0:
                    os.fsync(stream.fileno())
                    committed_bytes = stream.tell()
                    committed_hash = sha256_file(partial)
                    receipt = RepresentativeRecognitionReceipt(
                        **identity_fields,
                        status="running",
                        committed_records=len(completed),
                        committed_bytes=committed_bytes,
                        committed_sha256=committed_hash,
                        output_sha256=committed_hash,
                        inference_batches=base_inference_batches + inference_batches,
                        frame_cache_hits=base_frame_cache_hits + frame_cache.hits,
                        frame_cache_misses=base_frame_cache_misses + frame_cache.misses,
                        frame_cache_evictions=base_frame_cache_evictions + frame_cache.evictions,
                        frame_cache_oversized=base_frame_cache_oversized + frame_cache.oversized,
                    )
                    _write_receipt(receipt_path, receipt)
                    if fault_injector is not None:
                        fault_injector("after_running_receipt")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        frame_cache.close()

    final_payload = partial.read_bytes()
    final_hash = hashlib.sha256(final_payload).hexdigest()
    receipt = RepresentativeRecognitionReceipt(
        **identity_fields,
        status="running",
        committed_records=len(completed),
        committed_bytes=len(final_payload),
        committed_sha256=final_hash,
        output_sha256=final_hash,
        inference_batches=base_inference_batches + inference_batches,
        frame_cache_hits=base_frame_cache_hits + frame_cache.hits,
        frame_cache_misses=base_frame_cache_misses + frame_cache.misses,
        frame_cache_evictions=base_frame_cache_evictions + frame_cache.evictions,
        frame_cache_oversized=base_frame_cache_oversized + frame_cache.oversized,
    )
    _write_receipt(receipt_path, receipt)

    _reverify_release_inputs(
        paths=paths,
        input_hashes=input_hashes,
        model_config=model_config,
        model_config_hash=model_config_hash,
        model_weights=model_weights,
        model_weights_hash=model_weights_hash,
        vocab_sha256=vocab_sha,
        frames_by_uid=frames_by_uid,
        representative_records=representative_records,
        data_root=data_root,
    )

    os.replace(partial, output)
    _fsync_directory(output.parent)
    if fault_injector is not None:
        fault_injector("after_output_rename")
    _reverify_release_inputs(
        paths=paths,
        input_hashes=input_hashes,
        model_config=model_config,
        model_config_hash=model_config_hash,
        model_weights=model_weights,
        model_weights_hash=model_weights_hash,
        vocab_sha256=vocab_sha,
        frames_by_uid=frames_by_uid,
        representative_records=representative_records,
        data_root=data_root,
    )
    completed_receipt = RepresentativeRecognitionReceipt(
        **identity_fields,
        status="completed",
        committed_records=len(completed),
        committed_bytes=len(final_payload),
        committed_sha256=final_hash,
        output_sha256=final_hash,
        inference_batches=base_inference_batches + inference_batches,
        frame_cache_hits=base_frame_cache_hits + frame_cache.hits,
        frame_cache_misses=base_frame_cache_misses + frame_cache.misses,
        frame_cache_evictions=base_frame_cache_evictions + frame_cache.evictions,
        frame_cache_oversized=base_frame_cache_oversized + frame_cache.oversized,
    )
    _write_receipt(receipt_path, completed_receipt)
    return {
        "records": len(completed),
        "ok": sum(record.status == "ok" for record in completed),
        "empty": sum(record.status == "empty" for record in completed),
        "output_sha256": final_hash,
        "inference_batches": base_inference_batches + inference_batches,
        "frame_cache_hits": base_frame_cache_hits + frame_cache.hits,
        "frame_cache_misses": base_frame_cache_misses + frame_cache.misses,
        "frame_cache_evictions": base_frame_cache_evictions + frame_cache.evictions,
        "frame_cache_oversized": base_frame_cache_oversized + frame_cache.oversized,
    }
