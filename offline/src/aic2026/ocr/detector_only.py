"""Pinned, offline-only adapter for PP-OCRv6_small_det.

This module deliberately uses PaddleOCR's ``TextDetection`` pipeline instead
of ``PaddleOCR``.  No recognition model argument or recognition code path
exists here.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any
from weakref import WeakSet

import numpy as np

from .geometry import CropGeometryError, canonical_quad, validate_canonical_quad
from .ppocrv6 import PaddleOcrV6Error, network_forbidden

DETECTOR_ID = "PP-OCRv6_small_det"
DETECTOR_MODEL_ID = "ppocrv6-small-det"
DETECTOR_REVISION = "01038f24bb3ca833f40dfd0eba0b81f0d92b5275576e3696ced20a5a1f619d06"
PINNED_SOURCE_REGISTRY_SHA256 = "9b221b4dd366c850e8a8bf6b4f11ca13becb921d51dddf2dcabdd41a1eaab5f7"
# Hash of the exact ``model`` mapping in configs/offline/ocr_phase1.yaml.
PINNED_DETECTOR_PORT_CONFIG_SHA256 = (
    "577d38c2e6b6ea373f32a10b1ce90ef33db3b609b672a32cd7825b734c8ab535"
)
DETECTOR_TREE_SHA256 = "5ee508811bc9f799f68d83fafa7a60ee0f94e0ee5415c07d328895b6b41340bd"
RUNTIME_IDENTITY_SHA256 = "c89e9fcf8a6c1065f4cbd9ea20e06646130db601663a232eb6bfda58390a0723"
RUNTIME_DISTRIBUTIONS = (
    "paddlepaddle",
    "paddleocr",
    "paddlex",
    "pyclipper",
    "opencv-contrib-python",
    "Pillow",
    "numpy",
)
OPENCV_DISTRIBUTION = "opencv-contrib-python"


class DetectorOnlyError(RuntimeError):
    """Detector-only preflight or inference failed closed."""


@dataclass(frozen=True, slots=True)
class DetectorPolygon:
    """Raw PaddleX vertices plus deterministic canonical/clamped crop vertices."""

    source_order: int
    raw_points: tuple[tuple[float, float], ...]
    points: tuple[tuple[float, float], ...]
    score: float
    clamped: bool


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DetectorOnlyError(f"{name} must be a mapping")
    return value


def _finite_number(value: Any) -> bool:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return False
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value))


def _array(value: Any, name: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(value)
    if isinstance(value, Iterable) and not isinstance(value, Mapping | str | bytes | bytearray):
        return list(value)
    raise DetectorOnlyError(f"{name} must be an array")


def _nondegenerate_quad(points: Sequence[tuple[float, float]]) -> bool:
    if len(points) != 4 or len(set(points)) != 4:
        return False

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    ordered = sorted(points)
    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-6:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-6:
            upper.pop()
        upper.append(point)
    return len(lower[:-1] + upper[:-1]) == 4


def _model_path(cache_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise DetectorOnlyError("detector model path must be relative to AIC_CACHE_ROOT")
    root = cache_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise DetectorOnlyError("detector model path escapes AIC_CACHE_ROOT") from error
    return path


def _installed_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in RUNTIME_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise DetectorOnlyError(f"required package is not installed: {distribution}") from error
    return versions


def identity_from_locked_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Pure expected identity derivation; never imports or inspects OCR packages."""

    if config.get("execution_profile") != "cpu_pinned":
        raise DetectorOnlyError("runnable detector config requires execution_profile=cpu_pinned")
    model = _mapping(config.get("model"), "model")
    if _canonical_hash(model) != PINNED_DETECTOR_PORT_CONFIG_SHA256:
        raise DetectorOnlyError("portable detector-only configuration drift")
    expected_scalars = {
        "id": DETECTOR_MODEL_ID,
        "candidate_id": "ppocrv6-small-det-gpt4o-mini-high-v1",
        "revision": DETECTOR_REVISION,
        "source_registry_sha256": PINNED_SOURCE_REGISTRY_SHA256,
        "device": "cpu",
        "enable_mkldnn": False,
        "download_allowed": False,
        "fallback": False,
        "ensemble": False,
        "network_policy": "execution_environment_internet_disabled_plus_python_best_effort",
    }
    for name, expected in expected_scalars.items():
        if model.get(name) != expected:
            raise DetectorOnlyError(f"detector-only model field {name!r} differs from pin")
    if "recognizer" in model or "recognition" in model:
        raise DetectorOnlyError("detector-only config cannot contain a recognizer")

    expected_packages = _mapping(model.get("packages"), "model.packages")
    if set(expected_packages) != set(RUNTIME_DISTRIBUTIONS):
        raise DetectorOnlyError("exact OCR/crop runtime package identity is required")
    component = _mapping(model.get("detector"), "model.detector")
    if component.get("model_name") != DETECTOR_ID:
        raise DetectorOnlyError("detector identity differs from PP-OCRv6_small_det")
    return {
        "detector_id": DETECTOR_ID,
        "detector_revision": DETECTOR_REVISION,
        "detector_tree_sha256": _canonical_hash(component),
        "runtime_identity_sha256": _canonical_hash(
            {"packages": dict(expected_packages), "cv2_provider": OPENCV_DISTRIBUTION}
        ),
        "packages": dict(expected_packages),
        "cv2_provider": OPENCV_DISTRIBUTION,
        "network_policy": model["network_policy"],
    }


def runtime_identity_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Backward-compatible name for pure locked-config identity derivation."""

    return identity_from_locked_config(config)


def _verify_runtime_identity(
    config: Mapping[str, Any],
    *,
    package_versions: Mapping[str, str] | None = None,
    opencv_providers: Sequence[str] | None = None,
) -> dict[str, Any]:
    expected = identity_from_locked_config(config)
    actual_packages = (
        dict(package_versions) if package_versions is not None else _installed_package_versions()
    )
    if actual_packages != expected["packages"]:
        raise DetectorOnlyError("installed OCR/crop package versions differ from runtime pin")
    providers = (
        list(opencv_providers)
        if opencv_providers is not None
        else list(importlib.metadata.packages_distributions().get("cv2") or [])
    )
    if providers != [OPENCV_DISTRIBUTION]:
        raise DetectorOnlyError(
            f"cv2 must come only from {OPENCV_DISTRIBUTION!r}; got {providers!r}"
        )
    return expected


def verify_runtime_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Production runtime inspection with no caller-supplied version override."""

    return _verify_runtime_identity(config)


def verify_detector_only(
    config: Mapping[str, Any],
    cache_root: Path,
) -> dict[str, Any]:
    """Verify config, packages and every detector file before construction."""

    return _verify_detector_only(config, cache_root)


def _verify_detector_only(
    config: Mapping[str, Any],
    cache_root: Path,
    *,
    package_versions: Mapping[str, str] | None = None,
    opencv_providers: Sequence[str] | None = None,
) -> dict[str, Any]:
    identity = _verify_runtime_identity(
        config,
        package_versions=package_versions,
        opencv_providers=opencv_providers,
    )
    model = _mapping(config.get("model"), "model")
    component = _mapping(model.get("detector"), "model.detector")
    path = _model_path(cache_root, component.get("path"))
    files = component.get("files")
    if not path.is_dir() or not isinstance(files, list) or not files:
        raise DetectorOnlyError("detector model directory or file identity is unavailable")
    seen: set[str] = set()
    checked: list[dict[str, Any]] = []
    for raw_entry in files:
        entry = _mapping(raw_entry, "detector file")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative or relative in seen:
            raise DetectorOnlyError("detector model file path is invalid or duplicated")
        seen.add(relative)
        file_path = (path / relative).resolve()
        try:
            file_path.relative_to(path)
        except ValueError as error:
            raise DetectorOnlyError("detector model file escapes its directory") from error
        actual = {
            "path": relative,
            "bytes": file_path.stat().st_size if file_path.is_file() else -1,
            "sha256": _sha256(file_path) if file_path.is_file() else None,
        }
        if actual != dict(entry):
            raise DetectorOnlyError(f"detector model file checksum mismatch: {relative}")
        checked.append(actual)
    return {
        **identity,
        "model_path": str(path),
        "files": checked,
        "verified_before_constructor": True,
        "recognizer_configured": False,
        "python_network_guard": "best_effort_only",
    }


def _default_constructor(**kwargs: Any) -> Any:
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    from paddleocr import TextDetection

    return TextDetection(**kwargs)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_snapshot(path: Path, files: Sequence[Mapping[str, Any]]) -> None:
    for entry in files:
        target = (path / str(entry["path"])).resolve()
        try:
            target.relative_to(path)
        except ValueError as error:
            raise DetectorOnlyError("model snapshot file escapes content-addressed root") from error
        if (
            not target.is_file()
            or target.stat().st_size != entry["bytes"]
            or _sha256(target) != entry["sha256"]
            or target.stat().st_mode & 0o222
        ):
            raise DetectorOnlyError(f"model snapshot checksum mismatch: {entry['path']}")
    if path.stat().st_mode & 0o222:
        raise DetectorOnlyError("model snapshot root must be read-only")


def _snapshot_model(verification: Mapping[str, Any], runtime_root: Path) -> Path:
    """Copy and hash pinned model bytes into a private content-addressed tree."""

    source = Path(str(verification["model_path"])).resolve()
    files = list(verification["files"])
    snapshots = runtime_root / "ocr" / "model-snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    destination = snapshots / str(verification["detector_tree_sha256"])
    if destination.exists():
        _verify_snapshot(destination, files)
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=snapshots))
    try:
        for entry in files:
            relative = str(entry["path"])
            source_file = (source / relative).resolve()
            target = (temporary / relative).resolve()
            try:
                source_file.relative_to(source)
                target.relative_to(temporary)
            except ValueError as error:
                raise DetectorOnlyError("model snapshot path escapes its root") from error
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            copied = 0
            with source_file.open("rb") as reader, target.open("xb") as writer:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    digest.update(chunk)
                    copied += len(chunk)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            if copied != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                raise DetectorOnlyError(f"model changed while snapshotting: {relative}")
            target.chmod(0o444)
        for directory in sorted(
            (item for item in temporary.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
            directory.chmod(0o555)
        _fsync_directory(temporary)
        temporary.chmod(0o555)
        os.replace(temporary, destination)
        _fsync_directory(snapshots)
    except Exception:
        # Keep incomplete bytes as evidence; a later run never treats this
        # non-content-addressed temporary name as a valid snapshot.
        raise
    _verify_snapshot(destination, files)
    return destination


def _result_payload(value: Any, index: int) -> Mapping[str, Any]:
    payload = getattr(value, "json", value)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as error:
            raise DetectorOnlyError(f"detector result[{index}].json is invalid") from error
    outer = _mapping(payload, f"detector result[{index}]")
    return _mapping(outer.get("res", outer), f"detector result[{index}].res")


def parse_detector_output(results: Any, *, width: int, height: int) -> list[DetectorPolygon]:
    """Parse PaddleOCR/PaddleX 3.7 detector output (``dt_polys``/``dt_scores``)."""

    if width < 2 or height < 2:
        raise DetectorOnlyError("detector frame dimensions must be at least 2x2")
    outer_results = _array(results, "TextDetection.predict output")
    if len(outer_results) != 1:
        raise DetectorOnlyError("detector must return exactly one result per input frame")
    detections: list[DetectorPolygon] = []
    for result_index, result in enumerate(outer_results):
        payload = _result_payload(result, result_index)
        if "dt_polys" not in payload or "dt_scores" not in payload:
            raise DetectorOnlyError("pinned detector output requires dt_polys and dt_scores")
        polygons = _array(payload["dt_polys"], "dt_polys")
        scores = _array(payload["dt_scores"], "dt_scores")
        if len(polygons) != len(scores):
            raise DetectorOnlyError("dt_polys and dt_scores lengths differ")
        for polygon_value, score_value in zip(polygons, scores, strict=True):
            raw_points_value = _array(polygon_value, "dt_polys item")
            if len(raw_points_value) != 4:
                raise DetectorOnlyError("detector polygons must be quadrilaterals")
            raw_points: list[tuple[float, float]] = []
            for point_value in raw_points_value:
                point = _array(point_value, "dt_polys point")
                if len(point) != 2 or not all(_finite_number(value) for value in point):
                    raise DetectorOnlyError("detector polygon coordinate is invalid")
                raw_points.append((float(point[0]), float(point[1])))
            if not _nondegenerate_quad(raw_points):
                raise DetectorOnlyError("detector polygon is degenerate or non-convex")
            try:
                canonical_points = canonical_quad(raw_points)
            except CropGeometryError as error:
                raise DetectorOnlyError("detector polygon cannot be canonicalized") from error
            if not _finite_number(score_value) or not 0 <= float(score_value) <= 1:
                raise DetectorOnlyError("detector score must be finite inside [0, 1]")
            points = tuple(
                (
                    min(max(x, 0.0), width - 1.0),
                    min(max(y, 0.0), height - 1.0),
                )
                for x, y in canonical_points
            )
            if not _nondegenerate_quad(points):
                raise DetectorOnlyError("detector polygon collapsed after frame-bounds clamp")
            try:
                validate_canonical_quad(points)
            except CropGeometryError as error:
                raise DetectorOnlyError(
                    "detector polygon ordering invalid after frame-bounds clamp"
                ) from error
            detections.append(
                DetectorPolygon(
                    source_order=len(detections),
                    raw_points=tuple(raw_points),
                    points=points,
                    score=float(score_value),
                    clamped=points != canonical_points,
                )
            )
    return detections


def _frozen_evidence(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _frozen_evidence(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_frozen_evidence(item) for item in value)
    return value


def _detector_sealers() -> tuple[Callable[..., Any], Callable[..., Any], Callable[[Any], bool]]:
    """Keep production attestation state inside this process-local closure."""

    production_instances: WeakSet[Any] = WeakSet()

    def seal(
        detector_type: type[PaddleOcrV6Detector],
        engine: Any,
        verification: Mapping[str, Any],
        *,
        production: bool,
    ) -> PaddleOcrV6Detector:
        detector = object.__new__(detector_type)
        object.__setattr__(detector, "_engine", engine)
        object.__setattr__(detector, "_verification", _frozen_evidence(verification))
        if production:
            production_instances.add(detector)
        return detector

    def seal_production(
        detector_type: type[PaddleOcrV6Detector],
        engine: Any,
        verification: Mapping[str, Any],
    ) -> PaddleOcrV6Detector:
        return seal(detector_type, engine, verification, production=True)

    def seal_test(
        detector_type: type[PaddleOcrV6Detector],
        engine: Any,
        verification: Mapping[str, Any],
    ) -> PaddleOcrV6Detector:
        return seal(detector_type, engine, verification, production=False)

    def is_production(detector: Any) -> bool:
        return detector in production_instances

    return seal_production, seal_test, is_production


_seal_production_detector, _seal_test_detector, is_production_detector_attested = (
    _detector_sealers()
)


class PaddleOcrV6Detector:
    """Exactly one verified detector-only Paddle pipeline."""

    __slots__ = ("_engine", "_verification", "__weakref__")

    def __init__(self, engine: Any, *, verification: Mapping[str, Any]) -> None:
        raise DetectorOnlyError("direct detector construction is forbidden; use the factory")

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise DetectorOnlyError("detector engine and attestation evidence are sealed")

    @property
    def verification(self) -> Mapping[str, Any]:
        return self._verification

    @classmethod
    def create(
        cls,
        *,
        config: Mapping[str, Any],
        cache_root: Path,
        runtime_cache_root: Path,
    ) -> PaddleOcrV6Detector:
        engine, verification = cls._construct(
            config=config,
            cache_root=cache_root,
            runtime_cache_root=runtime_cache_root,
        )
        return _seal_production_detector(cls, engine, verification)

    @classmethod
    def _create_for_test(
        cls,
        *,
        config: Mapping[str, Any],
        cache_root: Path,
        runtime_cache_root: Path | None = None,
        constructor: Callable[..., Any] | None = None,
        package_versions: Mapping[str, str] | None = None,
        opencv_providers: Sequence[str] | None = None,
    ) -> PaddleOcrV6Detector:
        """Explicit non-production factory for isolated adapter tests."""

        engine, verification = cls._construct(
            config=config,
            cache_root=cache_root,
            runtime_cache_root=runtime_cache_root,
            constructor=constructor,
            package_versions=package_versions,
            opencv_providers=opencv_providers,
        )
        return _seal_test_detector(cls, engine, verification)

    @classmethod
    def _construct(
        cls,
        *,
        config: Mapping[str, Any],
        cache_root: Path,
        runtime_cache_root: Path | None = None,
        constructor: Callable[..., Any] | None = None,
        package_versions: Mapping[str, str] | None = None,
        opencv_providers: Sequence[str] | None = None,
    ) -> tuple[Any, Mapping[str, Any]]:
        verification = _verify_detector_only(
            config,
            cache_root,
            package_versions=package_versions,
            opencv_providers=opencv_providers,
        )
        model = _mapping(config["model"], "model")
        root = (runtime_cache_root or cache_root).resolve()
        runtime_cache = (root / "ocr" / "paddlex-runtime-detector").resolve()
        try:
            runtime_cache.relative_to(root)
        except ValueError as error:
            raise DetectorOnlyError("Paddle runtime cache escapes AIC_CACHE_ROOT") from error
        paddle_already_imported = any(
            name == "paddleocr"
            or name.startswith("paddleocr.")
            or name == "paddlex"
            or name.startswith("paddlex.")
            for name in sys.modules
        )
        if paddle_already_imported:
            raise DetectorOnlyError(
                "PaddleX/PaddleOCR was imported before detector setup; use a fresh isolated "
                "detector process because cached module constants cannot be proven safe"
            )
        runtime_cache.mkdir(parents=True, exist_ok=True)
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(runtime_cache)
        snapshot = _snapshot_model(verification, root)
        verification = {
            **verification,
            "source_model_path": verification["model_path"],
            "model_path": str(snapshot),
            "model_snapshot_verified": True,
            "model_snapshot_read_only": True,
        }
        kwargs = {
            "model_name": DETECTOR_ID,
            "model_dir": verification["model_path"],
            "device": model["device"],
            "enable_mkldnn": model["enable_mkldnn"],
        }
        try:
            with network_forbidden():
                engine = (constructor or _default_constructor)(**kwargs)
        except PaddleOcrV6Error as error:
            raise DetectorOnlyError(str(error)) from error
        _verify_snapshot(snapshot, verification["files"])
        return engine, verification

    def detect(self, image_bgr: np.ndarray, *, width: int, height: int) -> list[DetectorPolygon]:
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.dtype != np.uint8
            or image_bgr.shape != (height, width, 3)
        ):
            raise DetectorOnlyError("detector input must be canonical HxWx3 BGR uint8 pixels")
        try:
            with network_forbidden():
                raw = self._engine.predict(image_bgr)
                return parse_detector_output(raw, width=width, height=height)
        except PaddleOcrV6Error as error:
            raise DetectorOnlyError(str(error)) from error
