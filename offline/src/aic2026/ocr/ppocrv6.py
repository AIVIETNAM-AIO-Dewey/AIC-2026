"""Pinned, local-only PP-OCRv6-small adapter.

Model packages and every local detector/recognizer file are verified before
PaddleOCR is imported or constructed. Constructor and inference use a Python
best-effort socket guard; production jobs must additionally disable Internet
at the execution-environment level. There is no download, fallback, retry, or
ensemble.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import socket
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from numbers import Real
from pathlib import Path
from typing import Any

from PIL import Image

from aic2026.contracts import OcrText

from .pipeline import normalize_vietnamese_text

DEFAULT_MODEL_ID = "ppocrv6-small"
PINNED_SOURCE_REGISTRY_SHA256 = "9b221b4dd366c850e8a8bf6b4f11ca13becb921d51dddf2dcabdd41a1eaab5f7"
PINNED_CONFIGURATION_SHA256 = "01038f24bb3ca833f40dfd0eba0b81f0d92b5275576e3696ced20a5a1f619d06"
PINNED_PORT_CONFIG_SHA256 = "c2651f57565205b6706cd5d32273ba2526f6aa7ad71c81f76eac508f288378c7"
MODEL_DISTRIBUTIONS = ("paddlepaddle", "paddleocr", "paddlex")
_NETWORK_LOCK = threading.RLock()


class PaddleOcrV6Error(RuntimeError):
    """PP-OCRv6 preflight or extraction failed closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_finite(value: Any) -> bool:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return False
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value))


def installed_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in MODEL_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise PaddleOcrV6Error(f"required package is not installed: {distribution}") from error
    return versions


@contextmanager
def network_forbidden() -> Iterator[None]:
    """Best-effort guard for Python sockets created while this context is active.

    This cannot constrain pre-existing sockets, native code, or subprocesses;
    production isolation must disable Internet outside the Python process.
    """

    def denied(*_args: Any, **_kwargs: Any) -> Any:
        raise PaddleOcrV6Error("network access is forbidden during PP-OCRv6 execution")

    with _NETWORK_LOCK:
        original_socket = socket.socket
        original_create_connection = socket.create_connection

        class BlockedSocket(original_socket):
            def connect(self, *_args: Any, **_kwargs: Any) -> None:
                denied()

            def connect_ex(self, *_args: Any, **_kwargs: Any) -> int:
                denied()

            def sendto(self, *_args: Any, **_kwargs: Any) -> int:
                denied()

            def sendmsg(self, *_args: Any, **_kwargs: Any) -> int:
                denied()

        socket.socket = BlockedSocket  # type: ignore[assignment,misc]
        socket.create_connection = denied  # type: ignore[assignment]
        try:
            yield
        finally:
            socket.socket = original_socket  # type: ignore[assignment]
            socket.create_connection = original_create_connection  # type: ignore[assignment]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PaddleOcrV6Error(f"{name} must be a mapping")
    return value


def _component_path(cache_root: Path, component: Mapping[str, Any], role: str) -> Path:
    relative = component.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise PaddleOcrV6Error(f"{role} model path must be relative to AIC_CACHE_ROOT")
    root = cache_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise PaddleOcrV6Error(f"{role} model path escapes AIC_CACHE_ROOT") from error
    return path


def verify_ppocrv6(
    config: Mapping[str, Any],
    cache_root: Path,
    *,
    package_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify the complete PP-OCRv6-small identity without importing Paddle."""

    model = _mapping(config.get("model"), "model")
    if model.get("id") != DEFAULT_MODEL_ID:
        raise PaddleOcrV6Error("only the pinned PP-OCRv6-small model is runnable")
    if _mapping_sha256(model) != PINNED_PORT_CONFIG_SHA256:
        raise PaddleOcrV6Error("portable PP-OCRv6 model configuration drift")
    if model.get("source_registry_sha256") != PINNED_SOURCE_REGISTRY_SHA256:
        raise PaddleOcrV6Error("source registry identity differs from the approved OCR contract")
    if model.get("configuration_sha256") != PINNED_CONFIGURATION_SHA256:
        raise PaddleOcrV6Error("PP-OCRv6 configuration identity drift")
    if model.get("device") != "cpu":
        raise PaddleOcrV6Error("the approved PP-OCRv6 identity is pinned to CPU")
    if model.get("download_allowed") is not False:
        raise PaddleOcrV6Error("model download must be explicitly disabled")
    if model.get("fallback") is not False or model.get("ensemble") is not False:
        raise PaddleOcrV6Error("fallback and ensemble must be explicitly disabled")
    threshold = model.get("confidence_threshold")
    if not _is_finite(threshold) or float(threshold) != 0.5:
        raise PaddleOcrV6Error("confidence threshold must equal the pinned value 0.50")

    expected_packages = _mapping(model.get("packages"), "model.packages")
    actual_packages = (
        dict(package_versions) if package_versions is not None else installed_package_versions()
    )
    if set(expected_packages) != set(MODEL_DISTRIBUTIONS):
        raise PaddleOcrV6Error("exact Paddle package identity is required")
    if actual_packages != dict(expected_packages):
        raise PaddleOcrV6Error("installed Paddle package versions differ from the pinned identity")

    components = _mapping(model.get("components"), "model.components")
    if set(components) != {"detector", "recognizer"}:
        raise PaddleOcrV6Error("exact detector and recognizer identities are required")
    verified: dict[str, Any] = {}
    expected_names = {
        "detector": "PP-OCRv6_small_det",
        "recognizer": "PP-OCRv6_small_rec",
    }
    for role, expected_name in expected_names.items():
        component = _mapping(components[role], f"model.components.{role}")
        if component.get("model_name") != expected_name:
            raise PaddleOcrV6Error(f"{role} identity differs from PP-OCRv6-small")
        path = _component_path(cache_root, component, role)
        files = component.get("files")
        if not path.is_dir() or not isinstance(files, list) or not files:
            raise PaddleOcrV6Error(f"{role} model directory or file identity is unavailable")
        seen: set[str] = set()
        checked: list[dict[str, Any]] = []
        for entry_value in files:
            entry = _mapping(entry_value, f"{role} file")
            relative = entry.get("path")
            if not isinstance(relative, str) or not relative or relative in seen:
                raise PaddleOcrV6Error(f"{role} model file path is invalid or duplicated")
            seen.add(relative)
            file_path = (path / relative).resolve()
            try:
                file_path.relative_to(path)
            except ValueError as error:
                raise PaddleOcrV6Error(f"{role} model file escapes its directory") from error
            actual = {
                "path": relative,
                "bytes": file_path.stat().st_size if file_path.is_file() else -1,
                "sha256": _sha256(file_path) if file_path.is_file() else None,
            }
            if actual != dict(entry):
                raise PaddleOcrV6Error(f"{role} model file checksum mismatch: {relative}")
            checked.append(actual)
        verified[role] = {"model_name": expected_name, "path": str(path), "files": checked}
    return {
        "model_id": DEFAULT_MODEL_ID,
        "configuration_sha256": PINNED_CONFIGURATION_SHA256,
        "source_registry_sha256": PINNED_SOURCE_REGISTRY_SHA256,
        "packages": actual_packages,
        "components": verified,
        "verified_before_constructor": True,
    }


def _default_constructor(**kwargs: Any) -> Any:
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    from paddleocr import PaddleOCR

    return PaddleOCR(**kwargs)


def _as_array(value: Any, name: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise PaddleOcrV6Error(f"{name} must be an array")
    return list(value)


def _raw_detections(results: Any) -> list[dict[str, Any]]:
    rows = _as_array(results, "PaddleOCR predict output")
    detections: list[dict[str, Any]] = []
    for result_index, result in enumerate(rows):
        value = getattr(result, "json", result)
        payload = _mapping(value, f"predict result[{result_index}]")
        payload = _mapping(payload.get("res", payload), f"predict result[{result_index}].res")
        text_value = payload.get("rec_texts")
        polygon_value = payload.get("rec_polys")
        score_value = payload.get("rec_scores")
        texts = _as_array(text_value if text_value is not None else [], "rec_texts")
        polygons = _as_array(polygon_value if polygon_value is not None else [], "rec_polys")
        scores = _as_array(score_value if score_value is not None else [], "rec_scores")
        for index, text in enumerate(texts):
            detections.append(
                {
                    "raw_text": text,
                    "confidence": scores[index] if index < len(scores) else None,
                    "polygon": polygons[index] if index < len(polygons) else None,
                }
            )
    return detections


def _polygon(value: Any) -> list[tuple[float, float]] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return None
    points: list[tuple[float, float]] = []
    for point in value:
        if hasattr(point, "tolist"):
            point = point.tolist()
        if not isinstance(point, Sequence) or len(point) != 2:
            return None
        if not _is_finite(point[0]) or not _is_finite(point[1]):
            return None
        points.append((float(point[0]), float(point[1])))
    if len(points) < 3:
        return None
    area = (
        abs(
            sum(
                points[index][0] * points[(index + 1) % len(points)][1]
                - points[(index + 1) % len(points)][0] * points[index][1]
                for index in range(len(points))
            )
        )
        / 2
    )
    return points if area > 0 else None


def _normalize_detection(
    raw: Mapping[str, Any], index: int, width: int, height: int, threshold: float
) -> OcrText:
    raw_text = str(raw.get("raw_text", ""))
    normalized_text = normalize_vietnamese_text(raw_text)
    value = raw.get("confidence")
    if value is None:
        confidence = None
        confidence_semantics = "not_provided"
    elif _is_finite(value) and 0 <= float(value) <= 1:
        confidence = float(value)
        confidence_semantics = "engine_native_score"
    else:
        raise PaddleOcrV6Error(f"detection[{index}] has invalid confidence")
    raw_polygon = _polygon(raw.get("polygon"))
    native_polygon: list[tuple[float, float]] | None = None
    normalized_polygon: list[tuple[float, float]] | None = None
    clamped = False
    warning: str | None = None
    if raw_polygon is None:
        warning = "missing_or_malformed_engine_polygon"
    else:
        native_polygon = [
            (min(max(x, 0.0), width - 1.0), min(max(y, 0.0), height - 1.0)) for x, y in raw_polygon
        ]
        clamped = native_polygon != raw_polygon
        if _polygon(native_polygon) is None:
            native_polygon = None
            clamped = False
            warning = "engine_polygon_collapsed_after_native_clamp"
        else:
            normalized_polygon = [(x / (width - 1), y / (height - 1)) for x, y in native_polygon]
    accepted = bool(normalized_text) and (confidence is None or confidence >= threshold)
    return OcrText(
        line_id=f"line-{index:04d}",
        raw_text=raw_text,
        normalized_text=normalized_text,
        confidence=confidence,
        confidence_semantics=confidence_semantics,
        accepted=accepted,
        polygon_raw_xy=raw_polygon,
        polygon_xy=native_polygon,
        normalized_polygon_xy=normalized_polygon,
        polygon_clamped=clamped,
        geometry_warning=warning,
        source_order=index,
        reading_order=index,
    )


class PaddleOcrV6Reader:
    """Exactly one verified PP-OCRv6-small instance."""

    def __init__(self, engine: Any, *, threshold: float, verification: Mapping[str, Any]) -> None:
        self.engine = engine
        self.threshold = threshold
        self.verification = dict(verification)

    @classmethod
    def create(
        cls,
        *,
        config: Mapping[str, Any],
        cache_root: Path,
        constructor: Callable[..., Any] | None = None,
        package_versions: Mapping[str, str] | None = None,
    ) -> PaddleOcrV6Reader:
        verification = verify_ppocrv6(config, cache_root, package_versions=package_versions)
        model = _mapping(config["model"], "model")
        detector = verification["components"]["detector"]
        recognizer = verification["components"]["recognizer"]
        runtime_cache = (cache_root.resolve() / "ocr" / "paddlex-runtime").resolve()
        runtime_cache.mkdir(parents=True, exist_ok=True)
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(runtime_cache)
        kwargs = {
            "text_detection_model_name": detector["model_name"],
            "text_detection_model_dir": detector["path"],
            "text_recognition_model_name": recognizer["model_name"],
            "text_recognition_model_dir": recognizer["path"],
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": model["device"],
        }
        with network_forbidden():
            engine = (constructor or _default_constructor)(**kwargs)
        return cls(
            engine,
            threshold=float(model["confidence_threshold"]),
            verification=verification,
        )

    def extract(self, image: Image.Image, *, image_path: Path | None = None) -> list[OcrText]:
        if image_path is None:
            raise PaddleOcrV6Error("PP-OCRv6 requires the verified local image path")
        with network_forbidden():
            raw = _raw_detections(self.engine.predict(str(image_path)))
        return [
            _normalize_detection(item, index, image.width, image.height, self.threshold)
            for index, item in enumerate(raw)
        ]
