from __future__ import annotations

import hashlib
import socket
from pathlib import Path
from typing import Any

import aic2026.ocr.ppocrv6 as ppocrv6
import pytest
from aic2026.ocr.ppocrv6 import (
    PINNED_CONFIGURATION_SHA256,
    PINNED_SOURCE_REGISTRY_SHA256,
    PaddleOcrV6Error,
    PaddleOcrV6Reader,
    verify_ppocrv6,
)
from PIL import Image

PACKAGES = {"paddlepaddle": "3.3.1", "paddleocr": "3.7.0", "paddlex": "3.7.2"}


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _config(cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    detector = cache_root / "ocr" / "ppocrv6-small" / "detector"
    recognizer = cache_root / "ocr" / "ppocrv6-small" / "recognizer"
    detector.mkdir(parents=True)
    recognizer.mkdir(parents=True)
    (detector / "inference.pdiparams").write_bytes(b"detector fixture")
    (recognizer / "inference.pdiparams").write_bytes(b"recognizer fixture")
    config = {
        "model": {
            "id": "ppocrv6-small",
            "source_registry_sha256": PINNED_SOURCE_REGISTRY_SHA256,
            "configuration_sha256": PINNED_CONFIGURATION_SHA256,
            "device": "cpu",
            "confidence_threshold": 0.5,
            "download_allowed": False,
            "fallback": False,
            "ensemble": False,
            "packages": PACKAGES,
            "components": {
                "detector": {
                    "model_name": "PP-OCRv6_small_det",
                    "path": "ocr/ppocrv6-small/detector",
                    "files": [_identity(detector / "inference.pdiparams")],
                },
                "recognizer": {
                    "model_name": "PP-OCRv6_small_rec",
                    "path": "ocr/ppocrv6-small/recognizer",
                    "files": [_identity(recognizer / "inference.pdiparams")],
                },
            },
        }
    }
    monkeypatch.setattr(
        ppocrv6,
        "PINNED_PORT_CONFIG_SHA256",
        ppocrv6._mapping_sha256(config["model"]),
    )
    return config


def test_preflight_verifies_files_before_constructor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    detector = tmp_path / "ocr" / "ppocrv6-small" / "detector" / "inference.pdiparams"
    detector.write_bytes(b"tampered after identity was pinned")
    called = False

    def constructor(**_kwargs: Any) -> Any:
        nonlocal called
        called = True
        return object()

    with pytest.raises(PaddleOcrV6Error, match="checksum mismatch"):
        PaddleOcrV6Reader.create(
            config=config,
            cache_root=tmp_path,
            constructor=constructor,
            package_versions=PACKAGES,
        )
    assert called is False


def test_preflight_rejects_package_or_model_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    with pytest.raises(PaddleOcrV6Error, match="package versions"):
        verify_ppocrv6(
            config,
            tmp_path,
            package_versions={**PACKAGES, "paddleocr": "3.6.0"},
        )
    file_identity = config["model"]["components"]["detector"]["files"][0]
    original_sha256 = file_identity["sha256"]
    file_identity["sha256"] = "0" * 64
    with pytest.raises(PaddleOcrV6Error, match="configuration drift"):
        verify_ppocrv6(config, tmp_path, package_versions=PACKAGES)
    file_identity["sha256"] = original_sha256
    config["model"]["id"] = "ppocrv6-medium"
    with pytest.raises(PaddleOcrV6Error, match="only the pinned"):
        verify_ppocrv6(config, tmp_path, package_versions=PACKAGES)


def test_constructor_and_predict_are_network_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)

    def constructor(**_kwargs: Any) -> Any:
        socket.create_connection(("example.com", 80))

    with pytest.raises(PaddleOcrV6Error, match="network access"):
        PaddleOcrV6Reader.create(
            config=config,
            cache_root=tmp_path,
            constructor=constructor,
            package_versions=PACKAGES,
        )


class FakeEngine:
    def predict(self, path: str) -> list[dict[str, Any]]:
        assert path.endswith("frame.jpg")
        return [
            {
                "res": {
                    "rec_texts": ["Non sông liền một dải", "rejected", "không điểm"],
                    "rec_scores": [0.856, 0.4],
                    "rec_polys": [
                        [[-2, 1], [15, 1], [15, 5], [-2, 5]],
                        [[1, 1], [5, 1], [5, 3], [1, 3]],
                        [[2, 2], [8, 2], [8, 4], [2, 4]],
                    ],
                }
            }
        ]


def test_structured_output_keeps_native_geometry_and_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    reader = PaddleOcrV6Reader.create(
        config=config,
        cache_root=tmp_path,
        constructor=lambda **_kwargs: FakeEngine(),
        package_versions=PACKAGES,
    )
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (16, 9)).save(image_path)

    lines = reader.extract(Image.open(image_path).convert("RGB"), image_path=image_path)

    assert [line.accepted for line in lines] == [True, False, True]
    assert lines[0].normalized_text == "non sông liền một dải"
    assert lines[0].polygon_clamped is True
    assert lines[0].polygon_xy is not None
    assert lines[0].polygon_xy[0] == (0.0, 1.0)
    assert lines[2].confidence is None
    assert lines[2].confidence_semantics == "not_provided"
