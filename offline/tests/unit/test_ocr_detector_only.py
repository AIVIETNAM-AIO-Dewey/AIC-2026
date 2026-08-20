from __future__ import annotations

import hashlib
import inspect
import math
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aic2026.ocr.detector_only as detector_only
import numpy as np
import pytest
import yaml
from aic2026.ocr.detector_only import (
    DETECTOR_REVISION,
    DetectorOnlyError,
    PaddleOcrV6Detector,
    parse_detector_output,
    verify_detector_only,
)
from aic2026.ocr.ppocrv6 import network_forbidden
from PIL import Image

PACKAGES = {
    "paddlepaddle": "3.3.1",
    "paddleocr": "3.7.0",
    "paddlex": "3.7.2",
    "pyclipper": "1.4.0",
    "opencv-contrib-python": "4.10.0.84",
    "Pillow": "11.1.0",
    "numpy": "1.26.4",
}
GPU_PACKAGES = {
    "paddlepaddle-gpu": "3.3.1",
    **{name: version for name, version in PACKAGES.items() if name != "paddlepaddle"},
}
GPU_RUNTIME_EVIDENCE = {
    "device": "gpu:0",
    "cuda_build": "12.6",
    "cudnn_version": 91000,
    "gpu_device_count": 1,
    "gpu_device_name": "Kaggle test GPU",
    "gpu_kernel_probe_passed": True,
    "paddlex_device_fallback_disabled": True,
}


def _create_detector(**kwargs: Any) -> PaddleOcrV6Detector:
    gpu_profile = kwargs["config"]["execution_profile"] == "kaggle_gpu_pinned"
    return PaddleOcrV6Detector._create_for_test(
        **kwargs,
        package_versions=GPU_PACKAGES if gpu_profile else PACKAGES,
        opencv_providers=["opencv-contrib-python"],
        paddle_providers=["paddlepaddle-gpu" if gpu_profile else "paddlepaddle"],
        gpu_runtime_probe=(lambda _device: GPU_RUNTIME_EVIDENCE) if gpu_profile else None,
    )


def _identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def detector_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    model_dir = tmp_path / "ocr" / "ppocrv6-small" / "detector"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "inference.pdiparams"
    model_file.write_bytes(b"detector fixture")
    config = {
        "execution_profile": "cpu_pinned",
        "model": {
            "id": "ppocrv6-small-det",
            "candidate_id": "ppocrv6-small-det-gpt4o-mini-high-v1",
            "revision": DETECTOR_REVISION,
            "source_registry_sha256": detector_only.PINNED_SOURCE_REGISTRY_SHA256,
            "device": "cpu",
            "enable_mkldnn": False,
            "download_allowed": False,
            "fallback": False,
            "ensemble": False,
            "network_policy": "execution_environment_internet_disabled_plus_python_best_effort",
            "packages": PACKAGES,
            "detector": {
                "model_name": "PP-OCRv6_small_det",
                "path": "ocr/ppocrv6-small/detector",
                "files": [_identity(model_file)],
            },
        },
    }
    monkeypatch.setattr(
        detector_only,
        "PINNED_DETECTOR_PORT_CONFIG_SHA256",
        detector_only._canonical_hash(config["model"]),
    )
    return config


def gpu_detector_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    config = detector_config(tmp_path, monkeypatch)
    config["execution_profile"] = "kaggle_gpu_pinned"
    config["model"]["device"] = "gpu:0"
    config["model"]["packages"] = GPU_PACKAGES
    monkeypatch.setattr(
        detector_only,
        "PINNED_KAGGLE_GPU_DETECTOR_PORT_CONFIG_SHA256",
        detector_only._canonical_hash(config["model"]),
    )
    return config


class FakeEngine:
    def __init__(self, *, attempts_network: bool = False) -> None:
        self.attempts_network = attempts_network

    def predict(self, pixels: np.ndarray) -> list[dict[str, Any]]:
        assert pixels.shape == (9, 12, 3)
        if self.attempts_network:
            socket.create_connection(("example.com", 80))
        return [
            {
                "res": {
                    "dt_polys": [
                        [[-2, 1], [12, 2], [11, 7], [-1, 6]],
                        [[2, 2], [8, 2], [8, 5], [2, 5]],
                    ],
                    "dt_scores": [0.91, 0.73],
                }
            }
        ]


def test_detector_constructs_without_recognizer_and_parses_pinned_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = detector_config(tmp_path, monkeypatch)
    kwargs_seen: dict[str, Any] = {}

    def constructor(**kwargs: Any) -> FakeEngine:
        kwargs_seen.update(kwargs)
        return FakeEngine()

    detector = _create_detector(
        config=config,
        cache_root=tmp_path,
        constructor=constructor,
    )
    detections = detector.detect(np.zeros((9, 12, 3), dtype=np.uint8), width=12, height=9)

    assert kwargs_seen == {
        "model_name": "PP-OCRv6_small_det",
        "model_dir": detector.verification["model_path"],
        "device": "cpu",
        "enable_mkldnn": False,
    }
    assert Path(kwargs_seen["model_dir"]).is_relative_to(tmp_path / "ocr" / "model-snapshots")
    assert detector.verification["model_snapshot_verified"] is True
    assert not any("recogni" in name.casefold() for name in kwargs_seen)
    assert not (tmp_path / "ocr" / "ppocrv6-small" / "recognizer").exists()
    assert [item.score for item in detections] == [0.91, 0.73]
    assert detections[0].clamped is True
    assert detections[0].points[0] == (0.0, 1.0)

    with pytest.raises(DetectorOnlyError, match="sealed"):
        detector.engine = FakeEngine()
    with pytest.raises(DetectorOnlyError, match="sealed"):
        detector.verification = {"detector_tree_sha256": "0" * 64}
    with pytest.raises(TypeError):
        detector.verification["detector_tree_sha256"] = "0" * 64


def test_kaggle_gpu_profile_constructs_on_gpu_with_no_fallback_and_probe_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = gpu_detector_config(tmp_path, monkeypatch)
    kwargs_seen: dict[str, Any] = {}

    def constructor(**kwargs: Any) -> FakeEngine:
        kwargs_seen.update(kwargs)
        assert os.environ["PADDLE_PDX_DISABLE_DEVICE_FALLBACK"] == "True"
        return FakeEngine()

    detector = _create_detector(
        config=config,
        cache_root=tmp_path,
        constructor=constructor,
    )

    assert kwargs_seen["device"] == "gpu:0"
    assert kwargs_seen["enable_mkldnn"] is False
    assert dict(detector.verification["gpu_runtime"]) == {
        **GPU_RUNTIME_EVIDENCE,
        "probe_boundaries": ("before_constructor", "after_constructor"),
    }
    assert detector.verification["paddle_provider"] == "paddlepaddle-gpu"
    assert detector.verification["execution_profile"] == "kaggle_gpu_pinned"


def test_gpu_runtime_probe_executes_cuda_kernel_and_rejects_cpu_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = "cpu"

    class FakeTensor:
        def __init__(self, value: float) -> None:
            self.value = value

        def __add__(self, value: float) -> FakeTensor:
            return FakeTensor(self.value + value)

        def numpy(self) -> np.ndarray:
            return np.asarray([self.value], dtype=np.float32)

    def set_device(device: str) -> None:
        nonlocal selected
        selected = device

    fake_paddle = SimpleNamespace(
        device=SimpleNamespace(
            is_compiled_with_cuda=lambda: True,
            get_cudnn_version=lambda: 91000,
            get_device=lambda: selected,
            cuda=SimpleNamespace(
                device_count=lambda: 1,
                get_device_name=lambda _index: "Kaggle test GPU",
            ),
        ),
        version=SimpleNamespace(cuda=lambda: "12.6"),
        set_device=set_device,
        ones=lambda _shape, dtype: FakeTensor(1.0) if dtype == "float32" else None,
    )
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    assert detector_only._verify_gpu_runtime("gpu:0") == GPU_RUNTIME_EVIDENCE

    fake_paddle.device.is_compiled_with_cuda = lambda: False
    with pytest.raises(DetectorOnlyError, match="not compiled with CUDA"):
        detector_only._verify_gpu_runtime("gpu:0")


def test_direct_fake_engine_and_caller_supplied_evidence_cannot_construct_detector() -> None:
    with pytest.raises(DetectorOnlyError, match="direct detector construction is forbidden"):
        PaddleOcrV6Detector(
            FakeEngine(),
            verification={
                "detector_id": "PP-OCRv6_small_det",
                "detector_revision": "0" * 64,
                "detector_tree_sha256": "0" * 64,
                "runtime_identity_sha256": "0" * 64,
                "model_snapshot_verified": True,
            },
        )


def test_detector_preflight_rejects_tamper_before_constructor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = detector_config(tmp_path, monkeypatch)
    (tmp_path / "ocr" / "ppocrv6-small" / "detector" / "inference.pdiparams").write_bytes(
        b"tampered"
    )
    constructed = False

    def constructor(**_kwargs: Any) -> object:
        nonlocal constructed
        constructed = True
        return object()

    with pytest.raises(DetectorOnlyError, match="checksum mismatch"):
        _create_detector(
            config=config,
            cache_root=tmp_path,
            constructor=constructor,
        )
    assert constructed is False


def test_constructor_and_inference_network_are_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = detector_config(tmp_path, monkeypatch)

    def network_constructor(**_kwargs: Any) -> object:
        socket.create_connection(("example.com", 80))

    with pytest.raises(RuntimeError, match="network access"):
        _create_detector(
            config=config,
            cache_root=tmp_path,
            constructor=network_constructor,
        )
    detector = _create_detector(
        config=config,
        cache_root=tmp_path,
        constructor=lambda **_kwargs: FakeEngine(attempts_network=True),
    )
    with pytest.raises(RuntimeError, match="network access"):
        detector.detect(np.zeros((9, 12, 3), dtype=np.uint8), width=12, height=9)


@pytest.mark.parametrize(
    "route",
    ["connect", "connect_ex", "create_connection", "sendto", "sendmsg"],
)
def test_python_network_guard_blocks_documented_socket_routes(route: str) -> None:
    with network_forbidden(), pytest.raises(RuntimeError, match="network access"):
        if route == "create_connection":
            socket.create_connection(("127.0.0.1", 9))
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as guarded:
                if route == "connect":
                    guarded.connect(("127.0.0.1", 9))
                elif route == "connect_ex":
                    guarded.connect_ex(("127.0.0.1", 9))
                elif route == "sendto":
                    guarded.sendto(b"x", ("127.0.0.1", 9))
                else:
                    guarded.sendmsg([b"x"], [], 0, ("127.0.0.1", 9))


def test_detector_parser_fails_closed_on_schema_or_geometry_drift() -> None:
    with pytest.raises(DetectorOnlyError, match="exactly one result"):
        parse_detector_output([], width=10, height=10)
    with pytest.raises(DetectorOnlyError, match="exactly one result"):
        parse_detector_output(
            [
                {"res": {"dt_polys": [], "dt_scores": []}},
                {"res": {"dt_polys": [], "dt_scores": []}},
            ],
            width=10,
            height=10,
        )


def test_detector_accepts_near_vertical_and_canonicalizes_paddlex_permutations() -> None:
    raw = ((122, 94), (157, 88), (178, 206), (143, 212))
    variants = [raw[offset:] + raw[:offset] for offset in range(4)] + [
        tuple(reversed(raw[offset:] + raw[:offset])) for offset in range(4)
    ]
    parsed = []
    for variant in variants:
        detection = parse_detector_output(
            [{"res": {"dt_polys": [variant], "dt_scores": [0.9]}}],
            width=300,
            height=300,
        )[0]
        parsed.append(detection.points)
        assert detection.raw_points == tuple((float(x), float(y)) for x, y in variant)
    assert parsed == [parsed[0]] * len(parsed)
    assert parsed[0] == (
        (157.0, 88.0),
        (178.0, 206.0),
        (143.0, 212.0),
        (122.0, 94.0),
    )
    assert (
        parse_detector_output([{"res": {"dt_polys": [], "dt_scores": []}}], width=10, height=10)
        == []
    )
    with pytest.raises(DetectorOnlyError, match="dt_polys and dt_scores"):
        parse_detector_output([{"res": {"polys": [], "scores": []}}], width=10, height=10)
    with pytest.raises(DetectorOnlyError, match="lengths differ"):
        parse_detector_output(
            [{"res": {"dt_polys": [[[0, 0], [1, 0], [1, 1], [0, 1]]], "dt_scores": []}}],
            width=10,
            height=10,
        )


def test_clipped_hull_repro_and_all_edge_corner_angle_permutations() -> None:
    required = ((0, 45), (35, 100), (21, 100), (0, 59))
    variants = [required[offset:] + required[:offset] for offset in range(4)] + [
        tuple(reversed(required[offset:] + required[:offset])) for offset in range(4)
    ]
    parsed = [
        parse_detector_output(
            [{"res": {"dt_polys": [variant], "dt_scores": [0.9]}}],
            width=36,
            height=101,
        )[0]
        for variant in variants
    ]
    assert [item.points for item in parsed] == [parsed[0].points] * 8
    assert parsed[0].points == tuple((float(x), float(y)) for x, y in required)
    assert parsed[0].raw_points == tuple((float(x), float(y)) for x, y in required)

    centers = (
        (12, 100),
        (188, 100),
        (100, 12),
        (100, 188),
        (30, 30),
        (170, 30),
        (170, 170),
        (30, 170),
    )
    for center in centers:
        for angle in (45, -45, 60, -60, 75, -75, 80, -80, 89, -89):
            radians = math.radians(angle)
            cosine, sine = math.cos(radians), math.sin(radians)
            clipped = tuple(
                (
                    min(max(center[0] + x * cosine - y * sine, 0), 199),
                    min(max(center[1] + x * sine + y * cosine, 0), 199),
                )
                for x, y in ((-55, -14), (55, -14), (55, 14), (-55, 14))
            )
            serializations = [clipped[offset:] + clipped[:offset] for offset in range(4)]
            serializations += [tuple(reversed(value)) for value in serializations]
            canonical = [
                parse_detector_output(
                    [{"res": {"dt_polys": [value], "dt_scores": [0.9]}}],
                    width=200,
                    height=200,
                )[0].points
                for value in serializations
            ]
            assert canonical == [canonical[0]] * 8


@pytest.mark.parametrize(
    "polygon",
    [
        [[0, 0], [5, 0], [2, 1], [0, 5]],
        [[0, 0], [5, 0], [5, 0], [0, 5]],
    ],
)
def test_detector_rejects_concave_and_duplicate_quads(polygon: list[list[int]]) -> None:
    with pytest.raises(DetectorOnlyError, match="degenerate or non-convex"):
        parse_detector_output(
            [{"res": {"dt_polys": [polygon], "dt_scores": [0.9]}}],
            width=10,
            height=10,
        )
    for bad_score in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(DetectorOnlyError, match="score"):
            parse_detector_output(
                [
                    {
                        "res": {
                            "dt_polys": [[[0, 0], [4, 0], [4, 2], [0, 2]]],
                            "dt_scores": [bad_score],
                        }
                    }
                ],
                width=10,
                height=10,
            )
    for bad_coordinate in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(DetectorOnlyError, match="coordinate"):
            parse_detector_output(
                [
                    {
                        "res": {
                            "dt_polys": [[[0, 0], [4, 0], [4, bad_coordinate], [0, 2]]],
                            "dt_scores": [0.9],
                        }
                    }
                ],
                width=10,
                height=10,
            )
    with pytest.raises(DetectorOnlyError, match="quadrilaterals"):
        parse_detector_output(
            [{"res": {"dt_polys": [[[0, 0], [1, 0], [1, 1]]], "dt_scores": [0.9]}}],
            width=10,
            height=10,
        )
    with pytest.raises(DetectorOnlyError, match="degenerate"):
        parse_detector_output(
            [
                {
                    "res": {
                        "dt_polys": [[[0, 0], [1, 1], [2, 2], [3, 3]]],
                        "dt_scores": [0.9],
                    }
                }
            ],
            width=10,
            height=10,
        )


def test_production_model_mapping_hash_is_pinned_without_paddle_or_cache() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "offline" / "ocr_phase1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert (
        detector_only._canonical_hash(config["model"])
        == detector_only.PINNED_DETECTOR_PORT_CONFIG_SHA256
    )


def test_runtime_package_mismatch_and_cache_symlink_escape_fail_before_constructor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = detector_config(tmp_path, monkeypatch)
    with pytest.raises(DetectorOnlyError, match="package versions"):
        detector_only._verify_runtime_identity(
            config,
            package_versions={**PACKAGES, "Pillow": "11.2.0"},
            opencv_providers=["opencv-contrib-python"],
            paddle_providers=["paddlepaddle"],
        )
    with pytest.raises(DetectorOnlyError, match="cv2 must come only"):
        detector_only._verify_runtime_identity(
            config,
            package_versions=PACKAGES,
            opencv_providers=["opencv-python"],
            paddle_providers=["paddlepaddle"],
        )
    with pytest.raises(DetectorOnlyError, match="paddle must come only"):
        detector_only._verify_runtime_identity(
            config,
            package_versions=PACKAGES,
            opencv_providers=["opencv-contrib-python"],
            paddle_providers=["paddlepaddle-gpu"],
        )
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    runtime_parent = tmp_path / "ocr"
    runtime_parent.rename(tmp_path / "model-ocr")
    runtime_parent.symlink_to(outside, target_is_directory=True)
    config["model"]["detector"]["path"] = "model-ocr/ppocrv6-small/detector"
    monkeypatch.setattr(
        detector_only,
        "PINNED_DETECTOR_PORT_CONFIG_SHA256",
        detector_only._canonical_hash(config["model"]),
    )
    constructed = False

    def constructor(**_kwargs: Any) -> object:
        nonlocal constructed
        constructed = True
        return object()

    with pytest.raises(DetectorOnlyError, match="runtime cache escapes"):
        _create_detector(
            config=config,
            cache_root=tmp_path,
            constructor=constructor,
        )
    assert constructed is False


def test_locked_identity_is_lightweight_but_production_preflight_inspects_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "offline" / "ocr_phase1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    def absent(_distributions: tuple[str, ...]) -> dict[str, str]:
        raise DetectorOnlyError("Paddle is absent")

    monkeypatch.setattr(detector_only, "_installed_package_versions", absent)
    identity = detector_only.identity_from_locked_config(config)
    assert identity["runtime_identity_sha256"] == detector_only.RUNTIME_IDENTITY_SHA256
    assert detector_only.runtime_identity_from_config(config) == identity
    with pytest.raises(DetectorOnlyError, match="Paddle is absent"):
        detector_only.verify_runtime_identity(config)
    assert "package_versions" not in inspect.signature(verify_detector_only).parameters
    assert "package_versions" not in inspect.signature(PaddleOcrV6Detector.create).parameters
    assert "constructor" not in inspect.signature(PaddleOcrV6Detector.create).parameters


def test_preimported_paddlex_cache_fails_even_when_environment_is_changed_to_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = detector_config(tmp_path, monkeypatch)
    requested = tmp_path / "runtime" / "ocr" / "paddlex-runtime-detector"
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path / "cache-a"))
    fake_cache_module = type("FakeCache", (), {"CACHE_DIR": str(tmp_path / "cache-a")})()
    monkeypatch.setitem(sys.modules, "paddlex.utils.cache", fake_cache_module)
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(requested))
    with pytest.raises(DetectorOnlyError, match="fresh isolated detector process"):
        _create_detector(
            config=config,
            cache_root=tmp_path,
            runtime_cache_root=tmp_path / "runtime",
            constructor=lambda **_kwargs: FakeEngine(),
        )


def test_execution_profile_must_be_supported_and_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = detector_config(tmp_path, monkeypatch)
    config["execution_profile"] = "gpu_pinned"
    with pytest.raises(DetectorOnlyError, match="supported pinned execution profile"):
        detector_only.identity_from_locked_config(config)


def test_constructor_loads_private_snapshot_not_mutable_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = detector_config(tmp_path, monkeypatch)
    model_file = tmp_path / "ocr" / "ppocrv6-small" / "detector" / "inference.pdiparams"

    loaded_path: Path | None = None

    def mutating_constructor(**kwargs: Any) -> FakeEngine:
        nonlocal loaded_path
        loaded_path = Path(kwargs["model_dir"])
        model_file.write_bytes(b"changed during construction")
        return FakeEngine()

    detector = _create_detector(
        config=config,
        cache_root=tmp_path,
        runtime_cache_root=tmp_path / "writable-runtime",
        constructor=mutating_constructor,
    )
    assert loaded_path == Path(detector.verification["model_path"])
    assert loaded_path != model_file.parent
    assert (loaded_path / model_file.name).read_bytes() == b"detector fixture"


def test_read_only_model_root_and_writable_runtime_cache_are_separate_and_set_preimport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "readonly-model"
    runtime_root = tmp_path / "writable-runtime"
    config = detector_config(model_root, monkeypatch)
    expected_cache = runtime_root / "ocr" / "paddlex-runtime-detector"

    def constructor(**_kwargs: Any) -> FakeEngine:
        assert os.environ["PADDLE_PDX_CACHE_HOME"] == str(expected_cache)
        return FakeEngine()

    detector = _create_detector(
        config=config,
        cache_root=model_root,
        runtime_cache_root=runtime_root,
        constructor=constructor,
    )
    assert expected_cache.is_dir()
    assert not (model_root / "ocr" / "paddlex-runtime-detector").exists()
    snapshot = Path(detector.verification["model_path"])
    assert snapshot.is_relative_to(runtime_root)
    assert snapshot.stat().st_mode & 0o222 == 0

    monkeypatch.setitem(sys.modules, "paddlex", object())
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(expected_cache))
    with pytest.raises(DetectorOnlyError, match="imported before"):
        _create_detector(
            config=config,
            cache_root=model_root,
            runtime_cache_root=runtime_root,
            constructor=constructor,
        )


def test_phase1_package_mapping_and_cpu_gpu_production_profiles_are_exactly_pinned() -> None:
    offline_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (offline_root / "configs" / "offline" / "ocr_phase1.yaml").read_text(encoding="utf-8")
    )
    pins: dict[str, str] = {}
    for requirement in ("runtime-base.txt", "ppocrv6.txt"):
        for raw_line in (
            (offline_root / "requirements" / requirement).read_text(encoding="utf-8").splitlines()
        ):
            line = raw_line.strip()
            if not line or line.startswith(("#", "-r")) or "==" not in line:
                continue
            name, version = line.split("==", 1)
            pins[name.casefold()] = version
    expected = {name.casefold(): version for name, version in config["model"]["packages"].items()}
    assert {name: pins[name] for name in expected} == expected

    gpu = yaml.safe_load(
        (offline_root / "configs" / "offline" / "ocr_phase1_kaggle_gpu.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert gpu["schema_version"] == "aic26.ocr_phase1.config.v1"
    assert gpu["execution_profile"] == "kaggle_gpu_pinned"
    assert gpu["model"]["device"] == "gpu:0"
    assert gpu["model"]["enable_mkldnn"] is False
    assert gpu["model"]["fallback"] is False
    assert gpu["model"]["packages"] == GPU_PACKAGES
    assert (
        detector_only._canonical_hash(gpu["model"])
        == detector_only.PINNED_KAGGLE_GPU_DETECTOR_PORT_CONFIG_SHA256
    )
    identity = detector_only.identity_from_locked_config(gpu)
    assert identity["runtime_identity_sha256"] == detector_only.KAGGLE_GPU_RUNTIME_IDENTITY_SHA256
    assert identity["paddle_provider"] == "paddlepaddle-gpu"


def test_real_detector_only_smoke_when_local_pin_is_available(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "offline" / "ocr_phase1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cache_root = Path(
        os.environ.get("AIC_CACHE_ROOT", Path(__file__).resolve().parents[3] / "models")
    )
    detector_path = cache_root / config["model"]["detector"]["path"]
    if not detector_path.is_dir():
        pytest.skip("pinned local detector directory is absent")
    evidence = verify_detector_only(config, cache_root)
    detector = PaddleOcrV6Detector.create(
        config=config,
        cache_root=cache_root,
        runtime_cache_root=tmp_path,
    )
    assert detector.verification["detector_tree_sha256"] == evidence["detector_tree_sha256"]
    assert detector.verification["model_snapshot_verified"] is True
    image_path = tmp_path / "detector-smoke.png"
    Image.new("RGB", (64, 32), "white").save(image_path)
    pixels = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)[:, :, ::-1].copy()
    assert isinstance(detector.detect(pixels, width=64, height=32), list)
