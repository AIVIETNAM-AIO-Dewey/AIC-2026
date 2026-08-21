from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import sys
import types
import unicodedata
from pathlib import Path

import pytest
from aic2026.common import iter_jsonl, sha256_file, write_jsonl_atomic
from aic2026.contracts import FrameRef, RepresentativeCropBinding
from aic2026.ocr.detector_only import DetectorPolygon
from aic2026.ocr.frame_snapshot import decode_canonical_frame
from aic2026.ocr.geometry import CropConfig, canonical_quad, reconstruct_crop
from aic2026.ocr.local_recognition import (
    LocalRecognitionResult,
    RecognitionEvalSample,
    RecognitionPrediction,
    VietOcrRecognizer,
    evaluate_local_recognition,
    export_l23_verified_evaluation,
    run_local_recognition_evaluation,
)
from aic2026.ocr.phase1 import (
    Phase1Identity,
    _run_detect_crop_for_test,
    run_representative_selection,
    run_tracking,
)
from aic2026.ocr.representative_recognition import (
    DEFAULT_FRAME_CACHE_MAX_BYTES,
    RepresentativeInferenceError,
    RepresentativeRecognitionReceipt,
    _CanonicalFrameLruCache,
    _LegacyTrackingIdentityView,
    _reconstruct_crop_image,
    _tracking_identity_hashes,
    merge_representative_recognition_partitions,
    recognition_execution_policy_sha256,
    run_representative_recognition,
)
from aic2026.ocr.tracking import TrackingConfig
from aic2026.ocr.trajectory_consensus import (
    TrajectoryConsensusRecord,
    build_final_ocr_artifact,
    run_trajectory_consensus,
)
from aic_backend.ingest.artifacts import ArtifactFile, _text_points, validate_artifact
from PIL import Image


class FakeRecognizer:
    model_id = "fixture-recognizer"
    model_revision = "fixture-v1"

    def __init__(self, predictions: list[str], *, crash_after: int | None = None) -> None:
        self.predictions = iter(predictions)
        self.calls = 0
        self.crash_after = crash_after

    def predict(self, image: Image.Image) -> RecognitionPrediction:
        assert image.mode == "RGB"
        if self.crash_after is not None and self.calls == self.crash_after:
            raise KeyboardInterrupt
        self.calls += 1
        return RecognitionPrediction(transcript_raw=next(self.predictions), confidence=0.9)


def _png(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 8), color).save(path)
    return sha256_file(path)


def _manifest(tmp_path: Path) -> tuple[Path, Path, list[RecognitionEvalSample]]:
    crop_root = tmp_path / "crops-root"
    samples = [
        RecognitionEvalSample(
            sample_id="sample-1",
            video_id="video-1",
            crop_relpath="crops/one.png",
            crop_sha256=_png(crop_root / "crops/one.png", (255, 0, 0)),
            reference_transcript_nfc="Tiếng Việt",
        ),
        RecognitionEvalSample(
            sample_id="sample-2",
            video_id="video-1",
            crop_relpath="crops/two.png",
            crop_sha256=_png(crop_root / "crops/two.png", (0, 255, 0)),
            reference_transcript_nfc="ABC",
        ),
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl_atomic(manifest, samples)
    return manifest, crop_root, samples


def test_export_l23_verified_evaluation_is_read_only_and_hash_bound(tmp_path: Path) -> None:
    annotation_root = tmp_path / "annotations"
    state_db = annotation_root / "annotation_state.sqlite3"
    annotation_root.mkdir()
    connection = sqlite3.connect(state_db)
    connection.executescript(
        """
        CREATE TABLE base_annotations (
            annotation_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            crop_relpath TEXT NOT NULL,
            crop_sha256 TEXT NOT NULL
        );
        CREATE TABLE decisions (
            annotation_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            transcript_nfc TEXT
        );
        """
    )
    crop_path = annotation_root / "crops/video-1/crop.png"
    crop_sha256 = _png(crop_path, (1, 2, 3))
    connection.execute(
        "INSERT INTO base_annotations VALUES (?, ?, ?, ?)",
        ("verified", "video-1", "crops/video-1/crop.png", crop_sha256),
    )
    connection.execute(
        "INSERT INTO base_annotations VALUES (?, ?, ?, ?)",
        ("rejected", "video-1", "crops/video-1/crop.png", crop_sha256),
    )
    connection.execute(
        "INSERT INTO decisions VALUES (?, ?, ?)",
        ("verified", "verified", "Tiếng Việt"),
    )
    connection.execute("INSERT INTO decisions VALUES (?, ?, ?)", ("rejected", "rejected", None))
    connection.commit()
    connection.close()
    before = sha256_file(state_db)

    output = tmp_path / "evaluation.jsonl"
    summary = export_l23_verified_evaluation(
        state_db=state_db, annotation_root=annotation_root, output=output
    )

    assert summary == {
        "samples": 1,
        "videos": 1,
        "manifest_sha256": sha256_file(output),
    }
    assert sha256_file(state_db) == before
    assert list(iter_jsonl(output))[0]["reference_transcript_nfc"] == "Tiếng Việt"


def test_run_local_recognition_resume_is_exact_prefix(tmp_path: Path) -> None:
    manifest, crop_root, _ = _manifest(tmp_path)
    output = tmp_path / "results.jsonl"
    with pytest.raises(KeyboardInterrupt):
        run_local_recognition_evaluation(
            manifest=manifest,
            crop_root=crop_root,
            output=output,
            recognizer=FakeRecognizer(["Tiếng Việt"], crash_after=1),
        )
    partial = output.with_suffix(".jsonl.partial")
    assert len(list(iter_jsonl(partial))) == 1

    counts = run_local_recognition_evaluation(
        manifest=manifest,
        crop_root=crop_root,
        output=output,
        recognizer=FakeRecognizer(["ABC"]),
        resume=True,
    )

    assert counts == {"records": 2, "ok": 2, "empty": 0, "error": 0}
    assert not partial.exists()
    assert [row["sample_id"] for row in iter_jsonl(output)] == ["sample-1", "sample-2"]


def test_recognition_rejects_changed_crop_before_inference(tmp_path: Path) -> None:
    manifest, crop_root, samples = _manifest(tmp_path)
    (crop_root / samples[0].crop_relpath).write_bytes(b"changed")

    with pytest.raises(ValueError, match="crop identity mismatch"):
        run_local_recognition_evaluation(
            manifest=manifest,
            crop_root=crop_root,
            output=tmp_path / "results.jsonl",
            recognizer=FakeRecognizer([]),
        )


def test_evaluation_uses_edit_alignment_for_diacritic_recall(tmp_path: Path) -> None:
    manifest, _, samples = _manifest(tmp_path)
    results = tmp_path / "results.jsonl"
    records = [
        LocalRecognitionResult(
            sample_id=samples[0].sample_id,
            crop_sha256=samples[0].crop_sha256,
            model_id="fixture",
            model_revision="v1",
            status="ok",
            transcript_raw="XTiếng Việt",
            transcript_nfc="XTiếng Việt",
            confidence=0.9,
            latency_ms=1,
        ),
        LocalRecognitionResult(
            sample_id=samples[1].sample_id,
            crop_sha256=samples[1].crop_sha256,
            model_id="fixture",
            model_revision="v1",
            status="ok",
            transcript_raw="ABC",
            transcript_nfc="ABC",
            confidence=0.9,
            latency_ms=1,
        ),
    ]
    write_jsonl_atomic(results, records)

    report = evaluate_local_recognition(manifest=manifest, results=results, minimum_exact_match=0.8)

    assert report["exact_match"] == 0.5
    assert report["character_error_rate"] == pytest.approx(1 / 13)
    assert report["vietnamese_diacritic_recall_conservative"] == 1.0
    assert report["passed"] is False


def test_reference_transcript_requires_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "Việt")
    assert decomposed != "Việt"
    with pytest.raises(ValueError, match="NFC"):
        RecognitionEvalSample(
            sample_id="sample",
            video_id="video",
            crop_relpath="crop.png",
            crop_sha256=hashlib.sha256(b"crop").hexdigest(),
            reference_transcript_nfc=decomposed,
        )


def test_vietocr_constructor_disables_unused_torchvision_pretrained_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "vietocr.pth"
    weights.write_bytes(b"complete-local-checkpoint")
    captured: dict[str, object] = {}

    class FakeCfg:
        @staticmethod
        def load_config_from_file(path: str) -> dict[str, object]:
            assert path == str(tmp_path / "config.yml")
            return {
                "cnn": {"pretrained": True},
                "predictor": {"beamsearch": True},
            }

    class FakePredictor:
        def __init__(self, config: dict[str, object]) -> None:
            captured.update(config)

    vietocr_module = types.ModuleType("vietocr")
    vietocr_module.__path__ = []  # type: ignore[attr-defined]
    tool_module = types.ModuleType("vietocr.tool")
    tool_module.__path__ = []  # type: ignore[attr-defined]
    config_module = types.ModuleType("vietocr.tool.config")
    config_module.Cfg = FakeCfg  # type: ignore[attr-defined]
    predictor_module = types.ModuleType("vietocr.tool.predictor")
    predictor_module.Predictor = FakePredictor  # type: ignore[attr-defined]
    for name, module in {
        "vietocr": vietocr_module,
        "vietocr.tool": tool_module,
        "vietocr.tool.config": config_module,
        "vietocr.tool.predictor": predictor_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    recognizer = VietOcrRecognizer.create(
        config_path=tmp_path / "config.yml",
        weights_path=weights,
        device="cuda:0",
        expected_weights_sha256=sha256_file(weights),
    )

    assert recognizer.model_revision == sha256_file(weights)
    assert captured["weights"] == str(weights)
    assert captured["device"] == "cuda:0"
    assert captured["cnn"] == {"pretrained": False}
    assert captured["predictor"] == {"beamsearch": False}


def test_vietocr_batch_adapter_preserves_serial_text_confidence_and_order() -> None:
    class FakeProbability:
        def __init__(self, value: float) -> None:
            self.value = value

        def item(self) -> float:
            return self.value

    class FakePredictor:
        @staticmethod
        def _value(image: Image.Image) -> tuple[str, FakeProbability]:
            red = image.getpixel((0, 0))[0]
            return f"text-{red}", FakeProbability(red / 255)

        def predict(self, image: Image.Image, *, return_prob: bool):
            assert return_prob is True
            return self._value(image)

        def predict_batch(self, images: list[Image.Image], *, return_prob: bool):
            assert return_prob is True
            values = [self._value(image) for image in images]
            return [item[0] for item in values], [item[1] for item in values]

    images = [Image.new("RGB", (2, 2), (value, 0, 0)) for value in (10, 200, 30)]
    recognizer = VietOcrRecognizer(FakePredictor(), model_revision="f" * 64)

    serial = [recognizer.predict(image) for image in images]
    batched = recognizer.predict_batch(images)

    assert batched == serial
    assert [item.transcript_raw for item in batched] == ["text-10", "text-200", "text-30"]


def test_vietocr_adapter_records_nonfinite_native_confidence_as_unavailable() -> None:
    class FakePredictor:
        @staticmethod
        def predict(image: Image.Image, *, return_prob: bool):
            del image
            assert return_prob is True
            return "text", float("nan")

        @staticmethod
        def predict_batch(images: list[Image.Image], *, return_prob: bool):
            assert return_prob is True
            return ["text"] * len(images), [float("nan")] * len(images)

    image = Image.new("RGB", (2, 2), "white")
    recognizer = VietOcrRecognizer(FakePredictor(), model_revision="f" * 64)

    assert recognizer.predict(image).confidence is None
    assert recognizer.predict_batch([image, image])[0].confidence is None


class _Phase1Detector:
    def detect(self, image_bgr, *, width: int, height: int):
        del image_bgr
        raw = ((2.0, 2.0), (width - 3.0, 2.0), (width - 3.0, height - 3.0), (2.0, height - 3.0))
        return [
            DetectorPolygon(
                source_order=0,
                raw_points=raw,
                points=canonical_quad(raw),
                score=0.9,
                clamped=False,
            )
        ]


class _RepresentativeRecognizer:
    model_id = "fixture-vietocr"

    def __init__(self, revision: str, predictions: list[str | BaseException]) -> None:
        self.model_revision = revision
        self._predictions = iter(predictions)

    def predict(self, image: Image.Image) -> RecognitionPrediction:
        assert image.mode == "RGB"
        value = next(self._predictions)
        if isinstance(value, BaseException):
            raise value
        return RecognitionPrediction(transcript_raw=value, confidence=0.9)


class _BatchRepresentativeRecognizer(_RepresentativeRecognizer):
    def __init__(
        self,
        revision: str,
        predictions: list[str],
        *,
        batch_error: BaseException | None = None,
        fail_on_batch_call: int | None = None,
    ) -> None:
        super().__init__(revision, [])
        self._batch_predictions = iter(predictions)
        self.batch_error = batch_error
        self.fail_on_batch_call = fail_on_batch_call
        self.batch_calls: list[int] = []

    def predict_batch(self, images: list[Image.Image]) -> list[RecognitionPrediction]:
        self.batch_calls.append(len(images))
        if self.batch_error is not None and (
            self.fail_on_batch_call is None or len(self.batch_calls) == self.fail_on_batch_call
        ):
            raise self.batch_error
        assert all(image.mode == "RGB" for image in images)
        return [
            RecognitionPrediction(transcript_raw=next(self._batch_predictions), confidence=0.9)
            for _ in images
        ]


class _ScriptedClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class _PairClock:
    def __init__(self) -> None:
        self._values = iter([value for _ in range(20) for value in (0.0, 0.001)])

    def __call__(self) -> float:
        return next(self._values)


def _phase1_shard(
    tmp_path: Path,
    artifact_name: str = "phase1",
    *,
    frame_count: int = 2,
    legacy_tracking: bool = False,
    include_source_hashes: bool = True,
    tracking_config: TrackingConfig | None = None,
) -> tuple[Path, Path, Path, Path]:
    frame_dir = tmp_path / "frames" / "video"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames: list[FrameRef] = []
    for index in range(frame_count):
        frame_path = frame_dir / f"{index}.png"
        Image.new("RGB", (32, 18), color=(10 + index, 20, 30)).save(frame_path)
        frames.append(
            FrameRef(
                video_id="video",
                frame_uid=f"video:{index}",
                keyframe_n=index + 1,
                frame_idx=index,
                pts_time_s=index / 25,
                fps=25.0,
                frame_relpath=f"frames/video/{index}.png",
                source_image_sha256=(sha256_file(frame_path) if include_source_hashes else None),
                width=32,
                height=18,
            )
        )
    manifest = tmp_path / "frames.jsonl"
    write_jsonl_atomic(manifest, frames)
    root = tmp_path / artifact_name
    detections = root / "detections.jsonl"
    trajectories = root / "trajectories.jsonl"
    representatives = root / "representatives.jsonl"
    current_tracking = tracking_config or TrackingConfig()
    tracking = current_tracking
    if legacy_tracking:
        _, _, legacy_tracking_sha, legacy_resource_sha = _tracking_identity_hashes(current_tracking)
        tracking = _LegacyTrackingIdentityView(
            current_tracking,
            tracking_sha=legacy_tracking_sha,
            resource_sha=legacy_resource_sha,
        )
    _run_detect_crop_for_test(
        frame_manifest=manifest,
        data_root=tmp_path,
        output=detections,
        run_id="phase2-fixture",
        config_sha256="a" * 64,
        detector=_Phase1Detector(),
        crop_config=CropConfig(),
        tracking_config=tracking,
    )
    run_tracking(
        detections=detections,
        output=trajectories,
        run_id="phase2-fixture",
        config_sha256="a" * 64,
        tracking_config=tracking,
    )
    run_representative_selection(
        trajectories=trajectories,
        output=representatives,
        run_id="phase2-fixture",
        config_sha256="a" * 64,
        tracking_config=tracking,
    )
    return manifest, detections, trajectories, representatives


def test_representative_runner_accepts_lazy_phase1_manifest(
    tmp_path: Path,
) -> None:
    artifacts = _phase1_shard(tmp_path, include_source_hashes=False)
    assert all(row["source_image_sha256"] is None for row in iter_jsonl(artifacts[0]))
    bindings = [RepresentativeCropBinding.model_validate(row) for row in iter_jsonl(artifacts[3])]
    assert all(binding.source_image_sha256 for binding in bindings)

    weights = tmp_path / "model.pth"
    weights.write_bytes(b"fixture-weights")
    output = tmp_path / "lazy-results.jsonl"
    _run_representatives(
        tmp_path,
        artifacts=artifacts,
        output=output,
        recognizer=_RepresentativeRecognizer(sha256_file(weights), ["Tiếng Việt", "English"]),
    )

    assert [row["transcript_nfc"] for row in iter_jsonl(output)] == [
        "Tiếng Việt",
        "English",
    ]

    consensus = tmp_path / "lazy-consensus.jsonl"
    run_trajectory_consensus(
        trajectories=artifacts[2],
        representatives=artifacts[3],
        recognition_output=output,
        output=consensus,
        run_id="lazy-consensus",
    )
    final = tmp_path / "lazy-final.jsonl"
    build_final_ocr_artifact(
        trajectories=artifacts[2],
        consensus=consensus,
        output=final,
        run_id="lazy-final",
    )
    final_rows = list(iter_jsonl(final))
    assert [row["frame_uid"] for row in final_rows] == ["video:0", "video:1"]
    assert all(row["source_image_sha256"] for row in final_rows)


def _runtime_fixture() -> tuple[dict[str, str], dict[str, str], str]:
    packages = {"vietocr": "0.3.13", "torch": "fixture"}
    runtime = {"device": "cpu", "python": "fixture"}
    payload = json.dumps(
        {"packages": packages, "runtime": runtime},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return packages, runtime, hashlib.sha256(payload.encode()).hexdigest()


def _run_representatives(
    tmp_path: Path,
    *,
    artifacts: tuple[Path, Path, Path, Path],
    output: Path,
    recognizer: _RepresentativeRecognizer,
    tracking_config: TrackingConfig | None = None,
    resume: bool = False,
    fault_injector=None,
    commit_interval_records: int = 1,
    batch_size: int = 32,
    frame_cache_capacity: int = 8,
    frame_cache_max_bytes: int = DEFAULT_FRAME_CACHE_MAX_BYTES,
    representative_start: int = 0,
    representative_end: int | None = None,
    clock=None,
):
    manifest, detections, trajectories, representatives = artifacts
    weights = tmp_path / "model.pth"
    config = tmp_path / "model.yml"
    if not weights.exists():
        weights.write_bytes(b"fixture-weights")
    if not config.exists():
        config.write_text("vocab: 'abcđĐ '\n", encoding="utf-8")
    packages, runtime, runtime_hash = _runtime_fixture()
    return run_representative_recognition(
        frame_manifest=manifest,
        data_root=tmp_path,
        detections=detections,
        trajectories=trajectories,
        representatives=representatives,
        output=output,
        run_id="phase2-fixture",
        phase1_config_sha256="a" * 64,
        phase1_identity=Phase1Identity(),
        tracking_config=tracking_config or TrackingConfig(),
        recognizer=recognizer,
        model_config=config,
        model_weights=weights,
        expected_model_weights_sha256=sha256_file(weights),
        source_commit_sha="b" * 40,
        package_versions=packages,
        runtime=runtime,
        runtime_identity_sha256=runtime_hash,
        commit_interval_records=commit_interval_records,
        batch_size=batch_size,
        frame_cache_capacity=frame_cache_capacity,
        frame_cache_max_bytes=frame_cache_max_bytes,
        representative_start=representative_start,
        representative_end=representative_end,
        resume=resume,
        fault_injector=fault_injector,
        clock=clock or _PairClock(),
    )


def test_representative_partitions_merge_in_exact_input_order(tmp_path: Path) -> None:
    artifacts = _phase1_shard(tmp_path)
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"fixture-weights")
    revision = sha256_file(weights)
    first = tmp_path / "recognition.part-001.jsonl"
    second = tmp_path / "recognition.part-002.jsonl"

    _run_representatives(
        tmp_path,
        artifacts=artifacts,
        output=first,
        recognizer=_RepresentativeRecognizer(revision, ["Tiếng Việt"]),
        representative_start=0,
        representative_end=1,
    )
    _run_representatives(
        tmp_path,
        artifacts=artifacts,
        output=second,
        recognizer=_RepresentativeRecognizer(revision, ["English"]),
        representative_start=1,
        representative_end=2,
    )

    merged = tmp_path / "recognition.jsonl"
    counts = merge_representative_recognition_partitions(
        representatives=artifacts[3],
        partition_outputs=[second, first],
        output=merged,
    )

    assert counts["records"] == 2
    assert counts["partitions"] == 2
    assert [row["transcript_nfc"] for row in iter_jsonl(merged)] == [
        "Tiếng Việt",
        "English",
    ]
    receipt = RepresentativeRecognitionReceipt.model_validate_json(
        merged.with_suffix(".jsonl.receipt.json").read_text(encoding="utf-8")
    )
    assert receipt.status == "completed"
    assert receipt.source_total_representatives is None
    assert receipt.partition_start == 0
    assert receipt.partition_end is None
    assert receipt.total_representatives == 2


def test_representative_runner_resume_truncates_torn_tail_and_is_byte_identical(
    tmp_path: Path,
) -> None:
    artifacts = _phase1_shard(tmp_path)
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"fixture-weights")
    revision = sha256_file(weights)
    output = tmp_path / "resumed.jsonl"
    calls = 0

    def crash_after_first_commit(boundary: str) -> None:
        nonlocal calls
        assert boundary == "after_running_receipt"
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run_representatives(
            tmp_path,
            artifacts=artifacts,
            output=output,
            recognizer=_RepresentativeRecognizer(revision, ["Tiếng Việt"]),
            fault_injector=crash_after_first_commit,
        )
    partial = output.with_suffix(".jsonl.partial")
    with partial.open("ab") as stream:
        stream.write(b'{"torn":')
    _run_representatives(
        tmp_path,
        artifacts=artifacts,
        output=output,
        recognizer=_RepresentativeRecognizer(revision, ["English"]),
        resume=True,
    )

    uninterrupted = tmp_path / "uninterrupted.jsonl"
    _run_representatives(
        tmp_path,
        artifacts=artifacts,
        output=uninterrupted,
        recognizer=_RepresentativeRecognizer(revision, ["Tiếng Việt", "English"]),
    )
    assert output.read_bytes() == uninterrupted.read_bytes()
    assert [row["binding"]["representative_rank"] for row in iter_jsonl(output)] == [1, 2]
    receipt = RepresentativeRecognitionReceipt.model_validate_json(
        output.with_suffix(".jsonl.receipt.json").read_text(encoding="utf-8")
    )
    assert receipt.status == "completed"
    assert receipt.committed_records == 2


def test_representative_runner_rejects_reconstructed_crop_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _phase1_shard(tmp_path)
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"fixture-weights")
    monkeypatch.setattr(
        "aic2026.ocr.representative_recognition._reconstruct_crop_image",
        lambda image, crop: (Image.new("RGB", (1, 1)), b"tamper"),
    )
    with pytest.raises(ValueError, match="reconstructed crop checksum drift"):
        _run_representatives(
            tmp_path,
            artifacts=artifacts,
            output=tmp_path / "results.jsonl",
            recognizer=_RepresentativeRecognizer(sha256_file(weights), ["unused"]),
        )


def test_representative_inference_error_does_not_finalize(tmp_path: Path) -> None:
    artifacts = _phase1_shard(tmp_path)
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"fixture-weights")
    output = tmp_path / "results.jsonl"
    with pytest.raises(RepresentativeInferenceError, match="recognition failed"):
        _run_representatives(
            tmp_path,
            artifacts=artifacts,
            output=output,
            recognizer=_RepresentativeRecognizer(
                sha256_file(weights), [RuntimeError("model exploded")]
            ),
        )
    assert not output.exists()
    receipt = RepresentativeRecognitionReceipt.model_validate_json(
        output.with_suffix(".jsonl.receipt.json").read_text(encoding="utf-8")
    )
    assert receipt.status == "running"
    assert receipt.committed_records == 0


def test_representative_runner_accepts_only_exact_pre_cadence_tracking_identity(
    tmp_path: Path,
) -> None:
    pilot_tracking = TrackingConfig(maximum_frames_per_shard=390)
    artifacts = _phase1_shard(
        tmp_path,
        legacy_tracking=True,
        tracking_config=pilot_tracking,
    )
    _, _, legacy_tracking_sha, legacy_resource_sha = _tracking_identity_hashes(pilot_tracking)
    assert legacy_tracking_sha == "b42d0f5864a7cf01e824c828dc75486c9922c9ca131bae57e04060c40c200ded"
    assert legacy_resource_sha == "d95e9c557d9fca3059074e63683b664c2daa2fd109fb4aa111bc053c10575316"
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"fixture-weights")
    output = tmp_path / "legacy-results.jsonl"
    _run_representatives(
        tmp_path,
        artifacts=artifacts,
        output=output,
        recognizer=_RepresentativeRecognizer(sha256_file(weights), ["Tiếng Việt", "English"]),
        tracking_config=pilot_tracking,
    )
    receipt = RepresentativeRecognitionReceipt.model_validate_json(
        output.with_suffix(".jsonl.receipt.json").read_text(encoding="utf-8")
    )
    assert receipt.status == "completed"
    assert receipt.phase1_tracking_identity_mode == "legacy_without_commit_interval"

    drift_root = tmp_path / "drift"
    drift_root.mkdir()
    drift_artifacts = _phase1_shard(
        drift_root,
        artifact_name="phase1",
        legacy_tracking=True,
        tracking_config=pilot_tracking,
    )
    drift_representatives = drift_artifacts[3]
    rows = list(iter_jsonl(drift_representatives))
    rows[0]["tracking_config_sha256"] = "c" * 64
    write_jsonl_atomic(drift_representatives, rows)
    drift_weights = drift_root / "model.pth"
    drift_weights.write_bytes(b"fixture-weights")
    with pytest.raises(ValueError, match="neither current nor the exact legacy view"):
        _run_representatives(
            drift_root,
            artifacts=drift_artifacts,
            output=drift_root / "results.jsonl",
            recognizer=_RepresentativeRecognizer(sha256_file(drift_weights), ["unused", "unused"]),
            tracking_config=pilot_tracking,
        )


def test_phase2_direct_crop_image_preserves_exact_phase1_png_bytes(tmp_path: Path) -> None:
    manifest, _, _, representatives = _phase1_shard(tmp_path)
    frame = FrameRef.model_validate(next(iter_jsonl(manifest)))
    binding = RepresentativeCropBinding.model_validate(next(iter_jsonl(representatives)))
    snapshot = decode_canonical_frame(frame, tmp_path / frame.frame_relpath)

    image, payload = _reconstruct_crop_image(snapshot.image, binding.crop)
    try:
        assert payload == reconstruct_crop(snapshot.image, binding.crop)
        assert sha256_file_bytes(payload) == binding.crop.png_sha256
        assert image.mode == "RGB"
        assert image.size == (binding.crop.output_width, binding.crop.output_height)
        with Image.open(io.BytesIO(payload)) as reopened:
            reopened.load()
            assert reopened.convert("RGB").tobytes() == image.tobytes()
    finally:
        image.close()


def sha256_file_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_canonical_frame_lru_cache_hits_and_evicts_deterministically(tmp_path: Path) -> None:
    frames: list[FrameRef] = []
    for index in range(3):
        path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (16, 8), (index, 2, 3)).save(path)
        frames.append(
            FrameRef(
                video_id="video",
                frame_uid=f"video:{index}",
                keyframe_n=index + 1,
                frame_idx=index,
                pts_time_s=index / 25,
                fps=25.0,
                frame_relpath=path.name,
                source_image_sha256=sha256_file(path),
                width=16,
                height=8,
            )
        )
    cache = _CanonicalFrameLruCache(capacity=2, maximum_bytes=16 * 8 * 3 * 2)
    try:
        cache.get(frames[0], tmp_path / frames[0].frame_relpath)
        cache.get(frames[1], tmp_path / frames[1].frame_relpath)
        cache.get(frames[0], tmp_path / frames[0].frame_relpath)
        cache.get(frames[2], tmp_path / frames[2].frame_relpath)
        cache.get(frames[1], tmp_path / frames[1].frame_relpath)

        assert (cache.hits, cache.misses, cache.evictions) == (1, 4, 2)
        assert cache.resident_entries == 2
        assert cache.resident_bytes == 16 * 8 * 3 * 2
    finally:
        cache.close()
    assert cache.resident_entries == 0
    assert cache.resident_bytes == 0


def test_canonical_frame_lru_cache_does_not_retain_oversized_frame(tmp_path: Path) -> None:
    path = tmp_path / "oversized.png"
    Image.new("RGB", (16, 8), (1, 2, 3)).save(path)
    frame = FrameRef(
        video_id="video",
        frame_uid="video:0",
        keyframe_n=1,
        frame_idx=0,
        pts_time_s=0.0,
        fps=25.0,
        frame_relpath=path.name,
        source_image_sha256=sha256_file(path),
        width=16,
        height=8,
    )
    cache = _CanonicalFrameLruCache(capacity=8, maximum_bytes=1)
    first = cache.get(frame, path)
    second = cache.get(frame, path)
    try:
        assert first.resident is False
        assert second.resident is False
        assert (cache.hits, cache.misses, cache.oversized) == (0, 2, 2)
        assert cache.resident_entries == 0
        assert cache.resident_bytes == 0
    finally:
        first.image.close()
        second.image.close()
        cache.close()


def test_representative_batch_output_is_byte_identical_to_serial_output(tmp_path: Path) -> None:
    artifacts = _phase1_shard(tmp_path)
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"fixture-weights")
    revision = sha256_file(weights)
    serial_output = tmp_path / "serial.jsonl"
    batch_output = tmp_path / "batch.jsonl"

    _run_representatives(
        tmp_path,
        artifacts=artifacts,
        output=serial_output,
        recognizer=_RepresentativeRecognizer(revision, ["Tiếng Việt", "English"]),
        commit_interval_records=2,
        batch_size=1,
    )
    batch_recognizer = _BatchRepresentativeRecognizer(revision, ["Tiếng Việt", "English"])
    _run_representatives(
        tmp_path,
        artifacts=artifacts,
        output=batch_output,
        recognizer=batch_recognizer,
        commit_interval_records=2,
        batch_size=2,
        clock=_ScriptedClock([0.0, 0.002]),
    )

    assert batch_recognizer.batch_calls == [2]
    assert batch_output.read_bytes() == serial_output.read_bytes()
    batch_rows = list(iter_jsonl(batch_output))
    assert [row["transcript_nfc"] for row in batch_rows] == ["Tiếng Việt", "English"]
    receipt = RepresentativeRecognitionReceipt.model_validate_json(
        batch_output.with_suffix(".jsonl.receipt.json").read_text(encoding="utf-8")
    )
    assert receipt.inference_batches == 1
    assert receipt.frame_cache_hits == 0
    assert receipt.frame_cache_misses == 2
    assert receipt.frame_cache_evictions == 0
    assert receipt.frame_cache_oversized == 0


def test_batch_failure_writes_nothing_and_resumes_exactly(tmp_path: Path) -> None:
    artifacts = _phase1_shard(tmp_path)
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"fixture-weights")
    revision = sha256_file(weights)
    output = tmp_path / "resumed-batch.jsonl"

    with pytest.raises(RepresentativeInferenceError, match="recognition failed for batch"):
        _run_representatives(
            tmp_path,
            artifacts=artifacts,
            output=output,
            recognizer=_BatchRepresentativeRecognizer(
                revision, [], batch_error=RuntimeError("batch exploded")
            ),
            commit_interval_records=2,
            batch_size=2,
            clock=_ScriptedClock([0.0]),
        )
    partial = output.with_suffix(".jsonl.partial")
    receipt_path = output.with_suffix(".jsonl.receipt.json")
    receipt = RepresentativeRecognitionReceipt.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    assert partial.read_bytes() == b""
    assert receipt.committed_records == 0

    _run_representatives(
        tmp_path,
        artifacts=artifacts,
        output=output,
        recognizer=_BatchRepresentativeRecognizer(revision, ["Tiếng Việt", "English"]),
        commit_interval_records=2,
        batch_size=2,
        resume=True,
    )
    uninterrupted = tmp_path / "uninterrupted-batch.jsonl"
    _run_representatives(
        tmp_path,
        artifacts=artifacts,
        output=uninterrupted,
        recognizer=_BatchRepresentativeRecognizer(revision, ["Tiếng Việt", "English"]),
        commit_interval_records=2,
        batch_size=2,
    )
    assert output.read_bytes() == uninterrupted.read_bytes()


def test_batch_greater_than_one_resume_truncates_uncommitted_batch_tail(
    tmp_path: Path,
) -> None:
    artifacts = _phase1_shard(tmp_path, frame_count=4)
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"fixture-weights")
    revision = sha256_file(weights)
    output = tmp_path / "batch-tail.jsonl"

    with pytest.raises(RepresentativeInferenceError, match="second batch crashed"):
        _run_representatives(
            tmp_path,
            artifacts=artifacts,
            output=output,
            recognizer=_BatchRepresentativeRecognizer(
                revision,
                ["một", "hai"],
                batch_error=RuntimeError("second batch crashed"),
                fail_on_batch_call=2,
            ),
            commit_interval_records=3,
            batch_size=2,
        )
    partial = output.with_suffix(".jsonl.partial")
    receipt = RepresentativeRecognitionReceipt.model_validate_json(
        output.with_suffix(".jsonl.receipt.json").read_text(encoding="utf-8")
    )
    assert len(list(iter_jsonl(partial))) == 2
    assert receipt.committed_records == 0
    assert receipt.inference_batches == 0

    predictions = ["một", "hai", "ba"]
    _run_representatives(
        tmp_path,
        artifacts=artifacts,
        output=output,
        recognizer=_BatchRepresentativeRecognizer(revision, predictions.copy()),
        commit_interval_records=3,
        batch_size=2,
        resume=True,
    )
    uninterrupted = tmp_path / "batch-tail-uninterrupted.jsonl"
    _run_representatives(
        tmp_path,
        artifacts=artifacts,
        output=uninterrupted,
        recognizer=_BatchRepresentativeRecognizer(revision, predictions.copy()),
        commit_interval_records=3,
        batch_size=2,
    )
    assert output.read_bytes() == uninterrupted.read_bytes()


def test_source_mutation_after_cached_work_blocks_normal_publication(tmp_path: Path) -> None:
    artifacts = _phase1_shard(tmp_path, frame_count=3)
    manifest = artifacts[0]
    first_frame = FrameRef.model_validate(next(iter_jsonl(manifest)))
    source = tmp_path / first_frame.frame_relpath
    original = source.read_bytes()
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"fixture-weights")
    revision = sha256_file(weights)
    output = tmp_path / "source-drift.jsonl"
    mutated = False

    def mutate_after_first_commit(boundary: str) -> None:
        nonlocal mutated
        if boundary == "after_running_receipt" and not mutated:
            source.write_bytes(b"mutated-after-cache")
            mutated = True

    with pytest.raises(ValueError, match="source image changed during recognition"):
        _run_representatives(
            tmp_path,
            artifacts=artifacts,
            output=output,
            recognizer=_RepresentativeRecognizer(revision, ["một", "hai", "ba"]),
            commit_interval_records=1,
            batch_size=1,
            fault_injector=mutate_after_first_commit,
        )
    assert not output.exists()
    assert output.with_suffix(".jsonl.partial").exists()

    source.write_bytes(original)
    _run_representatives(
        tmp_path,
        artifacts=artifacts,
        output=output,
        recognizer=_RepresentativeRecognizer(revision, []),
        commit_interval_records=1,
        batch_size=1,
        resume=True,
    )
    assert output.exists()


def test_source_mutation_after_output_rename_blocks_recovery_promotion(tmp_path: Path) -> None:
    artifacts = _phase1_shard(tmp_path)
    first_frame = FrameRef.model_validate(next(iter_jsonl(artifacts[0])))
    source = tmp_path / first_frame.frame_relpath
    original = source.read_bytes()
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"fixture-weights")
    revision = sha256_file(weights)
    output = tmp_path / "orphan-output.jsonl"

    def mutate_after_rename(boundary: str) -> None:
        if boundary == "after_output_rename":
            source.write_bytes(b"mutated-after-output-rename")

    with pytest.raises(ValueError, match="source image changed during recognition"):
        _run_representatives(
            tmp_path,
            artifacts=artifacts,
            output=output,
            recognizer=_RepresentativeRecognizer(revision, ["một", "hai"]),
            fault_injector=mutate_after_rename,
        )
    receipt_path = output.with_suffix(".jsonl.receipt.json")
    assert output.exists()
    assert (
        RepresentativeRecognitionReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        ).status
        == "running"
    )

    with pytest.raises(ValueError, match="source image changed during recognition"):
        _run_representatives(
            tmp_path,
            artifacts=artifacts,
            output=output,
            recognizer=_RepresentativeRecognizer(revision, []),
            resume=True,
        )
    source.write_bytes(original)
    _run_representatives(
        tmp_path,
        artifacts=artifacts,
        output=output,
        recognizer=_RepresentativeRecognizer(revision, []),
        resume=True,
    )
    assert (
        RepresentativeRecognitionReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        ).status
        == "completed"
    )


@pytest.mark.parametrize(
    ("changed", "field"),
    [
        ({"batch_size": 1}, "batch_size"),
        ({"frame_cache_capacity": 7}, "frame_cache_capacity"),
        ({"frame_cache_max_bytes": 1024}, "frame_cache_max_bytes"),
    ],
)
def test_representative_resume_rejects_execution_policy_drift(
    tmp_path: Path, changed: dict[str, int], field: str
) -> None:
    artifacts = _phase1_shard(tmp_path)
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"fixture-weights")
    revision = sha256_file(weights)
    output = tmp_path / "policy-drift.jsonl"

    with pytest.raises(RepresentativeInferenceError):
        _run_representatives(
            tmp_path,
            artifacts=artifacts,
            output=output,
            recognizer=_BatchRepresentativeRecognizer(
                revision, [], batch_error=RuntimeError("stop before commit")
            ),
            commit_interval_records=2,
            batch_size=2,
        )
    options = {
        "batch_size": 2,
        "frame_cache_capacity": 8,
        "frame_cache_max_bytes": DEFAULT_FRAME_CACHE_MAX_BYTES,
        **changed,
    }
    with pytest.raises(ValueError, match=f"identity drift: {field}"):
        _run_representatives(
            tmp_path,
            artifacts=artifacts,
            output=output,
            recognizer=_BatchRepresentativeRecognizer(revision, ["unused", "unused"]),
            commit_interval_records=2,
            resume=True,
            **options,
        )


def test_receipt_rejects_tampered_execution_policy_hash(tmp_path: Path) -> None:
    artifacts = _phase1_shard(tmp_path)
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"fixture-weights")
    revision = sha256_file(weights)
    output = tmp_path / "policy-hash.jsonl"
    with pytest.raises(RepresentativeInferenceError):
        _run_representatives(
            tmp_path,
            artifacts=artifacts,
            output=output,
            recognizer=_BatchRepresentativeRecognizer(
                revision, [], batch_error=RuntimeError("stop before commit")
            ),
            commit_interval_records=2,
            batch_size=2,
        )
    receipt_path = output.with_suffix(".jsonl.receipt.json")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["recognition_execution_policy_sha256"] == (
        recognition_execution_policy_sha256(
            batch_size=2,
            frame_cache_capacity=8,
            frame_cache_max_bytes=DEFAULT_FRAME_CACHE_MAX_BYTES,
        )
    )
    payload["recognition_execution_policy_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="execution policy SHA-256 is inconsistent"):
        _run_representatives(
            tmp_path,
            artifacts=artifacts,
            output=output,
            recognizer=_BatchRepresentativeRecognizer(revision, ["unused", "unused"]),
            commit_interval_records=2,
            batch_size=2,
            resume=True,
        )


def _consensus_fixture(
    tmp_path: Path, predictions: list[str]
) -> tuple[tuple[Path, Path, Path, Path], Path, Path]:
    artifacts = _phase1_shard(tmp_path)
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"fixture-weights")
    recognition = tmp_path / "recognition.jsonl"
    _run_representatives(
        tmp_path,
        artifacts=artifacts,
        output=recognition,
        recognizer=_RepresentativeRecognizer(sha256_file(weights), predictions),
    )
    consensus = tmp_path / "consensus.jsonl"
    run_trajectory_consensus(
        trajectories=artifacts[2],
        representatives=artifacts[3],
        recognition_output=recognition,
        output=consensus,
        run_id="phase2-consensus-fixture",
    )
    return artifacts, recognition, consensus


@pytest.mark.parametrize(
    ("predictions", "expected_status", "expected_method", "expected_text"),
    [
        (["Tiếng Việt", "Tiếng Việt"], "accepted", "exact_agreement", "Tiếng Việt"),
        (["rank one", "rank two"], "accepted", "ranked_vote", "rank one"),
        (["", ""], "empty", "empty", ""),
    ],
)
def test_trajectory_consensus_agreement_disagreement_and_empty(
    tmp_path: Path,
    predictions: list[str],
    expected_status: str,
    expected_method: str,
    expected_text: str,
) -> None:
    _, _, consensus = _consensus_fixture(tmp_path, predictions)
    rows = [TrajectoryConsensusRecord.model_validate(row) for row in iter_jsonl(consensus)]
    assert len(rows) == 1
    assert rows[0].status == expected_status
    assert rows[0].method == expected_method
    assert rows[0].transcript_nfc == expected_text
    assert rows[0].frame_uids == ["video:0", "video:1"]


def test_final_ocr_adapter_satisfies_backend_ingest_contract(tmp_path: Path) -> None:
    artifacts, _, consensus = _consensus_fixture(tmp_path, ["Xin chào", "Xin chào"])
    output = tmp_path / "ocr" / "phase2-final-fixture" / "video.jsonl"
    counts = build_final_ocr_artifact(
        trajectories=artifacts[2],
        consensus=consensus,
        output=output,
        run_id="phase2-final-fixture",
    )
    validated = validate_artifact(ArtifactFile("ocr", output, output.with_suffix(".manifest.json")))
    points = list(_text_points(validated))

    assert counts["frames"] == 2
    assert counts["accepted_lines"] == 2
    assert [row["frame_uid"] for row in validated.rows] == ["video:0", "video:1"]
    assert [point[2] for point in points] == ["Xin chào", "Xin chào"]
    assert all(point[1]["ocr_line"]["accepted"] for point in points)


def test_final_ocr_adapter_keeps_empty_consensus_unindexed(tmp_path: Path) -> None:
    artifacts, _, consensus = _consensus_fixture(tmp_path, ["", ""])
    output = tmp_path / "ocr" / "phase2-empty-fixture" / "video.jsonl"
    build_final_ocr_artifact(
        trajectories=artifacts[2],
        consensus=consensus,
        output=output,
        run_id="phase2-empty-fixture",
    )
    validated = validate_artifact(ArtifactFile("ocr", output, output.with_suffix(".manifest.json")))
    assert list(_text_points(validated)) == []
    assert all(row["texts"][0]["accepted"] is False for row in validated.rows)


def test_final_ocr_adapter_rejects_consensus_hash_drift(tmp_path: Path) -> None:
    artifacts, _, consensus = _consensus_fixture(tmp_path, ["A", "A"])
    with consensus.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="consensus receipt input/output identity drift"):
        build_final_ocr_artifact(
            trajectories=artifacts[2],
            consensus=consensus,
            output=tmp_path / "ocr" / "tampered" / "video.jsonl",
            run_id="tampered",
        )
