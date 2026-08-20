from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from aic2026.common import sha256_file, write_jsonl_atomic
from aic2026.contracts import FrameRef, OcrPhase1Checkpoint, OcrPhase1Receipt
from aic2026.ocr.checkpoint import (
    CheckpointArtifactPaths,
    publish_checkpoint,
    restore_checkpoint,
    verify_checkpoint_bundle,
)
from aic2026.ocr.detector_only import DetectorPolygon
from aic2026.ocr.geometry import CropConfig, canonical_quad
from aic2026.ocr.phase1 import (
    Phase1Identity,
    _run_detect_crop_for_test,
    receipt_path_for,
    run_representative_selection,
    run_tracking,
)
from aic2026.ocr.sharding import (
    OcrShardArtifactBundle,
    plan_frame_shards,
    verify_global_shards,
)
from aic2026.ocr.tracking import TrackingConfig
from PIL import Image

CONFIG_SHA = "a" * 64
GIT_SHA = "b" * 40
RUN_ID = "phase1-checkpoint-test"
IDENTITY = Phase1Identity()
CROP = CropConfig()
TRACKING = TrackingConfig(maximum_frames_per_shard=10)


class _Detector:
    def __init__(self, *, interrupt_call: int | None = None) -> None:
        self.interrupt_call = interrupt_call
        self.calls = 0

    def detect(self, image_bgr: np.ndarray, *, width: int, height: int) -> list[DetectorPolygon]:
        assert image_bgr.shape == (height, width, 3)
        self.calls += 1
        if self.calls == self.interrupt_call:
            raise KeyboardInterrupt
        raw = ((3.0, 3.0), (24.0, 2.0), (25.0, 12.0), (2.0, 13.0))
        return [
            DetectorPolygon(
                source_order=0,
                raw_points=raw,
                points=canonical_quad(raw),
                score=0.9,
                clamped=False,
            )
        ]


def _source_manifest(
    root: Path,
    videos: dict[str, int] | None = None,
    *,
    include_source_hashes: bool = True,
) -> Path:
    videos = videos or {"video1": 2}
    refs = []
    for video_id, count in videos.items():
        for frame_idx in range(count):
            image = root / "frames" / video_id / f"{frame_idx}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 18), (frame_idx + 1,) * 3).save(image)
            refs.append(
                FrameRef(
                    video_id=video_id,
                    frame_uid=f"{video_id}:{frame_idx}",
                    keyframe_n=frame_idx + 1,
                    frame_idx=frame_idx,
                    pts_time_s=frame_idx / 25,
                    fps=25.0,
                    frame_relpath=image.relative_to(root).as_posix(),
                    source_image_sha256=(sha256_file(image) if include_source_hashes else None),
                    width=32,
                    height=18,
                )
            )
    source = root / "source.frames.jsonl"
    write_jsonl_atomic(source, refs)
    return source


def _planned(
    root: Path,
    videos: dict[str, int] | None = None,
    maximum: int = 10,
    *,
    include_source_hashes: bool = True,
):
    source = _source_manifest(root, videos, include_source_hashes=include_source_hashes)
    tracking = TrackingConfig(maximum_frames_per_shard=maximum)
    global_manifest, manifest = plan_frame_shards(
        source_manifest=source,
        output_dir=root / "shards",
        config_sha256=CONFIG_SHA,
        tracking_config=tracking,
    )
    return source, global_manifest, manifest, tracking


def _paths(root: Path) -> CheckpointArtifactPaths:
    return CheckpointArtifactPaths(
        detections=root / "detections.jsonl",
        trajectories=root / "trajectories.jsonl",
        representatives=root / "representatives.jsonl",
    )


def _run_stage(
    *,
    root: Path,
    frame_manifest: Path,
    shard_id: str,
    shard_hash: str,
    tracking: TrackingConfig,
    stage: str,
) -> CheckpointArtifactPaths:
    artifacts = _paths(root)
    _run_detect_crop_for_test(
        frame_manifest=frame_manifest,
        data_root=frame_manifest.parents[1],
        output=artifacts.detections,
        run_id=RUN_ID,
        config_sha256=CONFIG_SHA,
        detector=_Detector(),
        crop_config=CROP,
        identity=IDENTITY,
        tracking_config=tracking,
        shard_id=shard_id,
        shard_manifest_sha256=shard_hash,
    )
    if stage in {"select", "completed"}:
        run_tracking(
            detections=artifacts.detections,
            output=artifacts.trajectories,
            run_id=RUN_ID,
            config_sha256=CONFIG_SHA,
            tracking_config=tracking,
            identity=IDENTITY,
        )
    if stage == "completed":
        run_representative_selection(
            trajectories=artifacts.trajectories,
            output=artifacts.representatives,
            run_id=RUN_ID,
            config_sha256=CONFIG_SHA,
            tracking_config=tracking,
            identity=IDENTITY,
        )
    return artifacts


def _checkpoint_kwargs(
    *,
    checkpoint_root: Path,
    artifact_root: Path,
    artifacts: CheckpointArtifactPaths,
    source: Path,
    global_manifest: Path,
    frame_manifest: Path,
    tracking: TrackingConfig,
    shard_id: str,
) -> dict:
    return {
        "checkpoint_root": checkpoint_root,
        "artifact_root": artifact_root,
        "artifacts": artifacts,
        "source_manifest": source,
        "global_manifest": global_manifest,
        "frame_manifest": frame_manifest,
        "data_root": source.parent,
        "run_id": RUN_ID,
        "config_sha256": CONFIG_SHA,
        "git_commit_sha": GIT_SHA,
        "identity": IDENTITY,
        "crop_config": CROP,
        "tracking_config": tracking,
        "shard_id": shard_id,
        "created_at": datetime(2026, 8, 19, tzinfo=UTC),
    }


def _restore_kwargs(values: dict) -> dict:
    return {key: value for key, value in values.items() if key not in {"artifacts", "created_at"}}


def _verify_kwargs(values: dict) -> dict:
    return {
        key: value
        for key, value in values.items()
        if key not in {"artifact_root", "artifacts", "created_at"}
    }


def test_partial_checkpoint_discards_uncommitted_tail_and_resumes_byte_identically(
    tmp_path: Path,
) -> None:
    source, global_path, manifest, tracking = _planned(tmp_path, include_source_hashes=False)
    shard = manifest.shards[0]
    frame_manifest = global_path.parent / shard.manifest_relpath
    interrupted_root = tmp_path / "interrupted"
    artifacts = _paths(interrupted_root)
    with pytest.raises(KeyboardInterrupt):
        _run_detect_crop_for_test(
            frame_manifest=frame_manifest,
            data_root=tmp_path,
            output=artifacts.detections,
            run_id=RUN_ID,
            config_sha256=CONFIG_SHA,
            detector=_Detector(interrupt_call=2),
            crop_config=CROP,
            tracking_config=tracking,
            shard_id=shard.shard_id,
            shard_manifest_sha256=shard.manifest_sha256,
        )
    partial = artifacts.detections.with_suffix(".jsonl.partial")
    receipt = OcrPhase1Receipt.model_validate_json(
        receipt_path_for(artifacts.detections).read_text(encoding="utf-8")
    )
    assert receipt.committed_records == 1
    with partial.open("ab") as stream:
        stream.write(b'{"uncommitted":"tail"}\n')

    checkpoint_root = tmp_path / "checkpoints" / RUN_ID / shard.shard_id
    bundle = publish_checkpoint(
        **_checkpoint_kwargs(
            checkpoint_root=checkpoint_root,
            artifact_root=interrupted_root,
            artifacts=artifacts,
            source=source,
            global_manifest=global_path,
            frame_manifest=frame_manifest,
            tracking=tracking,
            shard_id=shard.shard_id,
        )
    )
    marker = OcrPhase1Checkpoint.model_validate_json(
        (bundle / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert (marker.stage, marker.next_frame_uid) == ("detect", "video1:1")
    assert (bundle / marker.artifacts[0].payload_relpath).stat().st_size == receipt.committed_bytes

    restored_root = tmp_path / "restored"
    writable_checkpoint_root = tmp_path / "working-checkpoints" / RUN_ID / shard.shard_id
    restore_checkpoint(
        **_restore_kwargs(
            _checkpoint_kwargs(
                checkpoint_root=checkpoint_root,
                artifact_root=restored_root,
                artifacts=_paths(restored_root),
                source=source,
                global_manifest=global_path,
                frame_manifest=frame_manifest,
                tracking=tracking,
                shard_id=shard.shard_id,
            )
        ),
        checkpoint_destination_root=writable_checkpoint_root,
    )
    copied_first = next(
        path for path in writable_checkpoint_root.iterdir() if path.name.startswith("checkpoint-")
    )
    assert (copied_first / "checkpoint.json").read_bytes() == (
        bundle / "checkpoint.json"
    ).read_bytes()
    restored = _paths(restored_root)
    detector = _Detector()
    _run_detect_crop_for_test(
        frame_manifest=frame_manifest,
        data_root=tmp_path,
        output=restored.detections,
        run_id=RUN_ID,
        config_sha256=CONFIG_SHA,
        detector=detector,
        crop_config=CROP,
        tracking_config=tracking,
        shard_id=shard.shard_id,
        shard_manifest_sha256=shard.manifest_sha256,
        resume=True,
    )
    assert detector.calls == 1
    second = publish_checkpoint(
        **_checkpoint_kwargs(
            checkpoint_root=writable_checkpoint_root,
            artifact_root=restored_root,
            artifacts=restored,
            source=source,
            global_manifest=global_path,
            frame_manifest=frame_manifest,
            tracking=tracking,
            shard_id=shard.shard_id,
        )
    )
    second_marker = OcrPhase1Checkpoint.model_validate_json(
        (second / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert second_marker.checkpoint_sequence == 2
    assert second_marker.previous_checkpoint_sha256 == sha256_file(copied_first / "checkpoint.json")

    uninterrupted = _run_stage(
        root=tmp_path / "uninterrupted",
        frame_manifest=frame_manifest,
        shard_id=shard.shard_id,
        shard_hash=shard.manifest_sha256,
        tracking=tracking,
        stage="track",
    )
    assert restored.detections.read_bytes() == uninterrupted.detections.read_bytes()
    assert (
        receipt_path_for(restored.detections).read_bytes()
        == receipt_path_for(uninterrupted.detections).read_bytes()
    )


def test_stage_checkpoints_form_chain_restore_read_only_and_are_idempotent(
    tmp_path: Path,
) -> None:
    source, global_path, manifest, tracking = _planned(tmp_path)
    shard = manifest.shards[0]
    frame_manifest = global_path.parent / shard.manifest_relpath
    session1 = tmp_path / "session1"
    artifacts = _run_stage(
        root=session1,
        frame_manifest=frame_manifest,
        shard_id=shard.shard_id,
        shard_hash=shard.manifest_sha256,
        tracking=tracking,
        stage="track",
    )
    checkpoint_root = tmp_path / "checkpoints" / RUN_ID / shard.shard_id
    kwargs = _checkpoint_kwargs(
        checkpoint_root=checkpoint_root,
        artifact_root=session1,
        artifacts=artifacts,
        source=source,
        global_manifest=global_path,
        frame_manifest=frame_manifest,
        tracking=tracking,
        shard_id=shard.shard_id,
    )
    first = publish_checkpoint(**kwargs)
    assert (
        OcrPhase1Checkpoint.model_validate_json(
            (first / "checkpoint.json").read_text(encoding="utf-8")
        ).stage
        == "track"
    )
    run_tracking(
        detections=artifacts.detections,
        output=artifacts.trajectories,
        run_id=RUN_ID,
        config_sha256=CONFIG_SHA,
        tracking_config=tracking,
    )
    second = publish_checkpoint(**kwargs)
    second_marker = OcrPhase1Checkpoint.model_validate_json(
        (second / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert second_marker.stage == "select"
    assert second_marker.previous_checkpoint_sha256 == sha256_file(first / "checkpoint.json")
    run_representative_selection(
        trajectories=artifacts.trajectories,
        output=artifacts.representatives,
        run_id=RUN_ID,
        config_sha256=CONFIG_SHA,
        tracking_config=tracking,
    )
    third = publish_checkpoint(**kwargs)
    assert (
        OcrPhase1Checkpoint.model_validate_json(
            (third / "checkpoint.json").read_text(encoding="utf-8")
        ).stage
        == "completed"
    )
    assert publish_checkpoint(**kwargs) == third
    assert (
        len([path for path in checkpoint_root.iterdir() if path.name.startswith("checkpoint-")])
        == 3
    )

    changed_permissions: list[Path] = []
    try:
        for path in checkpoint_root.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
                changed_permissions.append(path)
            elif path.is_dir():
                path.chmod(0o555)
                changed_permissions.append(path)
        checkpoint_root.chmod(0o555)
        changed_permissions.append(checkpoint_root)
        session2 = tmp_path / "session2"
        restore_kwargs = _checkpoint_kwargs(
            checkpoint_root=checkpoint_root,
            artifact_root=session2,
            artifacts=_paths(session2),
            source=source,
            global_manifest=global_path,
            frame_manifest=frame_manifest,
            tracking=tracking,
            shard_id=shard.shard_id,
        )
        writable_history = tmp_path / "working-checkpoints" / RUN_ID / shard.shard_id
        restored = restore_checkpoint(
            **_restore_kwargs(restore_kwargs),
            checkpoint_destination_root=writable_history,
        )
        assert restored.stage == "completed"
        restore_checkpoint(
            **_restore_kwargs(restore_kwargs),
            checkpoint_destination_root=writable_history,
        )
        assert (
            len(
                [path for path in writable_history.iterdir() if path.name.startswith("checkpoint-")]
            )
            == 3
        )
        copied_artifacts = _paths(session2)
        for original, copied in zip(
            (artifacts.detections, artifacts.trajectories, artifacts.representatives),
            (
                copied_artifacts.detections,
                copied_artifacts.trajectories,
                copied_artifacts.representatives,
            ),
            strict=True,
        ):
            assert copied.read_bytes() == original.read_bytes()
            assert receipt_path_for(copied).read_bytes() == receipt_path_for(original).read_bytes()
    finally:
        for path in reversed(changed_permissions):
            path.chmod(0o755 if path.is_dir() else 0o644)


@pytest.mark.parametrize(
    ("boundary", "committed"),
    [
        ("after_checkpoint_files_fsync", False),
        ("before_checkpoint_commit_marker", False),
        ("after_checkpoint_marker_fsync_before_rename", False),
        ("after_checkpoint_directory_rename", True),
    ],
)
def test_checkpoint_publication_boundaries_never_expose_partial_commit_marker(
    tmp_path: Path, boundary: str, committed: bool
) -> None:
    source, global_path, manifest, tracking = _planned(tmp_path)
    shard = manifest.shards[0]
    frame_manifest = global_path.parent / shard.manifest_relpath
    artifact_root = tmp_path / "artifacts"
    artifacts = _run_stage(
        root=artifact_root,
        frame_manifest=frame_manifest,
        shard_id=shard.shard_id,
        shard_hash=shard.manifest_sha256,
        tracking=tracking,
        stage="completed",
    )

    def crash(current: str) -> None:
        if current == boundary:
            raise RuntimeError("simulated checkpoint crash")

    checkpoint_root = tmp_path / "checkpoint-boundary"
    with pytest.raises(RuntimeError, match="simulated checkpoint crash"):
        publish_checkpoint(
            **_checkpoint_kwargs(
                checkpoint_root=checkpoint_root,
                artifact_root=artifact_root,
                artifacts=artifacts,
                source=source,
                global_manifest=global_path,
                frame_manifest=frame_manifest,
                tracking=tracking,
                shard_id=shard.shard_id,
            ),
            fault_injector=crash,
        )
    published = [
        child for child in checkpoint_root.iterdir() if child.name.startswith("checkpoint-")
    ]
    assert bool(published) is committed
    assert all((child / "checkpoint.json").is_file() for child in published)
    # A marker fsynced inside a hidden temporary directory is not published:
    # restore discovery deliberately ignores the temporary namespace.
    if not committed:
        with pytest.raises(ValueError, match="no committed checkpoint"):
            verify_checkpoint_bundle(
                **_verify_kwargs(
                    _checkpoint_kwargs(
                        checkpoint_root=checkpoint_root,
                        artifact_root=artifact_root,
                        artifacts=artifacts,
                        source=source,
                        global_manifest=global_path,
                        frame_manifest=frame_manifest,
                        tracking=tracking,
                        shard_id=shard.shard_id,
                    )
                )
            )


def test_tampered_truncated_or_wrong_identity_checkpoint_is_rejected(tmp_path: Path) -> None:
    source, global_path, manifest, tracking = _planned(tmp_path)
    shard = manifest.shards[0]
    frame_manifest = global_path.parent / shard.manifest_relpath
    artifact_root = tmp_path / "artifacts"
    artifacts = _run_stage(
        root=artifact_root,
        frame_manifest=frame_manifest,
        shard_id=shard.shard_id,
        shard_hash=shard.manifest_sha256,
        tracking=tracking,
        stage="completed",
    )
    checkpoint_root = tmp_path / "checkpoints"
    kwargs = _checkpoint_kwargs(
        checkpoint_root=checkpoint_root,
        artifact_root=artifact_root,
        artifacts=artifacts,
        source=source,
        global_manifest=global_path,
        frame_manifest=frame_manifest,
        tracking=tracking,
        shard_id=shard.shard_id,
    )
    bundle = publish_checkpoint(**kwargs)
    verify_kwargs = _verify_kwargs(kwargs)
    wrong_identities = (
        {"git_commit_sha": "c" * 40},
        {"config_sha256": "d" * 64},
        {"shard_id": "shard-000002"},
        {"identity": Phase1Identity(detector_revision="d" * 64)},
        {"identity": Phase1Identity(detector_tree_sha256="d" * 64)},
        {"identity": Phase1Identity(runtime_identity_sha256="d" * 64)},
        {"tracking_config": TrackingConfig(maximum_frames_per_shard=11)},
    )
    for override in wrong_identities:
        with pytest.raises(ValueError, match="identity drift"):
            verify_checkpoint_bundle(**{**verify_kwargs, **override})

    undeclared = bundle / "rclone.conf"
    undeclared.write_text("token = secret", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="undeclared files"):
            verify_checkpoint_bundle(**verify_kwargs)
    finally:
        undeclared.unlink()

    source_image = tmp_path / "frames" / "video1" / "0.png"
    source_bytes = source_image.read_bytes()
    Image.new("RGB", (32, 18), (99, 99, 99)).save(source_image)
    with pytest.raises(ValueError, match="source image checksum drift"):
        verify_checkpoint_bundle(**verify_kwargs)
    source_image.write_bytes(source_bytes)

    marker = OcrPhase1Checkpoint.model_validate_json(
        (bundle / "checkpoint.json").read_text(encoding="utf-8")
    )
    receipt = bundle / marker.artifacts[0].receipt_relpath
    original = receipt.read_bytes()
    receipt.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="checksum drift"):
        verify_checkpoint_bundle(**verify_kwargs)
    receipt.write_bytes(original)

    payload = bundle / marker.artifacts[0].payload_relpath
    original = payload.read_bytes()
    payload.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="missing or truncated"):
        verify_checkpoint_bundle(**verify_kwargs)
    payload.write_bytes(original)

    marker_path = bundle / "checkpoint.json"
    marker_value = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_value.pop("git_commit_sha")
    marker_path.write_text(json.dumps(marker_value), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid checkpoint commit marker"):
        verify_checkpoint_bundle(**verify_kwargs)


def test_missing_marker_and_secret_material_are_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "missing-marker"
    empty.mkdir()
    (empty / "checkpoint-000001-deadbeef").mkdir()
    with pytest.raises(ValueError, match="missing its commit marker"):
        verify_checkpoint_bundle(
            checkpoint_root=empty,
            source_manifest=tmp_path / "missing",
            global_manifest=tmp_path / "missing",
            frame_manifest=tmp_path / "missing",
            data_root=tmp_path,
            run_id=RUN_ID,
            config_sha256=CONFIG_SHA,
            git_commit_sha=GIT_SHA,
            identity=IDENTITY,
            crop_config=CROP,
            tracking_config=TRACKING,
            shard_id="shard-000001",
        )

    source, global_path, manifest, tracking = _planned(tmp_path / "secret")
    shard = manifest.shards[0]
    frame_manifest = global_path.parent / shard.manifest_relpath
    secret_run = "OPENAI_API_KEY=must-not-be-exported"
    artifact_root = tmp_path / "secret" / "artifacts"
    artifacts = _paths(artifact_root)
    _run_detect_crop_for_test(
        frame_manifest=frame_manifest,
        data_root=source.parent,
        output=artifacts.detections,
        run_id=secret_run,
        config_sha256=CONFIG_SHA,
        detector=_Detector(),
        crop_config=CROP,
        tracking_config=tracking,
        shard_id=shard.shard_id,
        shard_manifest_sha256=shard.manifest_sha256,
    )
    kwargs = _checkpoint_kwargs(
        checkpoint_root=tmp_path / "secret" / "checkpoints",
        artifact_root=artifact_root,
        artifacts=artifacts,
        source=source,
        global_manifest=global_path,
        frame_manifest=frame_manifest,
        tracking=tracking,
        shard_id=shard.shard_id,
    )
    kwargs["run_id"] = secret_run
    with pytest.raises(ValueError, match="credential material"):
        publish_checkpoint(**kwargs)
    assert secret_run not in json.dumps([str(path) for path in tmp_path.rglob("*")])


def test_checkpoints_from_two_sessions_restore_and_global_verify(tmp_path: Path) -> None:
    source, global_path, manifest, tracking = _planned(
        tmp_path, {"video1": 1, "video2": 1}, maximum=1
    )
    restored_bundles: dict[str, OcrShardArtifactBundle] = {}
    for session_number, shard in enumerate(manifest.shards, start=1):
        frame_manifest = global_path.parent / shard.manifest_relpath
        session = tmp_path / f"session-{session_number}"
        artifacts = _run_stage(
            root=session,
            frame_manifest=frame_manifest,
            shard_id=shard.shard_id,
            shard_hash=shard.manifest_sha256,
            tracking=tracking,
            stage="completed",
        )
        checkpoint_root = tmp_path / "outputs" / RUN_ID / shard.shard_id
        publish_checkpoint(
            **_checkpoint_kwargs(
                checkpoint_root=checkpoint_root,
                artifact_root=session,
                artifacts=artifacts,
                source=source,
                global_manifest=global_path,
                frame_manifest=frame_manifest,
                tracking=tracking,
                shard_id=shard.shard_id,
            )
        )
        restored_root = tmp_path / "new-session" / shard.shard_id
        restore_checkpoint(
            **_restore_kwargs(
                _checkpoint_kwargs(
                    checkpoint_root=checkpoint_root,
                    artifact_root=restored_root,
                    artifacts=_paths(restored_root),
                    source=source,
                    global_manifest=global_path,
                    frame_manifest=frame_manifest,
                    tracking=tracking,
                    shard_id=shard.shard_id,
                )
            )
        )
        restored = _paths(restored_root)
        restored_bundles[shard.shard_id] = OcrShardArtifactBundle(
            detections=restored.detections,
            trajectories=restored.trajectories,
            representatives=restored.representatives,
        )
    assert (
        verify_global_shards(
            source_manifest=source,
            global_manifest=global_path,
            expected_config_sha256=CONFIG_SHA,
            tracking_config=tracking,
            expected_run_id=RUN_ID,
            expected_identity=IDENTITY,
            shard_bundles=restored_bundles,
            data_root=tmp_path,
            crop_config=CROP,
        )["shards"]
        == 2
    )


def test_cli_exposes_read_only_resume_and_checkpoint_export() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_ocr_phase1.py"
    spec = importlib.util.spec_from_file_location("run_ocr_phase1_checkpoint_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(
        [
            "run",
            "--resume-from",
            "/kaggle/input/old-output/checkpoints/run/shard-000001",
            "--checkpoint-root",
            "/kaggle/working/checkpoints/run/shard-000001",
        ]
    )
    assert args.resume_from is not None
    assert args.checkpoint_root is not None
    with pytest.raises(SystemExit, match="writable storage"):
        module.main(
            [
                "run",
                "--resume-from",
                "/kaggle/input/old-output/checkpoints/run/shard-000001",
                "--output-root",
                "/kaggle/working/ocr/run/shard-000001",
            ]
        )
