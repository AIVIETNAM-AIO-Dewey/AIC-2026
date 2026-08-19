from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from aic2026.common.io import sha256_path

from aic_backend.ingest.artifacts import (
    ArtifactFile,
    _text_points,
    discover_artifacts,
    validate_artifact,
)


def _manifest(path: Path, outputs: list[Path], *, frames: int = 1) -> Path:
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "fixture-run",
                "counters": {"frames": frames},
                "models": [{"model_id": "fixture/model", "revision": "abc123"}],
                "outputs": [
                    {"source_id": str(output), "sha256": sha256_path(output)} for output in outputs
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_siglip_index_reads_companion_npy_by_row(tmp_path: Path) -> None:
    path = tmp_path / "scene_embeddings" / "L21_V011.jsonl"
    path.parent.mkdir()
    row = {
        "video_id": "L21_V011",
        "frame_uid": "L21_V011:24925",
        "frame_idx": 24925,
        "keyframe_n": 262,
        "pts_time_s": 997.0,
        "row": 0,
        "embedding_dim": 3,
        "dtype": "float16",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    matrix_path = path.with_name("L21_V011.f16.npy")
    np.save(matrix_path, np.asarray([[1.0, 0.0, 0.0]], dtype=np.float16))
    artifact = ArtifactFile("frames_sparse", path, _manifest(path, [path, matrix_path]))

    validated = validate_artifact(artifact)

    assert validated.matrix is not None
    assert validated.matrix.shape == (1, 3)
    assert validated.rows[0]["frame_idx"] == 24925


def test_dam_artifact_expands_one_point_per_successful_region(tmp_path: Path) -> None:
    path = tmp_path / "object_regions" / "L21_V011.jsonl"
    path.parent.mkdir()
    row = {
        "video_id": "L21_V011",
        "frame_uid": "L21_V011:24925",
        "frame_idx": 24925,
        "keyframe_n": 262,
        "pts_time_s": 997.0,
        "regions": [
            {
                "region_id": "r1",
                "bbox_xyxy_px": [1, 2, 3, 4],
                "caption": {"status": "ok", "description_en": "a red shirt"},
            },
            {"region_id": "r2", "caption": {"status": "error"}},
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    artifact = ArtifactFile("regions", path, _manifest(path, [path]))

    points = list(_text_points(validate_artifact(artifact)))

    assert len(points) == 1
    assert points[0][0] == "r1"
    assert points[0][1]["frame_idx"] == 24925


def test_ocr_ingest_skips_terminal_errors_and_rejected_lines(tmp_path: Path) -> None:
    path = tmp_path / "ocr" / "L21_V011.jsonl"
    path.parent.mkdir()
    base = {
        "video_id": "L21_V011",
        "frame_uid": "L21_V011:24925",
        "frame_idx": 24925,
        "keyframe_n": 262,
        "pts_time_s": 997.0,
        "width": 1280,
        "height": 720,
        "source_image_sha256": "a" * 64,
    }
    rows = [
        {
            **base,
            "terminal_status": "success",
            "texts": [
                {"normalized_text": "non sông liền một dải", "accepted": True},
                {"normalized_text": "low confidence", "accepted": False},
            ],
        },
        {
            **base,
            "frame_uid": "L21_V011:24926",
            "frame_idx": 24926,
            "terminal_status": "error",
            "texts": [{"normalized_text": "must not index", "accepted": True}],
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    artifact = ArtifactFile("ocr", path, _manifest(path, [path], frames=2))

    points = list(_text_points(validate_artifact(artifact)))

    assert len(points) == 1
    assert points[0][2] == "non sông liền một dải"
    assert points[0][1]["folded_text"] == "non song lien mot dai"
    assert points[0][1]["ocr_line"]["normalized_text"] == "non sông liền một dải"
    assert points[0][1]["ocr_frame"]["width"] == 1280
    assert len(points[0][1]["ocr_frame"]["lines"]) == 2
    assert points[0][1]["ocr_frame"]["model_revisions"] == ["fixture/model@abc123"]


def test_nested_ocr_artifact_is_discovered_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "ocr" / "fixture-run" / "L21_V011.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "video_id": "L21_V011",
                "frame_uid": "L21_V011:1",
                "frame_idx": 1,
                "keyframe_n": 1,
                "pts_time_s": 0.04,
                "width": 16,
                "height": 9,
                "source_image_sha256": "a" * 64,
                "terminal_status": "success",
                "full_text": "đích",
                "texts": [{"line_id": "line-0000", "normalized_text": "đích"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _manifest(path, [path])

    artifacts = discover_artifacts(tmp_path)

    assert artifacts == [ArtifactFile("ocr", path, path.with_suffix(".manifest.json"))]
    assert validate_artifact(artifacts[0]).run_id == "fixture-run"


def test_ocr_discovery_fails_on_missing_duplicate_or_unsafe_layout(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "ocr" / "run" / "video.jsonl"
    missing.parent.mkdir(parents=True)
    missing.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest is missing"):
        discover_artifacts(tmp_path / "missing")

    duplicate_root = tmp_path / "duplicate"
    direct = duplicate_root / "ocr" / "video.jsonl"
    nested = duplicate_root / "ocr" / "fixture-run" / "video.jsonl"
    direct.parent.mkdir(parents=True)
    nested.parent.mkdir(parents=True)
    for path in (direct, nested):
        path.write_text("{}\n", encoding="utf-8")
        _manifest(path, [path])
    with pytest.raises(ValueError, match="Duplicate OCR artifact identity"):
        discover_artifacts(duplicate_root)

    unsafe = tmp_path / "unsafe" / "ocr" / "fixture-run" / "extra" / "video.jsonl"
    unsafe.parent.mkdir(parents=True)
    unsafe.write_text("{}\n", encoding="utf-8")
    _manifest(unsafe, [unsafe])
    with pytest.raises(ValueError, match="Unsafe OCR artifact layout"):
        discover_artifacts(tmp_path / "unsafe")


def test_ocr_structured_rejections_never_fall_back_and_line_ids_are_unique(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ocr" / "video.jsonl"
    path.parent.mkdir()
    base = {
        "video_id": "video",
        "frame_uid": "video:1",
        "frame_idx": 1,
        "keyframe_n": 1,
        "pts_time_s": 0.04,
        "width": 16,
        "height": 9,
        "source_image_sha256": "a" * 64,
        "terminal_status": "success",
        "full_text": "must not index",
    }
    path.write_text(
        json.dumps({**base, "texts": [{"line_id": "line-0", "accepted": False}]}) + "\n",
        encoding="utf-8",
    )
    artifact = ArtifactFile("ocr", path, _manifest(path, [path]))
    assert list(_text_points(validate_artifact(artifact))) == []

    path.write_text(json.dumps(base) + "\n", encoding="utf-8")
    artifact = ArtifactFile("ocr", path, _manifest(path, [path]))
    assert len(list(_text_points(validate_artifact(artifact)))) == 1

    duplicate = {
        **base,
        "texts": [
            {"line_id": "same", "normalized_text": "a", "accepted": True},
            {"line_id": "same", "normalized_text": "b", "accepted": True},
        ],
    }
    path.write_text(json.dumps(duplicate) + "\n", encoding="utf-8")
    artifact = ArtifactFile("ocr", path, _manifest(path, [path]))
    with pytest.raises(ValueError, match="duplicate OCR line_id"):
        list(_text_points(validate_artifact(artifact)))


def test_legacy_asr_gets_checksummed_staging_receipt(tmp_path: Path) -> None:
    path = tmp_path / "asr_segments" / "L21_V011.jsonl"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "segment_id": "L21_V011:0",
                "video_id": "L21_V011",
                "start_ms": 0,
                "end_ms": 1000,
                "transcript_raw": "xin chào",
                "transcript_normalized": "xin chào",
                "keyframes": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "aic26.asr_manifest.v1",
                "status": "completed",
                "segment_count": 1,
                "model_id": "vinai/PhoWhisper-large",
                "config": {"model_revision": "fixture-revision"},
            }
        ),
        encoding="utf-8",
    )

    validated = validate_artifact(ArtifactFile("asr", path, manifest_path))

    receipt = tmp_path / ".ingest-staging" / "asr" / "L21_V011.manifest.json"
    assert receipt.is_file()
    assert validated.run_id.startswith("legacy-")
