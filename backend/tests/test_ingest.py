from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from aic2026.common.io import sha256_path

from aic_backend.ingest.artifacts import (
    ArtifactFile,
    _text_points,
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
