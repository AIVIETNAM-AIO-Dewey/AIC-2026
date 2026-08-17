from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from aic2026.contracts import RunManifest, SceneEmbeddingRecord
from aic2026.scene_embedding import l2_normalize, matrix_path_for, read_matrix, write_matrix_atomic
from pydantic import ValidationError


def _record(**overrides) -> dict:
    payload = {
        "run_id": "scene-embedding-v1",
        "video_id": "L01_V001",
        "frame_uid": "L01_V001:90",
        "keyframe_n": 2,
        "frame_idx": 90,
        "pts_time_s": 3.6,
        "fps": 25.0,
        "frame_relpath": "keyframes/L01_V001/000002.jpg",
        "width": 100,
        "height": 50,
        "row": 1,
        "embedding_dim": 8,
        "dtype": "float16",
        "l2_normalized": True,
    }
    payload.update(overrides)
    return payload


def test_record_roundtrip_keeps_schema_version() -> None:
    record = SceneEmbeddingRecord.model_validate(_record())
    assert record.schema_version == "aic26.scene_embeddings.v1"
    assert SceneEmbeddingRecord.model_validate(record.model_dump(mode="json")) == record


def test_record_rejects_unnormalized_vectors() -> None:
    with pytest.raises(ValidationError, match="L2-normalized"):
        SceneEmbeddingRecord.model_validate(_record(l2_normalized=False))


def test_record_inherits_frame_uid_validation() -> None:
    with pytest.raises(ValidationError, match="frame_uid"):
        SceneEmbeddingRecord.model_validate(_record(frame_uid="L01_V001:91"))


def test_record_rejects_unknown_dtype_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SceneEmbeddingRecord.model_validate(_record(dtype="int8"))
    with pytest.raises(ValidationError):
        SceneEmbeddingRecord.model_validate(_record(extra_field="nope"))


def test_run_manifest_accepts_the_new_stage() -> None:
    manifest = RunManifest.model_validate(
        {
            "run_id": "scene-embedding-v1",
            "stage": "scene_embeddings",
            "status": "running",
            "platform": {},
            "resolved_config": {},
            "config_sha256": "a" * 64,
            "seed": 2026,
            "started_at": datetime(2026, 8, 15, tzinfo=timezone.utc),
        }
    )
    assert manifest.stage == "scene_embeddings"


def test_matrix_path_is_derived_from_the_index_and_dtype() -> None:
    index = Path("/artifacts/scene_embeddings/L01_V001.jsonl")
    assert matrix_path_for(index, "float16").name == "L01_V001.f16.npy"
    assert matrix_path_for(index, "float32").name == "L01_V001.f32.npy"
    with pytest.raises(ValueError, match="Unsupported matrix dtype"):
        matrix_path_for(index, "int8")


def test_l2_normalize_produces_unit_rows() -> None:
    normalized = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    assert np.allclose(normalized[0], [0.6, 0.8])


def test_l2_normalize_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        l2_normalize(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError, match="zero-length row"):
        l2_normalize(np.array([[0.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        l2_normalize(np.array([[np.inf, 1.0]], dtype=np.float32))


def test_matrix_write_is_atomic_and_casts_dtype(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "L01_V001.f16.npy"
    matrix = l2_normalize(np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32))
    write_matrix_atomic(path, matrix, "float16")

    loaded = read_matrix(path)
    assert loaded.dtype == np.float16
    assert loaded.shape == (2, 2)
    assert not list(path.parent.glob("*.tmp"))
    assert np.allclose(np.linalg.norm(np.asarray(loaded, np.float32), axis=1), 1.0, atol=1e-2)
