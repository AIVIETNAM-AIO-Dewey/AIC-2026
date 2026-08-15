from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aic2026.common import write_jsonl_atomic
from aic2026.contracts import SceneEmbeddingRecord
from aic2026.scene_embedding import write_matrix_atomic
from aic2026.scene_embedding.qdrant_index import iter_shard_points, point_id, shard_paths


def _shard(tmp_path: Path, rows: int = 3, dim: int = 4) -> Path:
    index = tmp_path / "L21_V005.jsonl"
    records = [
        SceneEmbeddingRecord(
            video_id="L21_V005",
            frame_uid=f"L21_V005:{i * 90}",
            keyframe_n=i + 1,
            frame_idx=i * 90,
            pts_time_s=float(i * 3),
            fps=30.0,
            frame_relpath=f"keyframes/L21_V005/{i + 1:03d}.jpg",
            width=1280,
            height=720,
            run_id="scene-embedding-v1",
            row=i,
            embedding_dim=dim,
            dtype="float16",
            l2_normalized=True,
        )
        for i in range(rows)
    ]
    write_jsonl_atomic(index, records)
    matrix = np.eye(rows, dim, dtype=np.float32)
    write_matrix_atomic(tmp_path / "L21_V005.f16.npy", matrix, "float16")
    return index


def test_point_id_is_stable_and_unique() -> None:
    assert point_id("L21_V005:90") == point_id("L21_V005:90")
    assert point_id("L21_V005:90") != point_id("L21_V005:91")
    assert point_id("L21_V005:90") != point_id("L21_V006:90")


def test_points_pair_each_record_with_its_matrix_row(tmp_path: Path) -> None:
    index = _shard(tmp_path)
    points = list(iter_shard_points(index))

    assert len(points) == 3
    assert [p[2]["frame_idx"] for p in points] == [0, 90, 180]
    assert [p[0] for p in points] == [point_id(f"L21_V005:{i * 90}") for i in range(3)]
    # Row i of the identity matrix must land on record i, not be shuffled.
    for i, (_, vector, payload) in enumerate(points):
        assert vector[i] == pytest.approx(1.0)
        assert payload["frame_uid"] == f"L21_V005:{i * 90}"
        assert payload["video_id"] == "L21_V005"


def test_row_count_mismatch_is_rejected(tmp_path: Path) -> None:
    index = _shard(tmp_path)
    write_matrix_atomic(tmp_path / "L21_V005.f16.npy", np.eye(2, 4, dtype=np.float32), "float16")

    with pytest.raises(ValueError, match="matrix has 2"):
        list(iter_shard_points(index))


def test_shard_paths_are_sorted_and_exclude_sidecars(tmp_path: Path) -> None:
    _shard(tmp_path)
    (tmp_path / "L21_V004.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "L21_V005.manifest.json").write_text("{}", encoding="utf-8")

    assert [p.stem for p in shard_paths(tmp_path)] == ["L21_V004", "L21_V005"]
