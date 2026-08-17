from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from aic2026.common import iter_jsonl, write_jsonl_atomic
from aic2026.common.frame_manifest import build_frame_refs
from aic2026.contracts import SceneEmbeddingRecord
from aic2026.scene_embedding import (
    embed_frames,
    matrix_path_for,
    read_matrix,
    validate_embedding_stage_inputs,
    validate_published_embeddings,
)
from PIL import Image

VIDEO_ID = "L01_V001"
RUN_ID = "scene-embedding-v1"


class FakeSiglip:
    """Deterministic stand-in: one axis is constant, one encodes the image's brightness."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.batch_sizes: list[int] = []

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        self.batch_sizes.append(len(images))
        vectors = np.zeros((len(images), self.dim), dtype=np.float32)
        for index, image in enumerate(images):
            vectors[index, 0] = 1.0
            vectors[index, 1] = float(np.asarray(image, dtype=np.float32).mean())
        return vectors


class WrongCountSiglip:
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        return np.ones((len(images) + 1, 4), dtype=np.float32)


def _build_dataset(tmp_path: Path, frames: int = 3) -> tuple[Path, Path]:
    """Create keyframes + map CSV, then publish a frame manifest. Returns (root, manifest)."""
    data_root = tmp_path / "data"
    frames_dir = data_root / "keyframes" / VIDEO_ID
    frames_dir.mkdir(parents=True)
    rows = ["n,pts_time,fps,frame_idx"]
    for n in range(1, frames + 1):
        shade = 40 * n
        Image.new("RGB", (64, 32), color=(shade, shade, shade)).save(frames_dir / f"{n:06d}.jpg")
        rows.append(f"{n},{(n - 1) * 3.6},25.0,{(n - 1) * 90}")
    map_csv = data_root / "map-keyframes" / f"{VIDEO_ID}.csv"
    map_csv.parent.mkdir(parents=True)
    map_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    refs = build_frame_refs(
        video_id=VIDEO_ID, map_csv=map_csv, frames_dir=frames_dir, data_root=data_root
    )
    manifest = tmp_path / "artifacts" / "frame_manifests" / f"{VIDEO_ID}.jsonl"
    write_jsonl_atomic(manifest, refs)
    return data_root, manifest


def test_end_to_end_publishes_aligned_index_and_matrix(tmp_path: Path) -> None:
    data_root, frame_manifest = _build_dataset(tmp_path)
    index = tmp_path / "artifacts" / "scene_embeddings" / f"{VIDEO_ID}.jsonl"
    matrix_path = matrix_path_for(index, "float16")
    backend = FakeSiglip()

    counters = embed_frames(
        frame_manifest=frame_manifest,
        data_root=data_root,
        output_index=index,
        output_matrix=matrix_path,
        run_id=RUN_ID,
        backend=backend,
        matrix_dtype="float16",
        batch_size=2,
    )

    assert counters == {"frames": 3, "batches": 2, "embedding_dim": 8}
    assert backend.batch_sizes == [2, 1]

    matrix = read_matrix(matrix_path)
    assert matrix.shape == (3, 8)
    assert matrix.dtype == np.float16
    assert np.allclose(np.linalg.norm(np.asarray(matrix, np.float32), axis=1), 1.0, atol=1e-2)

    records = [SceneEmbeddingRecord.model_validate(raw) for raw in iter_jsonl(index)]
    expected_uids = [f"{VIDEO_ID}:0", f"{VIDEO_ID}:90", f"{VIDEO_ID}:180"]
    assert [record.frame_uid for record in records] == expected_uids
    assert [record.row for record in records] == [0, 1, 2]
    assert {record.run_id for record in records} == {RUN_ID}
    assert {record.dtype for record in records} == {"float16"}

    # Brighter frames must stay distinguishable, i.e. rows are not accidentally identical.
    assert len({tuple(np.asarray(row, np.float32)) for row in matrix}) == 3

    summary = validate_published_embeddings(
        index_path=index,
        matrix_path=matrix_path,
        video_id=VIDEO_ID,
        expected_frame_uids=expected_uids,
        expected_run_id=RUN_ID,
    )
    assert summary == {"frames": 3, "embedding_dim": 8}


def test_validation_rejects_a_matrix_that_lost_rows(tmp_path: Path) -> None:
    data_root, frame_manifest = _build_dataset(tmp_path)
    index = tmp_path / "artifacts" / "scene_embeddings" / f"{VIDEO_ID}.jsonl"
    matrix_path = matrix_path_for(index, "float16")
    embed_frames(
        frame_manifest=frame_manifest,
        data_root=data_root,
        output_index=index,
        output_matrix=matrix_path,
        run_id=RUN_ID,
        backend=FakeSiglip(),
        matrix_dtype="float16",
        batch_size=8,
    )
    np.save(matrix_path, read_matrix(matrix_path)[:2], allow_pickle=False)

    with pytest.raises(ValueError, match="matrix has 2"):
        validate_published_embeddings(
            index_path=index,
            matrix_path=matrix_path,
            video_id=VIDEO_ID,
            expected_frame_uids=[f"{VIDEO_ID}:0", f"{VIDEO_ID}:90", f"{VIDEO_ID}:180"],
        )


def test_validation_rejects_a_reordered_index(tmp_path: Path) -> None:
    data_root, frame_manifest = _build_dataset(tmp_path)
    index = tmp_path / "artifacts" / "scene_embeddings" / f"{VIDEO_ID}.jsonl"
    matrix_path = matrix_path_for(index, "float16")
    embed_frames(
        frame_manifest=frame_manifest,
        data_root=data_root,
        output_index=index,
        output_matrix=matrix_path,
        run_id=RUN_ID,
        backend=FakeSiglip(),
        matrix_dtype="float16",
        batch_size=8,
    )

    with pytest.raises(ValueError, match="frame order/completeness"):
        validate_published_embeddings(
            index_path=index,
            matrix_path=matrix_path,
            video_id=VIDEO_ID,
            expected_frame_uids=[f"{VIDEO_ID}:90", f"{VIDEO_ID}:0", f"{VIDEO_ID}:180"],
        )


def test_missing_matrix_is_reported_rather_than_ignored(tmp_path: Path) -> None:
    data_root, frame_manifest = _build_dataset(tmp_path)
    index = tmp_path / "artifacts" / "scene_embeddings" / f"{VIDEO_ID}.jsonl"
    matrix_path = matrix_path_for(index, "float16")
    embed_frames(
        frame_manifest=frame_manifest,
        data_root=data_root,
        output_index=index,
        output_matrix=matrix_path,
        run_id=RUN_ID,
        backend=FakeSiglip(),
        matrix_dtype="float16",
        batch_size=8,
    )
    matrix_path.unlink()

    with pytest.raises(FileNotFoundError, match="no companion matrix"):
        validate_published_embeddings(
            index_path=index,
            matrix_path=matrix_path,
            video_id=VIDEO_ID,
            expected_frame_uids=[f"{VIDEO_ID}:0"],
        )


def test_backend_row_count_mismatch_fails_before_publishing(tmp_path: Path) -> None:
    data_root, frame_manifest = _build_dataset(tmp_path)
    index = tmp_path / "artifacts" / "scene_embeddings" / f"{VIDEO_ID}.jsonl"
    matrix_path = matrix_path_for(index, "float16")

    with pytest.raises(ValueError, match="different number of vectors"):
        embed_frames(
            frame_manifest=frame_manifest,
            data_root=data_root,
            output_index=index,
            output_matrix=matrix_path,
            run_id=RUN_ID,
            backend=WrongCountSiglip(),
            matrix_dtype="float16",
            batch_size=8,
        )
    assert not index.exists()
    assert not matrix_path.exists()


def test_limit_truncates_both_outputs(tmp_path: Path) -> None:
    data_root, frame_manifest = _build_dataset(tmp_path)
    index = tmp_path / "artifacts" / "scene_embeddings" / f"{VIDEO_ID}.jsonl"
    matrix_path = matrix_path_for(index, "float16")

    counters = embed_frames(
        frame_manifest=frame_manifest,
        data_root=data_root,
        output_index=index,
        output_matrix=matrix_path,
        run_id=RUN_ID,
        backend=FakeSiglip(),
        matrix_dtype="float16",
        limit=2,
    )
    assert counters["frames"] == 2
    assert read_matrix(matrix_path).shape[0] == 2
    assert len(list(iter_jsonl(index))) == 2


def test_input_validation_catches_a_resized_frame(tmp_path: Path) -> None:
    data_root, frame_manifest = _build_dataset(tmp_path)
    assert validate_embedding_stage_inputs(
        frame_manifest=frame_manifest, data_root=data_root, video_id=VIDEO_ID, limit=None
    ) == {"frames": 3}

    Image.new("RGB", (32, 16), color="white").save(
        data_root / "keyframes" / VIDEO_ID / "000002.jpg"
    )
    with pytest.raises(ValueError, match="Frame dimensions do not match manifest"):
        validate_embedding_stage_inputs(
            frame_manifest=frame_manifest, data_root=data_root, video_id=VIDEO_ID, limit=None
        )


def test_input_validation_rejects_a_foreign_video_id(tmp_path: Path) -> None:
    data_root, frame_manifest = _build_dataset(tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        validate_embedding_stage_inputs(
            frame_manifest=frame_manifest, data_root=data_root, video_id="L02_V002", limit=None
        )
