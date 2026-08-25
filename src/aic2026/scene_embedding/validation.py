"""Input and output validation for SigLIP2 scene embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from aic2026.common.io import iter_jsonl
from aic2026.contracts import FrameRef, SceneEmbeddingRecord

from .store import read_matrix


def safe_image_path(data_root: Path, relative: str) -> Path:
    root = data_root.resolve()
    path = (root / relative).resolve()
    if path.is_file():
        return path
    # Fallback search
    candidates = list(root.glob(f"**/{Path(relative).name}"))
    if candidates and candidates[0].is_file():
        return candidates[0]
    return path


def validate_embedding_stage_inputs(
    *,
    frame_manifest: Path,
    data_root: Path,
    video_id: str,
    limit: int | None,
) -> dict[str, int]:
    refs = [FrameRef.model_validate(raw) for raw in iter_jsonl(frame_manifest)]
    if limit is not None:
        refs = refs[:limit]
    if not refs:
        raise ValueError("Frame manifest is empty")
    uids = [ref.frame_uid for ref in refs]
    if len(uids) != len(set(uids)):
        raise ValueError("Frame manifest contains duplicate frame_uid values")
    for ref in refs:
        if ref.video_id != video_id:
            raise ValueError(
                f"Frame manifest video_id {ref.video_id!r} does not match {video_id!r}"
            )
        image_path = safe_image_path(data_root, ref.frame_relpath)
        with Image.open(image_path) as image:
            if image.size != (ref.width, ref.height):
                raise ValueError(f"Frame dimensions do not match manifest: {image_path}")
            image.verify()
    return {"frames": len(refs)}


def validate_published_embeddings(
    *,
    index_path: Path,
    matrix_path: Path,
    video_id: str,
    expected_frame_uids: list[str],
    expected_run_id: str | None = None,
    unit_norm_tolerance: float = 1e-2,
) -> dict[str, int]:
    """Validate a published shard: index/matrix agreement, order, and unit norms."""
    if not matrix_path.exists():
        raise FileNotFoundError(f"Embedding index has no companion matrix: {matrix_path}")
    matrix = read_matrix(matrix_path)
    if matrix.ndim != 2:
        raise ValueError("Embedding matrix must be two-dimensional")

    records = [SceneEmbeddingRecord.model_validate(raw) for raw in iter_jsonl(index_path)]
    if not records:
        raise ValueError("Embedding index is empty")
    if len(records) != matrix.shape[0]:
        raise ValueError(f"Index has {len(records)} rows but matrix has {matrix.shape[0]}")
    if [record.frame_uid for record in records] != expected_frame_uids:
        raise ValueError("Published index frame order/completeness differs from its input")

    dtype_name = str(matrix.dtype)
    for position, record in enumerate(records):
        if record.row != position:
            raise ValueError(f"Index row {record.row} is out of order at line {position}")
        if record.video_id != video_id:
            raise ValueError(f"Index video_id {record.video_id!r} does not match {video_id!r}")
        if expected_run_id is not None and record.run_id != expected_run_id:
            raise ValueError(f"Index run_id {record.run_id!r} does not match {expected_run_id!r}")
        if record.embedding_dim != matrix.shape[1]:
            raise ValueError("Index embedding_dim does not match the matrix width")
        if record.dtype != dtype_name:
            raise ValueError(f"Index dtype {record.dtype!r} does not match matrix {dtype_name!r}")

    norms = np.linalg.norm(np.asarray(matrix, dtype=np.float32), axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("Embedding matrix contains non-finite values")
    if np.abs(norms - 1.0).max() > unit_norm_tolerance:
        raise ValueError("Embedding matrix rows are not unit length")
    return {"frames": len(records), "embedding_dim": int(matrix.shape[1])}
