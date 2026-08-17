"""Atomic `.npy` embedding-matrix I/O and normalization helpers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MATRIX_DTYPES = {"float16": np.float16, "float32": np.float32}


def matrix_path_for(index_path: Path, dtype: str) -> Path:
    """Derive the companion matrix path from the index path and its dtype."""
    if dtype not in MATRIX_DTYPES:
        raise ValueError(f"Unsupported matrix dtype: {dtype!r}")
    tag = "f16" if dtype == "float16" else "f32"
    return index_path.with_name(f"{index_path.stem}.{tag}.npy")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return unit-length rows, computed in float32 regardless of input dtype."""
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embedding matrix must be two-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not (norms > 0).all():
        raise ValueError("embedding matrix contains a zero-length row")
    return values / norms


def write_matrix_atomic(path: Path, matrix: np.ndarray, dtype: str) -> None:
    """Publish the matrix through a temporary file so readers never see a torn write."""
    if dtype not in MATRIX_DTYPES:
        raise ValueError(f"Unsupported matrix dtype: {dtype!r}")
    values = np.ascontiguousarray(matrix, dtype=MATRIX_DTYPES[dtype])
    if values.ndim != 2:
        raise ValueError("embedding matrix must be two-dimensional")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_matrix(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)
