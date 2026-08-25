"""Batched SigLIP2 embedding of keyframes named by a frame manifest."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from aic2026.common.io import iter_jsonl, write_jsonl_atomic
from aic2026.contracts import FrameRef, SceneEmbeddingRecord

from .store import l2_normalize, write_matrix_atomic
from .validation import safe_image_path


class ImageEmbeddingBackend(Protocol):
    def encode_images(self, images: list[Image.Image]) -> np.ndarray: ...


def _batched(refs: list[FrameRef], size: int) -> Iterator[list[FrameRef]]:
    for start in range(0, len(refs), size):
        yield refs[start : start + size]


def embed_frames(
    *,
    frame_manifest: Path,
    data_root: Path,
    output_index: Path,
    output_matrix: Path,
    run_id: str,
    backend: ImageEmbeddingBackend,
    matrix_dtype: str = "float16",
    batch_size: int = 32,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Embed one video and publish its index plus matrix."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    refs = [FrameRef.model_validate(raw) for raw in iter_jsonl(frame_manifest)]
    if limit is not None:
        refs = refs[:limit]
    if not refs:
        raise ValueError("frame manifest is empty")
    uids = [ref.frame_uid for ref in refs]
    if len(uids) != len(set(uids)):
        raise ValueError("frame manifest contains duplicate frame_uid values")

    blocks: list[np.ndarray] = []
    batches = 0
    for batch in _batched(refs, batch_size):
        images: list[Image.Image] = []
        try:
            for ref in batch:
                image_path = safe_image_path(data_root, ref.frame_relpath)
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                if image.size != (ref.width, ref.height):
                    raise ValueError(f"Frame dimensions changed since manifest: {image_path}")
                images.append(image)
            vectors = np.asarray(backend.encode_images(images), dtype=np.float32)
        finally:
            for image in images:
                image.close()
        if vectors.ndim != 2 or vectors.shape[0] != len(batch):
            raise ValueError("SigLIP2 backend returned a different number of vectors than images")
        blocks.append(vectors)
        batches += 1
        if progress:
            progress(sum(block.shape[0] for block in blocks), len(refs))

    matrix = l2_normalize(np.concatenate(blocks, axis=0))
    if matrix.shape[0] != len(refs):
        raise ValueError("Embedding matrix row count does not match the frame manifest")
    embedding_dim = int(matrix.shape[1])

    records = [
        SceneEmbeddingRecord(
            **ref.model_dump(),
            run_id=run_id,
            row=row,
            embedding_dim=embedding_dim,
            dtype=matrix_dtype,
            l2_normalized=True,
        )
        for row, ref in enumerate(refs)
    ]

    write_matrix_atomic(output_matrix, matrix, matrix_dtype)
    write_jsonl_atomic(output_index, records)
    return {"frames": len(refs), "batches": batches, "embedding_dim": embedding_dim}
