"""Load published scene-embedding shards into a Qdrant collection."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from aic2026.common import iter_jsonl
from aic2026.contracts import SceneEmbeddingRecord

from .store import matrix_path_for, read_matrix

# Stable namespace so a re-load overwrites the same point instead of duplicating it.
POINT_NAMESPACE = uuid.UUID("6f9d1a52-8b4e-5c37-9a10-2c1d4e7f8b30")


def point_id(frame_uid: str) -> str:
    """Qdrant ids must be uint64 or UUID, and frame_uid is neither."""
    return str(uuid.uuid5(POINT_NAMESPACE, frame_uid))


def shard_paths(embeddings_dir: Path) -> list[Path]:
    return sorted(embeddings_dir.glob("*.jsonl"))


def iter_shard_points(index_path: Path) -> Iterator[tuple[str, list[float], dict[str, Any]]]:
    """Yield (id, vector, payload) for one video, matching index rows to matrix rows."""
    records = [SceneEmbeddingRecord.model_validate(raw) for raw in iter_jsonl(index_path)]
    if not records:
        raise ValueError(f"Embedding index is empty: {index_path}")
    matrix = read_matrix(matrix_path_for(index_path, records[0].dtype))
    if matrix.shape[0] != len(records):
        raise ValueError(
            f"{index_path.name}: index has {len(records)} rows but matrix has {matrix.shape[0]}"
        )
    vectors = np.asarray(matrix, dtype=np.float32)
    for record in records:
        if record.row >= len(vectors):
            raise ValueError(f"{index_path.name}: row {record.row} is outside the matrix")
        yield (
            point_id(record.frame_uid),
            vectors[record.row].tolist(),
            {
                "frame_uid": record.frame_uid,
                "video_id": record.video_id,
                "frame_idx": record.frame_idx,
                "keyframe_n": record.keyframe_n,
                "pts_time_s": record.pts_time_s,
                "fps": record.fps,
                "frame_relpath": record.frame_relpath,
                "run_id": record.run_id,
                "schema_version": record.schema_version,
            },
        )


def ensure_collection(client: Any, name: str, dim: int, *, recreate: bool) -> None:
    """Create the collection and the payload indexes the online filters need."""
    from qdrant_client import models

    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            # Vectors are already unit length, so cosine and dot rank identically;
            # cosine is kept so a future un-normalized loader cannot corrupt ranking.
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
    # TRAKE and per-video refinement both filter by video before scoring.
    for field, schema in (
        ("video_id", models.PayloadSchemaType.KEYWORD),
        ("frame_idx", models.PayloadSchemaType.INTEGER),
        ("run_id", models.PayloadSchemaType.KEYWORD),
    ):
        client.create_payload_index(
            collection_name=name, field_name=field, field_schema=schema, wait=True
        )


def load_shard(client: Any, name: str, index_path: Path, *, batch_size: int = 256) -> int:
    from qdrant_client import models

    points: list[Any] = []
    loaded = 0
    for identifier, vector, payload in iter_shard_points(index_path):
        points.append(models.PointStruct(id=identifier, vector=vector, payload=payload))
        if len(points) >= batch_size:
            client.upsert(collection_name=name, points=points, wait=True)
            loaded += len(points)
            points = []
    if points:
        client.upsert(collection_name=name, points=points, wait=True)
        loaded += len(points)
    return loaded
