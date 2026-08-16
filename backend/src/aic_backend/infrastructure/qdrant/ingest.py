"""Validated, idempotent artifact ingestion and atomic alias activation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from aic2026.common.io import sha256_path

from ..encoders.sparse import sparse_vector
from .ids import point_id


@dataclass(frozen=True)
class ArtifactFile:
    collection: str
    path: Path
    manifest_path: Path


def _records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def validate_artifact(artifact: ArtifactFile) -> list[dict]:
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"{artifact.path}: manifest is not completed")
    if sha256_path(artifact.path) not in {
        item.get("sha256") for item in manifest.get("outputs", [])
    }:
        raise ValueError(f"{artifact.path}: checksum does not match manifest")
    rows = list(_records(artifact.path))
    expected = manifest.get("counters", {}).get("frames") or manifest.get("counters", {}).get(
        "records"
    )
    if expected is not None and int(expected) != len(rows):
        raise ValueError(f"{artifact.path}: record count does not match manifest")
    for row in rows:
        for field in ("video_id", "frame_idx", "pts_time_s", "run_id"):
            if field not in row:
                raise ValueError(f"{artifact.path}: record missing {field}")
    return rows


def discover_artifacts(root: Path) -> list[ArtifactFile]:
    mapping = {
        "scene_embeddings": "frames_sparse",
        "dense_scene_embeddings": "frames_dense",
        "object_regions": "regions",
        "ocr": "ocr",
        "asr_segments": "asr",
    }
    found: list[ArtifactFile] = []
    for dirname, collection in mapping.items():
        for path in sorted((root / dirname).glob("*.jsonl")) if (root / dirname).exists() else []:
            manifest = path.with_suffix(".manifest.json")
            if manifest.exists():
                found.append(ArtifactFile(collection, path, manifest))
    return found


def _text_for(collection: str, row: dict) -> str:
    if collection == "regions":
        return " ".join(
            item.get("caption", {}).get("description_en", "") for item in row.get("regions", [])
        ).strip()
    if collection == "ocr":
        return " ".join(
            item.get("normalized_text", item.get("raw_text", "")) for item in row.get("texts", [])
        )
    return (
        row.get("transcript_normalized", "") if collection == "asr" else row.get("scene_text", "")
    )


def ingest(
    client, artifacts: list[ArtifactFile], *, dense_encoder, activate: bool
) -> dict[str, int]:  # type: ignore[no-untyped-def]
    from qdrant_client import models

    versions: dict[str, str] = {}
    counts: dict[str, int] = {}
    for artifact in artifacts:
        rows = validate_artifact(artifact)
        version = f"{artifact.collection}_{uuid4().hex[:12]}"
        vectors = (
            {"scene": models.VectorParams(size=768, distance=models.Distance.COSINE)}
            if artifact.collection.startswith("frames_")
            else {"dense": models.VectorParams(size=768, distance=models.Distance.COSINE)}
        )
        sparse = (
            None
            if artifact.collection.startswith("frames_")
            else {"lexical": models.SparseVectorParams()}
        )
        client.create_collection(
            collection_name=version, vectors_config=vectors, sparse_vectors_config=sparse
        )
        points = []
        for index, row in enumerate(rows):
            payload = {
                key: row.get(key)
                for key in ("video_id", "frame_idx", "pts_time_s", "keyframe_n", "run_id")
            }
            payload.update(
                text=_text_for(artifact.collection, row), source_artifact=artifact.path.name
            )
            source_id = (
                row.get("frame_uid")
                or row.get("segment_id")
                or f"{payload['video_id']}:{payload['frame_idx']}:{index}"
            )
            if artifact.collection.startswith("frames_"):
                vector = row.get("embedding")
                if vector is None:
                    raise ValueError(f"{artifact.path}: frame record has no inline embedding")
                vector_payload = {"scene": vector}
            else:
                text = payload["text"]
                if not text:
                    continue
                dense = dense_encoder.encode([text], query=False)[0].tolist()
                indices, values = sparse_vector(text)
                vector_payload = {
                    "dense": dense,
                    "lexical": models.SparseVector(indices=indices, values=values),
                }
            points.append(
                models.PointStruct(
                    id=str(point_id(collection=artifact.collection, source_id=str(source_id))),
                    vector=vector_payload,
                    payload=payload,
                )
            )
        if points:
            client.upsert(collection_name=version, points=points, wait=True)
        versions[artifact.collection] = version
        counts[artifact.collection] = len(points)
    if activate:
        for base, version in versions.items():
            client.update_collection_aliases(
                change_aliases_operations=[
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(
                            collection_name=version, alias_name=f"{base}_current"
                        )
                    )
                ]
            )
    return counts
