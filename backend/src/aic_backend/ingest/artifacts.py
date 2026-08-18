"""Validated ingestion of real AIC artifacts into versioned Qdrant collections."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from aic2026.common.io import atomic_write_json, sha256_path

from .ids import point_id
from .sparse import sparse_vector


@dataclass(frozen=True)
class ArtifactFile:
    collection: str
    path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ValidatedArtifact:
    source: ArtifactFile
    rows: list[dict[str, Any]]
    run_id: str
    model_revisions: tuple[str, ...]
    matrix: np.ndarray | None = None


def _records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _matrix_path(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"{path}: embedding index is empty")
    tag = "f16" if rows[0].get("dtype") == "float16" else "f32"
    return path.with_name(f"{path.stem}.{tag}.npy")


def _write_legacy_receipt(
    artifact: ArtifactFile, manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[str, tuple[str, ...]]:
    """Normalize old ASR manifests into a checksummed local staging receipt."""
    if artifact.collection != "asr" or manifest.get("schema_version") != "aic26.asr_manifest.v1":
        raise ValueError(f"{artifact.path}: manifest has no verifiable output checksum")
    if int(manifest.get("segment_count", -1)) != len(rows):
        raise ValueError(f"{artifact.path}: legacy ASR record count mismatch")
    model_id = str(manifest.get("model_id", ""))
    revision = str(manifest.get("config", {}).get("model_revision", "legacy-unpinned"))
    run_id = f"legacy-{sha256_path(artifact.path)[:12]}"
    receipt = {
        "schema_version": "aic26.ingest_receipt.v1",
        "status": "completed",
        "run_id": run_id,
        "source_manifest": str(artifact.manifest_path),
        "outputs": [{"source_id": str(artifact.path), "sha256": sha256_path(artifact.path)}],
        "records": len(rows),
        "models": [{"model_id": model_id, "revision": revision}],
    }
    receipt_path = (
        artifact.path.parents[1]
        / ".ingest-staging"
        / artifact.collection
        / f"{artifact.path.stem}.manifest.json"
    )
    atomic_write_json(receipt_path, receipt)
    return run_id, (f"{model_id}@{revision}",)


def validate_artifact(artifact: ArtifactFile) -> ValidatedArtifact:
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"{artifact.path}: manifest is not completed")
    rows = _records(artifact.path)
    outputs = manifest.get("outputs", [])
    output_hashes = {item.get("sha256") for item in outputs}
    if outputs:
        if sha256_path(artifact.path) not in output_hashes:
            raise ValueError(f"{artifact.path}: checksum does not match manifest")
        run_id = str(manifest.get("run_id", ""))
        if not run_id:
            raise ValueError(f"{artifact.path}: run_id is missing")
        models = manifest.get("models", [])
        if not models or any(not item.get("revision") for item in models):
            raise ValueError(f"{artifact.path}: pinned model revision is missing")
        model_revisions = tuple(f"{item.get('model_id')}@{item.get('revision')}" for item in models)
    else:
        run_id, model_revisions = _write_legacy_receipt(artifact, manifest, rows)

    expected = manifest.get("counters", {}).get("frames")
    if expected is None:
        expected = manifest.get("counters", {}).get("records")
    if expected is not None and int(expected) != len(rows):
        raise ValueError(f"{artifact.path}: record count does not match manifest")

    matrix: np.ndarray | None = None
    if artifact.collection.startswith("frames_"):
        matrix_path = _matrix_path(artifact.path, rows)
        if not matrix_path.is_file():
            raise ValueError(f"{artifact.path}: companion matrix is missing: {matrix_path}")
        if outputs and sha256_path(matrix_path) not in output_hashes:
            raise ValueError(f"{matrix_path}: checksum does not match manifest")
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[0] != len(rows):
            raise ValueError(f"{matrix_path}: matrix shape does not match index")
        if [row.get("row") for row in rows] != list(range(len(rows))):
            raise ValueError(f"{artifact.path}: row values must be contiguous from zero")
        if any(int(row.get("embedding_dim", -1)) != matrix.shape[1] for row in rows):
            raise ValueError(f"{artifact.path}: embedding_dim does not match matrix")
    return ValidatedArtifact(artifact, rows, run_id, model_revisions, matrix)


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
        directory = root / dirname
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            manifest = path.with_suffix(".manifest.json")
            if manifest.exists():
                found.append(ArtifactFile(collection, path, manifest))
    return found


def _base_payload(row: dict[str, Any], run_id: str, source: Path) -> dict[str, Any]:
    return {
        "video_id": str(row["video_id"]),
        "frame_idx": int(row["frame_idx"]),
        "pts_time_s": float(row["pts_time_s"]),
        "keyframe_n": row.get("keyframe_n"),
        "run_id": run_id,
        "source_artifact": source.name,
    }


def _text_points(artifact: ValidatedArtifact) -> Iterator[tuple[str, dict[str, Any], str]]:
    for row in artifact.rows:
        if artifact.source.collection == "regions":
            base = _base_payload(row, artifact.run_id, artifact.source.path)
            for region in row.get("regions", []):
                caption = region.get("caption", {})
                if caption.get("status") != "ok" or not caption.get("description_en"):
                    continue
                payload = {
                    **base,
                    "region_id": region["region_id"],
                    "detector": region.get("detector"),
                    "bbox_xyxy_px": region.get("bbox_xyxy_px"),
                    "text": caption["description_en"],
                }
                yield str(region["region_id"]), payload, payload["text"]
        elif artifact.source.collection == "ocr":
            if row.get("terminal_status", "success") != "success":
                continue
            base = _base_payload(row, artifact.run_id, artifact.source.path)
            lines = []
            for index, span in enumerate(row.get("texts", [])):
                raw_text = str(span.get("raw_text", ""))
                normalized_text = str(span.get("normalized_text") or raw_text)
                lines.append(
                    {
                        "line_id": str(span.get("line_id", f"line-{index:04d}")),
                        "raw_text": raw_text,
                        "normalized_text": normalized_text,
                        "confidence": span.get("confidence"),
                        "accepted": span.get("accepted", True) is True,
                        "polygon_xy": span.get("polygon_xy"),
                        "polygon_clamped": span.get("polygon_clamped", False) is True,
                        "reading_order": int(span.get("reading_order", index)),
                    }
                )
            ocr_frame = {
                "terminal_status": row.get("terminal_status", "success"),
                "full_text": row.get("full_text")
                or " ".join(line["normalized_text"] for line in lines if line["accepted"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "run_id": artifact.run_id,
                "model_revisions": list(artifact.model_revisions),
                "source_image_sha256": row.get("source_image_sha256"),
                "lines": lines,
            }
            text = str(ocr_frame["full_text"])
            if text:
                source_id = f"{row['frame_uid']}:ocr"
                yield source_id, {**base, "text": text, "ocr_frame": ocr_frame}, text
        elif artifact.source.collection == "asr":
            text = row.get("transcript_normalized") or row.get("transcript_raw")
            if not text:
                continue
            for keyframe in row.get("keyframes", []):
                payload = {
                    "video_id": str(row["video_id"]),
                    "frame_idx": int(keyframe["frame_idx"]),
                    "pts_time_s": float(keyframe["pts_time_s"]),
                    "keyframe_n": keyframe.get("keyframe_n"),
                    "run_id": artifact.run_id,
                    "segment_id": row["segment_id"],
                    "start_ms": row["start_ms"],
                    "end_ms": row["end_ms"],
                    "source_artifact": artifact.source.path.name,
                    "text": text,
                }
                yield f"{row['segment_id']}:{keyframe['frame_idx']}", payload, text


def _alias_operations(client: Any, base: str, version: str, models: Any) -> list[Any]:
    alias = f"{base}_current"
    aliases = {item.alias_name for item in client.get_aliases().aliases}
    operations: list[Any] = []
    if alias in aliases:
        operations.append(
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
        )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=version, alias_name=alias)
        )
    )
    return operations


def ingest(
    client: Any,
    artifacts: list[ArtifactFile],
    *,
    dense_encoder: Any | None,
    activate: bool,
) -> dict[str, int]:
    from qdrant_client import models

    validated = [validate_artifact(item) for item in artifacts]
    grouped: dict[str, list[ValidatedArtifact]] = defaultdict(list)
    for item in validated:
        grouped[item.source.collection].append(item)

    versions: dict[str, str] = {}
    counts: dict[str, int] = {}
    for collection, sources in grouped.items():
        version = f"{collection}_{uuid4().hex[:12]}"
        if collection.startswith("frames_"):
            dimensions = {int(item.matrix.shape[1]) for item in sources if item.matrix is not None}
            if len(dimensions) != 1:
                raise ValueError(f"{collection}: inconsistent embedding dimensions")
            vectors = {
                "scene": models.VectorParams(size=dimensions.pop(), distance=models.Distance.COSINE)
            }
            sparse = None
        else:
            vectors = (
                {"dense": models.VectorParams(size=768, distance=models.Distance.COSINE)}
                if dense_encoder is not None
                else {}
            )
            sparse = {"lexical": models.SparseVectorParams()}
        client.create_collection(
            collection_name=version,
            vectors_config=vectors,
            sparse_vectors_config=sparse,
        )
        total = 0
        for source in sources:
            points: list[Any] = []
            if collection.startswith("frames_"):
                assert source.matrix is not None
                for row in source.rows:
                    payload = _base_payload(row, source.run_id, source.source.path)
                    payload["frame_relpath"] = row.get("frame_relpath")
                    vector = source.matrix[int(row["row"])].astype(np.float32).tolist()
                    points.append(
                        models.PointStruct(
                            id=str(point_id(collection=collection, source_id=row["frame_uid"])),
                            vector={"scene": vector},
                            payload=payload,
                        )
                    )
            else:
                for source_id, payload, text in _text_points(source):
                    indices, values = sparse_vector(text)
                    vectors_by_name: dict[str, Any] = {
                        "lexical": models.SparseVector(indices=indices, values=values)
                    }
                    if dense_encoder is not None:
                        vectors_by_name["dense"] = dense_encoder.encode(
                            [text], query=False
                        )[0].tolist()
                    points.append(
                        models.PointStruct(
                            id=str(point_id(collection=collection, source_id=source_id)),
                            vector=vectors_by_name,
                            payload=payload,
                        )
                    )
            for start in range(0, len(points), 256):
                client.upsert(
                    collection_name=version,
                    points=points[start : start + 256],
                    wait=True,
                )
            total += len(points)
        versions[collection] = version
        counts[collection] = total
    if activate:
        for base, version in versions.items():
            client.update_collection_aliases(
                change_aliases_operations=_alias_operations(client, base, version, models)
            )
    return counts
