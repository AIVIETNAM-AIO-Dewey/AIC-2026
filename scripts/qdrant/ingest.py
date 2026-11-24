#!/usr/bin/env python3
"""Resumable ingestion of the local AIC vectors into a Qdrant server."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models
from safetensors.numpy import load_file

LOGGER = logging.getLogger("aic-qdrant-ingest")

FRAME_COLLECTION = "aic_frames"
BEIT3_COLLECTION = "aic_beit3_frames"
DAM_COLLECTION = "aic_dam_regions"
EXPECTED_FRAMES = 247_956
EXPECTED_DAM_REGIONS = 681_355
POINT_SCHEMA_VERSION = "aic.ingest.v2"
INGEST_MANIFEST_SCHEMA_VERSION = "qdrant.ingestion.v3"

# A single process handles one collection at a time.  Keeping the report in
# memory avoids changing the public ingest_* return values while still
# allowing the manifest writer to publish verification evidence.
RECONCILIATION_REPORTS: dict[str, dict[str, Any]] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_paths(
    data_root: Path,
    selected: tuple[str, ...],
    beit3_dir: Path | None = None,
) -> list[Path]:
    paths: list[Path] = []
    if "frames" in selected:
        paths.extend(
            [
                data_root / "visual_embeddings" / "metaclip2" / "keyframes_visual_vectors.f16.npy",
                data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl",
            ]
        )
        paths.extend(sorted((data_root / "scene_embeddings").glob("*.safetensors")))
    if "beit3" in selected:
        beit_root = beit3_dir or data_root / "visual_embeddings" / "beit3"
        paths.extend(
            [
                beit_root / "keyframes_visual_vectors.f16.npy",
                beit_root / "keyframes_metadata.jsonl",
            ]
        )
    if "dam" in selected:
        paths.extend(
            [
                data_root / "dense_text_embeddings" / "dam_vectors.f16.npy",
                data_root / "dense_text_embeddings" / "dam_metadata.jsonl",
                data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl",
            ]
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ingestion artifact(s): " + ", ".join(missing[:5]))
    return list(dict.fromkeys(paths))


def write_ingestion_manifest(
    state_root: Path,
    data_root: Path,
    collection_results: dict[str, int],
    selected: tuple[str, ...],
    beit3_dir: Path | None = None,
    *,
    status: str = "ready",
    error: str | None = None,
    verification: dict[str, dict[str, Any]] | None = None,
) -> None:
    state_root.mkdir(parents=True, exist_ok=True)
    destination = state_root / "qdrant_ingestion_manifest.json"
    previous: dict[str, Any] = {}
    if destination.is_file():
        try:
            previous = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            previous = {}
    merged_collections = dict(previous.get("collections") or {})
    merged_collections.update(collection_results)
    payload: dict[str, Any] = {
        "schema_version": INGEST_MANIFEST_SCHEMA_VERSION,
        "status": status,
        "passed": status == "ready",
        "ingest_schema_version": POINT_SCHEMA_VERSION,
        "collections": merged_collections,
    }
    merged_verification = dict(previous.get("verification") or {})
    if verification:
        merged_verification.update(verification)
    if status == "ready":
        collection_names = {
            "frames": FRAME_COLLECTION,
            "beit3": BEIT3_COLLECTION,
            "dam": DAM_COLLECTION,
        }
        expected_counts = {
            FRAME_COLLECTION: EXPECTED_FRAMES,
            BEIT3_COLLECTION: EXPECTED_FRAMES,
            DAM_COLLECTION: EXPECTED_DAM_REGIONS,
        }
        incomplete = []
        for selected_name in selected:
            collection_name = collection_names[selected_name]
            evidence = merged_verification.get(collection_name) or {}
            if not (
                int(merged_collections.get(collection_name, -1)) == expected_counts[collection_name]
                and evidence.get("expected_count") == expected_counts[collection_name]
                and evidence.get("verified_count") == expected_counts[collection_name]
                and evidence.get("payload_verified") is True
                and evidence.get("vector_content_verified") is True
                and (evidence.get("verification_threshold") or {}).get("cosine_min")
                == VECTOR_DIRECTION_COSINE_MIN
                and (evidence.get("verification_threshold") or {}).get("max_abs_error")
                == VECTOR_DIRECTION_MAX_ABS_ERROR
                and bool(evidence.get("completed_at"))
            ):
                incomplete.append(collection_name)
        if incomplete:
            raise ValueError(
                "Cannot publish a ready ingestion manifest without complete "
                f"verification evidence: {', '.join(incomplete)}"
            )
    if merged_verification:
        payload["verification"] = merged_verification
    if status != "ready" and previous.get("artifacts"):
        payload["artifacts"] = dict(previous["artifacts"])
    if status == "ready":

        def artifact_key(path: Path) -> str:
            try:
                return path.relative_to(data_root).as_posix()
            except ValueError:
                return path.resolve().as_posix()

        artifacts = dict(previous.get("artifacts") or {})
        artifacts.update(
            {
                artifact_key(path): {
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "sha256": sha256_file(path),
                }
                for path in _artifact_paths(data_root, selected, beit3_dir)
            }
        )
        payload["artifacts"] = artifacts
    if error:
        payload["error"] = error
    staging = destination.with_suffix(destination.suffix + ".staging")
    staging.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(staging, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://qdrant:6333")
    parser.add_argument("--grpc-host", default="qdrant")
    parser.add_argument("--grpc-port", type=int, default=6334)
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path(os.environ.get("AIC_STATE_ROOT", "/state")),
        help="Writable state directory for the verified ingestion manifest.",
    )
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--dam-batch-size", type=int, default=256)
    parser.add_argument(
        "--only",
        choices=("all", "frames", "beit3", "dam"),
        default="all",
        help="Limit ingestion to one collection.",
    )
    parser.add_argument(
        "--beit3-dir",
        type=Path,
        help="Override the BEiT-3 artifact directory (used for validated staging imports).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the selected collections before ingesting.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify collection contents without repairing incomplete points.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Leave HNSW disabled after upload; exact search remains available.",
    )
    return parser.parse_args()


def request_json(
    base_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: float = 120.0,
) -> dict[str, Any]:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=payload,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qdrant HTTP {error.code} for {path}: {details}") from error


def wait_for_qdrant(base_url: str, timeout_seconds: int = 180) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = request_json(base_url, "GET", "/", timeout=5)
            if response.get("title") == "qdrant - vector search engine":
                return
        except (OSError, RuntimeError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError(f"Qdrant did not become healthy at {base_url}")


def collection_exists(base_url: str, collection: str) -> bool:
    try:
        request_json(base_url, "GET", f"/collections/{collection}")
        return True
    except RuntimeError as error:
        if "HTTP 404" in str(error):
            return False
        raise


def delete_collection(base_url: str, collection: str) -> None:
    if collection_exists(base_url, collection):
        LOGGER.warning("Deleting collection %s", collection)
        request_json(base_url, "DELETE", f"/collections/{collection}?timeout=120")


def create_collection(
    base_url: str,
    collection: str,
    vectors: dict[str, int],
    payload_indexes: Iterable[tuple[str, str]],
) -> None:
    LOGGER.info("Creating collection %s", collection)
    vector_config = {
        name: {
            "size": dimension,
            "distance": "Cosine",
            "datatype": "float16",
            "memory": "cold",
        }
        for name, dimension in vectors.items()
    }
    request_json(
        base_url,
        "PUT",
        f"/collections/{collection}?timeout=120",
        {
            "vectors": vector_config,
            "hnsw_config": {"m": 0, "memory": "cold"},
            "optimizers_config": {"indexing_threshold": 0},
            "quantization_config": {
                "scalar": {
                    "type": "int8",
                    "quantile": 0.99,
                    "memory": "pinned",
                }
            },
            "payload": {"memory": "cold"},
        },
    )
    for field_name, field_schema in payload_indexes:
        request_json(
            base_url,
            "PUT",
            f"/collections/{collection}/index?wait=true",
            {"field_name": field_name, "field_schema": field_schema},
        )


def ensure_payload_indexes(
    base_url: str,
    collection: str,
    payload_indexes: Iterable[tuple[str, str]],
) -> None:
    """Ensure indexes needed by the canonical health/search filters exist.

    Qdrant returns a conflict when an index already exists.  That is a safe,
    idempotent outcome for a repair run, so only that particular error is
    ignored; connection and schema errors still fail the command.
    """
    for field_name, field_schema in payload_indexes:
        try:
            request_json(
                base_url,
                "PUT",
                f"/collections/{collection}/index?wait=true",
                {"field_name": field_name, "field_schema": field_schema},
            )
        except RuntimeError as error:
            message = str(error).lower()
            if "already" not in message or ("exist" not in message and "index" not in message):
                raise


def validate_collection_definition(
    base_url: str,
    collection: str,
    vectors: dict[str, int],
) -> dict[str, Any]:
    """Fail closed when an existing collection has an incompatible schema."""
    response = request_json(base_url, "GET", f"/collections/{collection}")
    result = response.get("result", response)
    configured = ((result.get("config") or {}).get("params") or {}).get("vectors") or {}
    if not isinstance(configured, dict):
        raise ValueError(f"{collection} has an unsupported unnamed-vector configuration")
    extra_names = set(configured) - set(vectors)
    if extra_names:
        raise ValueError(
            f"{collection} contains unexpected named vectors {sorted(extra_names)}; "
            "use --recreate explicitly"
        )
    for name, dimension in vectors.items():
        definition = configured.get(name)
        if not isinstance(definition, dict):
            raise ValueError(
                f"{collection} is missing named vector {name!r}; use --recreate explicitly"
            )
        if int(definition.get("size", -1)) != dimension:
            raise ValueError(
                f"{collection}/{name} has dimension {definition.get('size')!r}; "
                "use --recreate explicitly"
            )
        if str(definition.get("distance", "")).lower() != "cosine":
            raise ValueError(f"{collection}/{name} is not Cosine; use --recreate explicitly")
    return result


def enable_hnsw(base_url: str, collection: str) -> None:
    LOGGER.info("Enabling disk-backed HNSW for %s", collection)
    request_json(
        base_url,
        "PATCH",
        f"/collections/{collection}?timeout=120",
        {
            "hnsw_config": {
                "m": 8,
                "ef_construct": 64,
                "memory": "cold",
            },
            "optimizers_config": {"indexing_threshold": 20_000},
        },
    )


def qdrant_count(client: QdrantClient, collection: str) -> int:
    result = client.count(collection_name=collection, exact=True)
    # QdrantClient returns CountResult, while lightweight/native adapters may
    # return the count directly.  Supporting both keeps verification portable.
    return int(getattr(result, "count", result))


def validate_numpy_matrix(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != expected_shape:
        raise ValueError(f"{path}: expected shape {expected_shape}, got {matrix.shape}")
    if matrix.dtype != np.float16:
        raise ValueError(f"{path}: expected float16, got {matrix.dtype}")
    return matrix


def upsert_with_retry(
    client: QdrantClient,
    collection: str,
    ids: list[int],
    vectors: dict[str, np.ndarray],
    payloads: list[dict[str, Any]],
) -> None:
    batch = models.Batch(
        ids=ids,
        vectors={
            name: value.astype(np.float32, copy=False).tolist() for name, value in vectors.items()
        },
        payloads=payloads,
    )
    for attempt in range(1, 6):
        try:
            client.upsert(collection_name=collection, points=batch, wait=True)
            return
        except Exception:
            if attempt == 5:
                raise
            delay = 2**attempt
            LOGGER.exception("Upsert failed; retrying in %ss", delay)
            time.sleep(delay)


def ensure_frame_collection(base_url: str, recreate: bool, verify_only: bool = False) -> None:
    if recreate:
        delete_collection(base_url, FRAME_COLLECTION)
    indexes = (
        ("video_id", "keyword"),
        ("frame_idx", "integer"),
        ("frame_uid", "keyword"),
        ("keyframe_n", "integer"),
        ("ingest_schema_version", "keyword"),
    )
    if not collection_exists(base_url, FRAME_COLLECTION):
        if verify_only:
            raise FileNotFoundError(f"Missing collection {FRAME_COLLECTION}")
        create_collection(
            base_url,
            FRAME_COLLECTION,
            {"siglip2": 768, "metaclip2": 1024},
            indexes,
        )
    elif not verify_only:
        validate_collection_definition(
            base_url, FRAME_COLLECTION, {"siglip2": 768, "metaclip2": 1024}
        )
        ensure_payload_indexes(base_url, FRAME_COLLECTION, indexes)


def ensure_dam_collection(base_url: str, recreate: bool, verify_only: bool = False) -> None:
    if recreate:
        delete_collection(base_url, DAM_COLLECTION)
    indexes = (
        ("video_id", "keyword"),
        ("frame_idx", "integer"),
        ("class_entity", "keyword"),
        ("parent_point_id", "integer"),
        ("frame_uid", "keyword"),
        ("region_id", "keyword"),
        ("ingest_schema_version", "keyword"),
    )
    if not collection_exists(base_url, DAM_COLLECTION):
        if verify_only:
            raise FileNotFoundError(f"Missing collection {DAM_COLLECTION}")
        create_collection(
            base_url,
            DAM_COLLECTION,
            {"dam": 1024},
            indexes,
        )
    elif not verify_only:
        validate_collection_definition(base_url, DAM_COLLECTION, {"dam": 1024})
        ensure_payload_indexes(base_url, DAM_COLLECTION, indexes)


def ensure_beit3_collection(base_url: str, recreate: bool, verify_only: bool = False) -> None:
    if recreate:
        delete_collection(base_url, BEIT3_COLLECTION)
    indexes = (
        ("video_id", "keyword"),
        ("frame_idx", "integer"),
        ("frame_uid", "keyword"),
        ("keyframe_n", "integer"),
        ("ingest_schema_version", "keyword"),
    )
    if not collection_exists(base_url, BEIT3_COLLECTION):
        if verify_only:
            raise FileNotFoundError(f"Missing collection {BEIT3_COLLECTION}")
        create_collection(
            base_url,
            BEIT3_COLLECTION,
            {"beit3": 768},
            indexes,
        )
    elif not verify_only:
        validate_collection_definition(base_url, BEIT3_COLLECTION, {"beit3": 768})
        ensure_payload_indexes(base_url, BEIT3_COLLECTION, indexes)


def load_frame_metadata(metadata_path: Path) -> dict[str, dict[str, Any]]:
    """Load the canonical frame identity map used to enrich DAM payloads.

    The DAM export intentionally stores only the video/frame identity.  The
    keyframe number and canonical point id live in the frame metadata, so they
    must be joined here before DAM points are written to Qdrant.
    """
    mapping: dict[str, dict[str, Any]] = {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            item = json.loads(line)
            point_id = int(item["point_id"])
            if point_id != row + 1:
                raise ValueError(f"Unexpected frame point_id {point_id} at row {row}")
            frame_uid = str(item["frame_uid"])
            derived_uid = f"{item.get('video_id')}:{int(item.get('frame_idx', -1))}"
            if frame_uid != derived_uid or frame_uid in mapping:
                raise ValueError(f"Invalid or duplicate canonical frame_uid at row {row + 1}")
            mapping[frame_uid] = item
    if len(mapping) != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} frame IDs, found {len(mapping)}")
    return mapping


def load_frame_point_ids(metadata_path: Path) -> dict[str, int]:
    """Backward-compatible point-id view of :func:`load_frame_metadata`."""
    return {
        frame_uid: int(item["point_id"])
        for frame_uid, item in load_frame_metadata(metadata_path).items()
    }


def _record_value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _record_vectors(record: Any) -> dict[str, Any]:
    value = _record_value(record, "vector", {})
    return value if isinstance(value, dict) else {}


def _record_payload(record: Any) -> dict[str, Any]:
    value = _record_value(record, "payload", {})
    return value if isinstance(value, dict) else {}


def _vector_is_valid(value: Any, dimension: int) -> bool:
    try:
        vector = np.asarray(value, dtype=np.float32)
        return (
            vector.shape == (dimension,)
            and bool(np.isfinite(vector).all())
            and float(np.linalg.norm(vector)) > 0.0
        )
    except (TypeError, ValueError):
        return False


VECTOR_DIRECTION_COSINE_MIN = 0.99999
VECTOR_DIRECTION_MAX_ABS_ERROR = 0.002


def _vector_matches_source(
    actual: Any,
    expected: Any,
    dimension: int,
) -> tuple[bool, dict[str, float]]:
    """Compare vector direction while allowing Qdrant cosine normalization.

    Qdrant may normalize cosine vectors when they are stored.  Comparing raw
    bytes would therefore reject a correct point.  Comparing normalized
    direction catches shuffled/stale vectors without depending on the stored
    vector magnitude.
    """
    if not _vector_is_valid(actual, dimension) or not _vector_is_valid(expected, dimension):
        return False, {"cosine": -1.0, "max_abs_error": float("inf")}
    actual_array = np.asarray(actual, dtype=np.float32)
    expected_array = np.asarray(expected, dtype=np.float32)
    actual_norm = float(np.linalg.norm(actual_array))
    expected_norm = float(np.linalg.norm(expected_array))
    if actual_norm <= 0.0 or expected_norm <= 0.0:
        return False, {"cosine": -1.0, "max_abs_error": float("inf")}
    actual_unit = actual_array / actual_norm
    expected_unit = expected_array / expected_norm
    cosine = float(np.dot(actual_unit, expected_unit))
    max_abs_error = float(np.max(np.abs(actual_unit - expected_unit)))
    matched = (
        np.isfinite(cosine)
        and np.isfinite(max_abs_error)
        and cosine >= VECTOR_DIRECTION_COSINE_MIN
        and max_abs_error <= VECTOR_DIRECTION_MAX_ABS_ERROR
    )
    return matched, {"cosine": cosine, "max_abs_error": max_abs_error}


def _payload_value_matches(actual: Any, expected: Any) -> bool:
    """Compare JSON payload values while tolerating storage float roundoff."""
    if isinstance(expected, float):
        try:
            return bool(
                np.isfinite(float(actual))
                and np.isclose(float(actual), expected, rtol=1e-7, atol=1e-7)
            )
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _payload_value_matches(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=False)
            )
        )
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _payload_value_matches(actual[key], expected_value)
            for key, expected_value in expected.items()
        )
    return actual == expected


def _payload_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return _payload_value_matches(actual, expected)


def _record_matches_source(
    record: Any,
    source: dict[str, Any],
    vector_dimensions: dict[str, int],
) -> tuple[bool, dict[str, Any]]:
    """Return whether a Qdrant record matches one canonical source point."""
    if record is None:
        return False, {"reason": "missing_point"}
    try:
        if int(_record_value(record, "id")) != int(source["id"]):
            return False, {
                "reason": "point_id_mismatch",
                "actual_point_id": int(_record_value(record, "id")),
                "expected_point_id": int(source["id"]),
            }
    except (TypeError, ValueError, KeyError):
        return False, {"reason": "point_id_mismatch"}
    payload = _record_payload(record)
    if not _payload_matches(payload, source["payload"]):
        return False, {"reason": "payload_mismatch"}
    vectors = _record_vectors(record)
    vector_diagnostics: dict[str, Any] = {}
    for name, dimension in vector_dimensions.items():
        matched, diagnostic = _vector_matches_source(
            vectors.get(name), source["vectors"].get(name), dimension
        )
        vector_diagnostics[name] = diagnostic
        if not matched:
            return False, {
                "reason": "vector_mismatch",
                "vector": name,
                "vector_diagnostics": vector_diagnostics,
            }
    return True, {"reason": "verified", "vector_diagnostics": vector_diagnostics}


def _upsert_source_batch(
    client: QdrantClient,
    collection: str,
    sources: list[dict[str, Any]],
    vector_dimensions: dict[str, int],
) -> None:
    if not sources:
        return
    for source in sources:
        source_vectors = source.get("vectors") or {}
        if set(source_vectors) != set(vector_dimensions):
            raise ValueError(
                f"Invalid named vectors for {collection}: "
                f"got={sorted(source_vectors)}, expected={sorted(vector_dimensions)}"
            )
        for name, dimension in vector_dimensions.items():
            if not _vector_is_valid(source_vectors.get(name), dimension):
                raise ValueError(f"Invalid source vector for {collection}/{name}")
        payload = source.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"Missing canonical payload for {collection}/{source.get('id')}")
        try:
            source_id = int(source["id"])
            payload_id = int(payload["point_id"])
            video_id = str(payload["video_id"])
            frame_idx = int(payload["frame_idx"])
            frame_uid = str(payload["frame_uid"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid canonical payload for {collection}/{source.get('id')}"
            ) from error
        if payload_id != source_id or not video_id or frame_uid != f"{video_id}:{frame_idx}":
            raise ValueError(
                f"Payload identity does not match source point {source_id} in {collection}"
            )
    ids = [int(source["id"]) for source in sources]
    maximum_id = EXPECTED_DAM_REGIONS if collection == DAM_COLLECTION else EXPECTED_FRAMES
    if len(set(ids)) != len(ids) or any(not 1 <= point_id <= maximum_id for point_id in ids):
        raise ValueError(f"Invalid or duplicate source point IDs for {collection}")
    upsert_with_retry(
        client,
        collection,
        ids,
        {
            name: np.stack([source["vectors"][name] for source in sources])
            for name in vector_dimensions
        },
        [source["payload"] for source in sources],
    )


def _validate_existing_ids(
    client: QdrantClient,
    collection: str,
    expected_count: int,
    current_count: int,
) -> set[int]:
    """Return the existing integer IDs and reject out-of-range/duplicate IDs."""
    seen: set[int] = set()
    offset: Any = None
    while True:
        records, next_offset = client.scroll(
            collection_name=collection,
            limit=2048,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for record in records:
            try:
                point_id = int(_record_value(record, "id"))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{collection} contains a non-integer point ID") from error
            if not 1 <= point_id <= expected_count:
                raise ValueError(
                    f"{collection} contains out-of-range point ID {point_id}; "
                    "no automatic pruning is performed"
                )
            if point_id in seen:
                raise ValueError(f"{collection} returned duplicate point ID {point_id}")
            seen.add(point_id)
        if next_offset is None:
            break
        offset = next_offset
    if len(seen) != current_count:
        raise ValueError(
            f"{collection} scroll returned {len(seen)} unique IDs but count reported {current_count}"
        )
    return seen


def reconcile_collection(
    client: QdrantClient,
    base_url: str,
    collection: str,
    expected_count: int,
    vector_dimensions: dict[str, int],
    source_records: Iterable[dict[str, Any]],
    batch_size: int,
    repair: bool = True,
) -> int:
    """Verify expected IDs and repair only points that are incomplete.

    Existing collections are never trusted merely because their count matches.
    Source records are consumed in canonical order, while Qdrant is retrieved
    by the source IDs so holes and non-contiguous legacy collections cannot
    shift the embedding row alignment.
    """
    validate_collection_definition(base_url, collection, vector_dimensions)
    current_count = qdrant_count(client, collection)
    if current_count > expected_count:
        raise ValueError(
            f"{collection} has {current_count} points; expected exactly {expected_count}. "
            "No automatic pruning is performed."
        )
    existing_ids = _validate_existing_ids(client, collection, expected_count, current_count)
    if current_count == expected_count and len(existing_ids) != expected_count:
        raise ValueError(
            f"{collection} has incomplete canonical ID coverage; no repair was attempted"
        )

    source_batch: list[dict[str, Any]] = []
    repaired = 0
    verified_points = 0
    mismatches: dict[str, int] = {}
    mismatch_diagnostics: list[dict[str, Any]] = []
    seen = 0
    for source in source_records:
        seen += 1
        source_batch.append(source)
        if len(source_batch) < batch_size and seen < expected_count:
            continue
        ids = [int(item["id"]) for item in source_batch]
        records = client.retrieve(
            collection_name=collection,
            ids=ids,
            with_payload=True,
            with_vectors=True,
        )
        by_id = {int(_record_value(record, "id")): record for record in records}
        needs_repair: list[dict[str, Any]] = []
        for source_item in source_batch:
            point_id = int(source_item["id"])
            record = by_id.get(point_id)
            valid, diagnostic = _record_matches_source(record, source_item, vector_dimensions)
            if not valid:
                needs_repair.append(source_item)
                reason = str(diagnostic.get("reason", "mismatch"))
                mismatches[reason] = mismatches.get(reason, 0) + 1
                if len(mismatch_diagnostics) < 100:
                    mismatch_diagnostics.append({"point_id": point_id, **diagnostic})
                # Keep enough samples for diagnosis without emitting one log
                # line per legacy point during a full schema migration.
                if len(mismatch_diagnostics) <= 10:
                    LOGGER.warning(
                        "%s point %s failed source verification: %s",
                        collection,
                        point_id,
                        diagnostic,
                    )
            else:
                verified_points += 1
        if needs_repair:
            if not repair:
                RECONCILIATION_REPORTS[collection] = {
                    "expected_count": expected_count,
                    "verified_count": verified_points,
                    "repaired_count": 0,
                    "payload_verified": False,
                    "vector_content_verified": False,
                    "vector_direction_cosine_min": VECTOR_DIRECTION_COSINE_MIN,
                    "vector_direction_max_abs_error": VECTOR_DIRECTION_MAX_ABS_ERROR,
                    "verification_threshold": {
                        "cosine_min": VECTOR_DIRECTION_COSINE_MIN,
                        "max_abs_error": VECTOR_DIRECTION_MAX_ABS_ERROR,
                    },
                    "mismatch_counts": dict(mismatches),
                    "mismatch_diagnostics": list(mismatch_diagnostics),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "verify_only": True,
                }
                raise ValueError(
                    f"{collection}: verification found {len(needs_repair)} mismatched points; "
                    f"reasons={mismatches}"
                )
            _upsert_source_batch(client, collection, needs_repair, vector_dimensions)
            repaired_ids = [int(item["id"]) for item in needs_repair]
            repaired_records = client.retrieve(
                collection_name=collection,
                ids=repaired_ids,
                with_payload=True,
                with_vectors=True,
            )
            repaired_by_id = {
                int(_record_value(record, "id")): record for record in repaired_records
            }
            for source_item in needs_repair:
                point_id = int(source_item["id"])
                repaired_valid, diagnostic = _record_matches_source(
                    repaired_by_id.get(point_id), source_item, vector_dimensions
                )
                if not repaired_valid:
                    raise RuntimeError(
                        f"{collection}: readback verification failed for point {point_id}: "
                        f"{diagnostic}"
                    )
            verified_points += len(needs_repair)
            repaired += len(needs_repair)
        if seen == expected_count or seen % 10_000 < len(source_batch):
            LOGGER.info(
                "%s reconciliation progress: %s/%s verified, %s repaired",
                collection,
                verified_points,
                expected_count,
                repaired,
            )
        source_batch.clear()

    if seen != expected_count:
        raise ValueError(f"{collection}: expected {expected_count} source records, found {seen}")
    final_count = qdrant_count(client, collection)
    if final_count != expected_count:
        raise ValueError(
            f"{collection}: after repair expected {expected_count} points, found {final_count}; "
            "the collection may contain out-of-range IDs"
        )
    RECONCILIATION_REPORTS[collection] = {
        "expected_count": expected_count,
        "verified_count": verified_points,
        "repaired_count": repaired,
        "payload_verified": True,
        "vector_content_verified": True,
        "vector_direction_cosine_min": VECTOR_DIRECTION_COSINE_MIN,
        "vector_direction_max_abs_error": VECTOR_DIRECTION_MAX_ABS_ERROR,
        "verification_threshold": {
            "cosine_min": VECTOR_DIRECTION_COSINE_MIN,
            "max_abs_error": VECTOR_DIRECTION_MAX_ABS_ERROR,
        },
        "mismatch_counts": mismatches,
        "mismatch_diagnostics": mismatch_diagnostics,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "verify_only": not repair,
    }
    LOGGER.info("%s verified (%s points repaired)", collection, repaired)
    return final_count


def _frame_sources(data_root: Path) -> Iterable[dict[str, Any]]:
    metaclip_dir = data_root / "visual_embeddings" / "metaclip2"
    metadata_path = metaclip_dir / "keyframes_metadata.jsonl"
    metaclip = validate_numpy_matrix(
        metaclip_dir / "keyframes_visual_vectors.f16.npy",
        (EXPECTED_FRAMES, 1024),
    )
    scene_dir = data_root / "scene_embeddings"
    current_video: str | None = None
    siglip: np.ndarray | None = None
    seen_frame_uids: set[str] = set()
    with metadata_path.open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            item = json.loads(line)
            point_id = int(item["point_id"])
            if point_id != row + 1:
                raise ValueError(f"Unexpected frame point_id {point_id} at row {row + 1}")
            video_id = str(item["video_id"])
            frame_uid = str(item["frame_uid"])
            if frame_uid != f"{video_id}:{int(item['frame_idx'])}":
                raise ValueError(f"Invalid frame_uid at canonical row {row + 1}: {frame_uid}")
            if frame_uid in seen_frame_uids:
                raise ValueError(f"Duplicate frame_uid at canonical row {row + 1}: {frame_uid}")
            seen_frame_uids.add(frame_uid)
            if video_id != current_video:
                tensor_path = scene_dir / f"{video_id}.safetensors"
                tensors = load_file(tensor_path)
                if set(tensors) != {"embeddings"}:
                    raise ValueError(f"Unexpected tensors in {tensor_path}: {sorted(tensors)}")
                siglip = tensors["embeddings"]
                if siglip.ndim != 2 or siglip.shape[1] != 768 or siglip.dtype != np.float16:
                    raise ValueError(
                        f"Invalid SigLIP shard {tensor_path}: {siglip.shape} {siglip.dtype}"
                    )
                current_video = video_id
            assert siglip is not None
            local_row = int(item["vector_row"])
            if not 0 <= local_row < siglip.shape[0]:
                raise IndexError(f"SigLIP row {local_row} missing for {video_id}")
            payload = {
                "point_id": point_id,
                "video_id": video_id,
                "keyframe_n": int(item["keyframe_n"]),
                "frame_idx": int(item["frame_idx"]),
                "pts_time_s": float(item["pts_time_s"]),
                "fps": float(item["fps"]),
                "frame_uid": frame_uid,
                "image_relpath": str(item["image_relpath"]),
                "ingest_schema_version": POINT_SCHEMA_VERSION,
            }
            yield {
                "id": point_id,
                "vectors": {"siglip2": siglip[local_row], "metaclip2": metaclip[row]},
                "payload": payload,
            }


def _beit3_sources(data_root: Path, artifact_dir: Path | None) -> Iterable[dict[str, Any]]:
    beit3_dir = artifact_dir or data_root / "visual_embeddings" / "beit3"
    metadata_path = beit3_dir / "keyframes_metadata.jsonl"
    canonical_path = data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
    vectors = validate_numpy_matrix(
        beit3_dir / "keyframes_visual_vectors.f16.npy",
        (EXPECTED_FRAMES, 768),
    )
    _validate_finite_matrix(vectors, beit3_dir / "keyframes_visual_vectors.f16.npy")
    with (
        metadata_path.open("r", encoding="utf-8") as beit_handle,
        canonical_path.open("r", encoding="utf-8") as canonical_handle,
    ):
        for row, pair in enumerate(itertools.zip_longest(beit_handle, canonical_handle)):
            beit_line, canonical_line = pair
            if beit_line is None or canonical_line is None:
                raise ValueError("BEiT-3 and canonical metadata row counts differ")
            item = json.loads(beit_line)
            canonical = json.loads(canonical_line)
            for field in ("point_id", "frame_uid", "video_id", "frame_idx", "keyframe_n"):
                if item.get(field) != canonical.get(field):
                    raise ValueError(f"BEiT-3 metadata mismatch at row {row + 1}, field {field}")
            point_id = int(item["point_id"])
            if point_id != row + 1:
                raise ValueError(f"Unexpected BEiT-3 point_id {point_id} at row {row + 1}")
            yield {
                "id": point_id,
                "vectors": {"beit3": vectors[row]},
                "payload": {
                    "point_id": point_id,
                    "video_id": str(item["video_id"]),
                    "keyframe_n": int(item["keyframe_n"]),
                    "frame_idx": int(item["frame_idx"]),
                    "pts_time_s": float(item["pts_time_s"]),
                    "fps": float(item["fps"]),
                    "frame_uid": str(item["frame_uid"]),
                    "image_relpath": str(item["image_relpath"]),
                    "ingest_schema_version": POINT_SCHEMA_VERSION,
                },
            }


def _dam_sources(data_root: Path) -> Iterable[dict[str, Any]]:
    dense_dir = data_root / "dense_text_embeddings"
    metadata_path = dense_dir / "dam_metadata.jsonl"
    vectors = validate_numpy_matrix(
        dense_dir / "dam_vectors.f16.npy",
        (EXPECTED_DAM_REGIONS, 1024),
    )
    _validate_finite_matrix(vectors, dense_dir / "dam_vectors.f16.npy")
    frame_metadata = data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
    frame_records = load_frame_metadata(frame_metadata)
    seen_region_ids: set[str] = set()
    with metadata_path.open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            item = json.loads(line)
            point_id = row + 1
            region_id = str(item.get("region_id") or "")
            if not region_id or region_id in seen_region_ids:
                raise ValueError(
                    f"DAM metadata has a missing or duplicate region_id at row {row + 1}"
                )
            seen_region_ids.add(region_id)
            if len(item.get("bbox") or []) != 4:
                raise ValueError(f"DAM metadata has an invalid bbox at row {row + 1}")
            frame_uid = f"{item['video_id']}:{int(item['frame_idx'])}"
            frame_record = frame_records.get(frame_uid)
            if frame_record is None:
                raise KeyError(f"No parent frame for DAM region {item.get('region_id')}")
            exported_keyframe_n = item.get("keyframe_n")
            if exported_keyframe_n is not None and int(exported_keyframe_n) != int(
                frame_record["keyframe_n"]
            ):
                raise ValueError(
                    f"DAM keyframe_n disagrees with canonical metadata for {frame_uid}"
                )
            yield {
                "id": point_id,
                "vectors": {"dam": vectors[row]},
                "payload": {
                    "point_id": point_id,
                    "region_id": region_id,
                    "video_id": str(item["video_id"]),
                    "keyframe_n": int(frame_record["keyframe_n"]),
                    "frame_idx": int(item["frame_idx"]),
                    "frame_uid": frame_uid,
                    "parent_point_id": int(frame_record["point_id"]),
                    "bbox": list(item.get("bbox") or []),
                    "class_entity": str(item.get("class_entity", "")),
                    "description_en": str(item.get("description_en", "")),
                    "ingest_schema_version": POINT_SCHEMA_VERSION,
                },
            }


def ingest_frames(
    client: QdrantClient, base_url: str, data_root: Path, batch_size: int, repair: bool = True
) -> int:
    return reconcile_collection(
        client,
        base_url,
        FRAME_COLLECTION,
        EXPECTED_FRAMES,
        {"siglip2": 768, "metaclip2": 1024},
        _frame_sources(data_root),
        batch_size,
        repair,
    )


def _validate_finite_matrix(matrix: np.ndarray, path: Path, chunk_size: int = 8192) -> None:
    for start in range(0, matrix.shape[0], chunk_size):
        chunk = np.asarray(matrix[start : start + chunk_size])
        if not np.isfinite(chunk).all():
            raise ValueError(f"{path}: contains NaN or infinite values near row {start}")


def ingest_beit3(
    client: QdrantClient,
    base_url: str,
    data_root: Path,
    artifact_dir: Path | None,
    batch_size: int,
    repair: bool = True,
) -> int:
    return reconcile_collection(
        client,
        base_url,
        BEIT3_COLLECTION,
        EXPECTED_FRAMES,
        {"beit3": 768},
        _beit3_sources(data_root, artifact_dir),
        batch_size,
        repair,
    )


def ingest_dam(
    client: QdrantClient, base_url: str, data_root: Path, batch_size: int, repair: bool = True
) -> int:
    return reconcile_collection(
        client,
        base_url,
        DAM_COLLECTION,
        EXPECTED_DAM_REGIONS,
        {"dam": 1024},
        _dam_sources(data_root),
        batch_size,
        repair,
    )


def main() -> int:
    args = parse_args()
    RECONCILIATION_REPORTS.clear()
    if args.recreate and args.verify_only:
        raise ValueError("--recreate cannot be combined with --verify-only")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    wait_for_qdrant(args.url)
    client = QdrantClient(
        host=args.grpc_host,
        port=6333,
        grpc_port=args.grpc_port,
        prefer_grpc=True,
        timeout=180,
    )

    selected_frames = args.only in ("all", "frames")
    selected_beit3 = args.only in ("all", "beit3")
    selected_dam = args.only in ("all", "dam")
    selected_names = tuple(
        name
        for name, enabled in (
            ("frames", selected_frames),
            ("beit3", selected_beit3),
            ("dam", selected_dam),
        )
        if enabled
    )
    write_ingestion_manifest(
        args.state_root,
        args.data_root,
        {},
        selected_names,
        args.beit3_dir,
        status="validating",
    )
    collection_results: dict[str, int] = {}
    if selected_frames:
        ensure_frame_collection(args.url, args.recreate, args.verify_only)
        count = ingest_frames(
            client, args.url, args.data_root, args.frame_batch_size, not args.verify_only
        )
        collection_results[FRAME_COLLECTION] = count
        LOGGER.info("Frame collection count: %s", count)
        if not args.skip_index and not args.verify_only:
            enable_hnsw(args.url, FRAME_COLLECTION)
    if selected_beit3:
        ensure_beit3_collection(args.url, args.recreate, args.verify_only)
        count = ingest_beit3(
            client,
            args.url,
            args.data_root,
            args.beit3_dir,
            args.frame_batch_size,
            not args.verify_only,
        )
        collection_results[BEIT3_COLLECTION] = count
        LOGGER.info("BEiT-3 collection count: %s", count)
        if not args.skip_index and not args.verify_only:
            enable_hnsw(args.url, BEIT3_COLLECTION)
    if selected_dam:
        ensure_dam_collection(args.url, args.recreate, args.verify_only)
        count = ingest_dam(
            client, args.url, args.data_root, args.dam_batch_size, not args.verify_only
        )
        collection_results[DAM_COLLECTION] = count
        LOGGER.info("DAM collection count: %s", count)
        if not args.skip_index and not args.verify_only:
            enable_hnsw(args.url, DAM_COLLECTION)
    write_ingestion_manifest(
        args.state_root,
        args.data_root,
        collection_results,
        selected_names,
        args.beit3_dir,
        status="ready",
        verification=dict(RECONCILIATION_REPORTS),
    )
    LOGGER.info("Ingestion completed successfully")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; rerun the same command to resume")
        sys.exit(130)
    except Exception as error:
        try:
            failed_args = parse_args()
            failed_selected = tuple(
                name
                for name, enabled in (
                    ("frames", failed_args.only in ("all", "frames")),
                    ("beit3", failed_args.only in ("all", "beit3")),
                    ("dam", failed_args.only in ("all", "dam")),
                )
                if enabled
            )
            write_ingestion_manifest(
                failed_args.state_root,
                failed_args.data_root,
                {},
                failed_selected,
                failed_args.beit3_dir,
                status="failed",
                error=f"{type(error).__name__}: {error}",
                verification=dict(RECONCILIATION_REPORTS),
            )
        except Exception:
            LOGGER.exception("Could not write failed ingestion manifest")
        raise
