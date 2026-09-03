"""Fail-closed readiness checks for Branch-1 data, Qdrant, and encoders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...encoders.sequential_manager import SequentialBranch1Encoders
from ...infrastructure.qdrant import QdrantHttpClient
from ...infrastructure.resources import current_process_rss_bytes, resource_qualification


EXPECTED_FRAMES = 247_956
DATA_GATE_SCHEMA_VERSION = "branch1.data-gate.v4"
COMPATIBILITY_SCHEMA_VERSION = "branch1.encoder-compatibility.v2"
INGEST_MANIFEST_SCHEMA_VERSION = "qdrant.ingestion.v3"
POINT_SCHEMA_VERSION = "aic.ingest.v2"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ingestion_status(
    state_root: Path,
    collection: str,
    expected_count: int,
    data_root: Path | None = None,
    artifact_paths: tuple[Path, ...] = (),
    expected_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    path = state_root / "qdrant_ingestion_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return {"ready": False, "checks": {"manifest_object": False}, "manifest": None}
        collections = manifest.get("collections")
        if not isinstance(collections, dict):
            return {"ready": False, "checks": {"collections_object": False}, "manifest": manifest}
        count = int(collections.get(collection, -1))
        checks = {
            "manifest_passed": manifest.get("passed") is True,
            "manifest_status": manifest.get("status") == "ready",
            "manifest_schema_version": manifest.get("schema_version") == INGEST_MANIFEST_SCHEMA_VERSION,
            "point_schema_version": manifest.get("ingest_schema_version") == POINT_SCHEMA_VERSION,
            "collection_count": count == expected_count,
        }
        if data_root is not None and artifact_paths:
            artifacts = manifest.get("artifacts") or {}
            files_match = True
            for artifact in artifact_paths:
                try:
                    key = artifact.relative_to(data_root).as_posix()
                except ValueError:
                    key = artifact.resolve().as_posix()
                stat = artifact.stat()
                # v3 writers use POSIX keys so a manifest can move between
                # Windows and Linux.  Accept an equivalent legacy separator
                # only as a compatibility read; the schema/fingerprint gates
                # still decide whether the manifest is trusted.
                recorded = artifacts.get(key)
                if recorded is None:
                    try:
                        recorded = artifacts.get(str(artifact.relative_to(data_root)))
                    except ValueError:
                        recorded = None
                recorded = recorded or {}
                files_match = files_match and (
                    int(recorded.get("size", -1)) == stat.st_size
                    and int(recorded.get("mtime_ns", -1)) == stat.st_mtime_ns
                )
                if expected_sha256 is not None:
                    files_match = files_match and recorded.get("sha256") == expected_sha256.get(key)
            checks["artifacts_match"] = files_match
        verification = (manifest.get("verification") or {}).get(collection) or {}
        checks["verification_evidence"] = (
            verification.get("expected_count") == expected_count
            and verification.get("verified_count") == expected_count
            and verification.get("payload_verified") is True
            and verification.get("vector_content_verified") is True
            and bool(verification.get("completed_at"))
        )
        threshold = verification.get("verification_threshold") or {}
        checks["verification_threshold"] = (
            threshold.get("cosine_min") == 0.99999
            and threshold.get("max_abs_error") == 0.002
        )
        return {"ready": all(checks.values()), "checks": checks, "manifest": manifest, "verification": verification}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"ready": False, "checks": {}, "manifest": None}


def _manifest_status(path: Path, family: str, dimension: int, model_id: str) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return {"ready": False, "checks": {"manifest_object": False}}
        checks = {
            "family": manifest.get("model_family") == family,
            "count": manifest.get("keyframe_count") == EXPECTED_FRAMES,
            "dimension": manifest.get("embedding_dimension") == dimension,
            "dtype": manifest.get("dtype") == "float16",
            "l2_normalized": manifest.get("l2_normalized") is True,
            "model_id": manifest.get("model_id") == model_id,
        }
        return {"ready": all(checks.values()), "checks": checks, "manifest": manifest}
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return {"ready": False, "error": str(error), "checks": {}}


def _gate_model_status(
    gate: dict[str, Any] | None,
    model: str,
    dimension: int,
) -> dict[str, Any]:
    if not isinstance(gate, dict):
        return {"ready": False, "checks": {"gate_object": False}, "gate": gate}
    section = gate.get(model) or {}
    if not isinstance(section, dict):
        return {"ready": False, "checks": {"section": False}, "gate": gate}
    matrix = section.get("matrix") or section
    if not isinstance(matrix, dict):
        return {"ready": False, "checks": {"matrix": False}, "gate": gate}
    checks = {
        "gate_passed": gate.get("passed") is True,
        "schema_version": gate.get("schema_version") == DATA_GATE_SCHEMA_VERSION,
        "vector_count": _safe_int(section.get("vector_count", matrix.get("shape", [0])[0] if matrix.get("shape") else 0)) == EXPECTED_FRAMES,
        "dimension": _safe_int(section.get("dimension", matrix.get("shape", [0, 0])[1] if matrix.get("shape") else 0)) == dimension,
        "dtype": section.get("dtype", matrix.get("dtype")) == "float16",
        "finite_verified": (
            section.get("finite_verified") is True
            if "finite_verified" in section
            else isinstance(matrix.get("min_norm"), (int, float))
            and isinstance(matrix.get("max_norm"), (int, float))
        ),
        "l2_normalized": (
            section.get("l2_normalized") is True
            if "l2_normalized" in section
            else isinstance(matrix.get("min_norm"), (int, float))
            and isinstance(matrix.get("max_norm"), (int, float))
            and float(matrix.get("min_norm")) >= 0.995
            and float(matrix.get("max_norm")) <= 1.005
        ),
        "ordering_verified": section.get("ordering_verified") is True,
        "offline_identity_recorded": (
            isinstance(section.get("offline_identity"), dict)
            and "revision_verified" in section["offline_identity"]
            and bool(section["offline_identity"].get("evidence"))
        ),
    }
    if model != "siglip2":
        checks["metadata_rows"] = _safe_int(section.get("metadata_rows", 0)) == EXPECTED_FRAMES
        checks["index_rows"] = _safe_int(section.get("index_rows", 0)) == EXPECTED_FRAMES
    return {"ready": all(checks.values()), "checks": checks, "gate": gate}


def _offline_identity(section: dict[str, Any] | None) -> dict[str, Any]:
    value = (section or {}).get("offline_identity") if isinstance(section, dict) else None
    if not isinstance(value, dict):
        return {
            "ready": False,
            "recorded": False,
            "revision_verified": False,
            "reason": "offline identity is missing from the data gate",
        }
    return {
        "ready": True,
        "recorded": True,
        "revision_verified": value.get("revision_verified") is True,
        "model_id": value.get("model_id"),
        "immutable_revision": value.get("immutable_revision"),
        "checkpoint_sha256": value.get("checkpoint_sha256"),
        "source_revision": value.get("source_revision"),
        "evidence": value.get("evidence"),
        "reason": value.get("unverified_reason"),
    }


def _gate_metadata_current(gate: dict[str, Any] | None, path: Path) -> bool:
    if not isinstance(gate, dict):
        return False
    expected = gate.get("canonical_metadata") or {}
    try:
        stat = path.stat()
        return (
            int(expected.get("size", -1)) == stat.st_size
            and int(expected.get("mtime_ns", -1)) == stat.st_mtime_ns
        )
    except (OSError, TypeError, ValueError):
        return False


def _gate_artifacts_current(gate: dict[str, Any] | None, data_root: Path) -> bool:
    if not isinstance(gate, dict):
        return False
    try:
        metaclip = (gate or {})["metaclip2"]["matrix"]
        beit3 = (gate or {})["beit3"]["matrix"]
        for section, path in (
            (metaclip, data_root / "visual_embeddings" / "metaclip2" / "keyframes_visual_vectors.f16.npy"),
            (beit3, data_root / "visual_embeddings" / "beit3" / "keyframes_visual_vectors.f16.npy"),
        ):
            stat = path.stat()
            if (
                int(section.get("size", -1)) != stat.st_size
                or int(section.get("mtime_ns", -1)) != stat.st_mtime_ns
                or not section.get("sha256")
            ):
                return False
        canonical_path = data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
        canonical = (gate or {}).get("canonical_metadata") or {}
        canonical_stat = canonical_path.stat()
        if (
            int(canonical.get("size", -1)) != canonical_stat.st_size
            or int(canonical.get("mtime_ns", -1)) != canonical_stat.st_mtime_ns
            or not canonical.get("sha256")
        ):
            return False
        beit_metadata = (gate or {}).get("beit3", {}).get("metadata") or {}
        beit_metadata_path = data_root / "visual_embeddings" / "beit3" / "keyframes_metadata.jsonl"
        beit_metadata_stat = beit_metadata_path.stat()
        if (
            int(beit_metadata.get("size", -1)) != beit_metadata_stat.st_size
            or int(beit_metadata.get("mtime_ns", -1)) != beit_metadata_stat.st_mtime_ns
            or not beit_metadata.get("sha256")
        ):
            return False
        shards = (gate or {}).get("siglip2", {}).get("shards") or []
        if not shards:
            return False
        for shard in shards:
            path = data_root / str(shard["path"])
            stat = path.stat()
            if (
                int(shard.get("size", -1)) != stat.st_size
                or int(shard.get("mtime_ns", -1)) != stat.st_mtime_ns
                or not shard.get("sha256")
            ):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, AttributeError):
        return False


def _collection_status(
    qdrant: QdrantHttpClient, collection: str, vector_name: str, dimension: int,
    expected_count: int = EXPECTED_FRAMES,
) -> dict[str, Any]:
    try:
        value = qdrant.collection(collection)
        vector = value["config"]["params"]["vectors"][vector_name]
        # Qdrant's collection-info points_count is an approximate telemetry
        # value and may temporarily over-count after a large reconciliation.
        # Capability readiness must use the exact count endpoint instead.
        approximate_count = int(value.get("points_count", -1))
        exact_count = qdrant.count(collection)
        checks = {
            "green": value.get("status") == "green",
            "count": exact_count == expected_count,
            "dimension": int(vector.get("size", -1)) == dimension,
            "distance": vector.get("distance") == "Cosine",
        }
        schema_filter = {
            "must": [
                {
                    "key": "ingest_schema_version",
                    "match": {"value": POINT_SCHEMA_VERSION},
                }
            ]
        }
        checks["schema_coverage"] = qdrant.count(collection, schema_filter) == expected_count
        return {
            "ready": all(checks.values()),
            "checks": checks,
            "points_count": exact_count,
            "exact_points_count": exact_count,
            "approximate_points_count": approximate_count,
        }
    except Exception as error:
        return {"ready": False, "error": str(error), "checks": {}}


def _gate_fingerprints(
    gate: dict[str, Any] | None,
    data_root: Path,
    artifact_paths: tuple[Path, ...],
) -> dict[str, str]:
    """Map current ingestion paths to hashes published by the data gate."""
    if not isinstance(gate, dict):
        return {}
    records: dict[str, str] = {}

    def key_for(path: Path) -> str:
        try:
            return path.relative_to(data_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def add(path: Path, value: Any) -> None:
        if isinstance(value, dict) and value.get("sha256"):
            records[key_for(path)] = str(value["sha256"])

    add(
        data_root / "visual_embeddings" / "metaclip2" / "keyframes_visual_vectors.f16.npy",
        (gate.get("metaclip2") or {}).get("matrix"),
    )
    add(
        data_root / "visual_embeddings" / "beit3" / "keyframes_visual_vectors.f16.npy",
        (gate.get("beit3") or {}).get("matrix"),
    )
    add(
        data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl",
        gate.get("canonical_metadata"),
    )
    add(
        data_root / "visual_embeddings" / "beit3" / "keyframes_metadata.jsonl",
        (gate.get("beit3") or {}).get("metadata"),
    )
    shard_by_path = {
        str(item.get("path")).replace("\\", "/"): item
        for item in ((gate.get("siglip2") or {}).get("shards") or [])
        if isinstance(item, dict)
    }
    for path in artifact_paths:
        relative = key_for(path)
        if relative in shard_by_path:
            add(path, shard_by_path[relative])
    wanted = {key_for(path) for path in artifact_paths}
    return {key: value for key, value in records.items() if key in wanted}


def branch1_health(
    data_root: Path,
    qdrant: QdrantHttpClient,
    encoders: SequentialBranch1Encoders,
    state_root: Path | None = None,
) -> dict[str, Any]:
    metaclip_dir = data_root / "visual_embeddings" / "metaclip2"
    beit3_dir = data_root / "visual_embeddings" / "beit3"
    gate_root = state_root or (data_root / "visual_embeddings")
    ingestion_root = state_root or (data_root / "visual_embeddings")
    data_gate_path = gate_root / "branch1_data_gate.json"
    compatibility_path = gate_root / "branch1_encoder_compatibility.json"
    try:
        data_gate = json.loads(data_gate_path.read_text(encoding="utf-8"))
        if not isinstance(data_gate, dict):
            raise ValueError("Branch-1 data gate must be a JSON object")
        data_gate_ready = (
            data_gate.get("passed") is True
            and data_gate.get("status") == "ready"
            and data_gate.get("schema_version") == DATA_GATE_SCHEMA_VERSION
            and _gate_metadata_current(
                data_gate,
                data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl",
            )
            and _gate_artifacts_current(data_gate, data_root)
        )
    except (OSError, ValueError, TypeError):
        data_gate, data_gate_ready = None, False
    try:
        compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
        if not isinstance(compatibility, dict):
            raise ValueError("Branch-1 compatibility gate must be a JSON object")
        compatibility_ready = (
            compatibility.get("passed") is True
            and compatibility.get("schema_version") == COMPATIBILITY_SCHEMA_VERSION
            and compatibility.get("text_encoder_contract", {}).get("siglip2", {}).get("languages") == ["vi", "en"]
            and compatibility.get("text_encoder_contract", {}).get("metaclip2", {}).get("languages") == ["vi", "en"]
            and compatibility.get("text_encoder_contract", {}).get("beit3", {}).get("languages") == ["en"]
        )
    except (OSError, ValueError, TypeError, AttributeError):
        compatibility, compatibility_ready = None, False

    encoder_health = encoders.health()
    siglip_collection = _collection_status(qdrant, "aic_frames", "siglip2", 768)
    siglip_gate = _gate_model_status(data_gate, "siglip2", 768)
    metaclip_gate = _gate_model_status(data_gate, "metaclip2", 1024)
    beit_gate = _gate_model_status(data_gate, "beit3", 768)
    frame_artifacts = (
        data_root / "visual_embeddings" / "metaclip2" / "keyframes_visual_vectors.f16.npy",
        data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl",
        *tuple(
            sorted((data_root / "scene_embeddings").glob("*.safetensors"))
        ),
    )
    beit_artifacts = (
        data_root / "visual_embeddings" / "beit3" / "keyframes_visual_vectors.f16.npy",
        data_root / "visual_embeddings" / "beit3" / "keyframes_metadata.jsonl",
    )
    frame_fingerprints = _gate_fingerprints(data_gate, data_root, frame_artifacts)
    beit_fingerprints = _gate_fingerprints(data_gate, data_root, beit_artifacts)
    models = {
        "siglip2": {
            "data": siglip_gate,
            "collection": siglip_collection,
            "ingestion": _ingestion_status(
                ingestion_root,
                "aic_frames",
                EXPECTED_FRAMES,
                data_root,
                frame_artifacts,
                frame_fingerprints,
            ),
            "text_encoder": encoder_health.get("siglip2", {"ready": False}),
            "offline_identity": _offline_identity((data_gate or {}).get("siglip2")),
        },
        "metaclip2": {
            "data": {
                "ready": metaclip_gate.get("ready") is True and _manifest_status(
                    metaclip_dir / "run_manifest.json",
                    "metaclip2",
                    1024,
                    "facebook/metaclip-2-worldwide-huge-quickgelu",
                ).get("ready") is True,
                "data_gate": metaclip_gate,
                "manifest": _manifest_status(
                    metaclip_dir / "run_manifest.json",
                    "metaclip2",
                    1024,
                    "facebook/metaclip-2-worldwide-huge-quickgelu",
                ),
            },
            "collection": _collection_status(qdrant, "aic_frames", "metaclip2", 1024),
            "ingestion": _ingestion_status(
                ingestion_root,
                "aic_frames",
                EXPECTED_FRAMES,
                data_root,
                frame_artifacts,
                frame_fingerprints,
            ),
            "text_encoder": encoder_health.get("metaclip2", {"ready": False}),
            "offline_identity": _offline_identity((data_gate or {}).get("metaclip2")),
        },
        "beit3": {
            "data": {
                "ready": beit_gate.get("ready") is True and _manifest_status(
                    beit3_dir / "run_manifest.json",
                    "beit3",
                    768,
                    "https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth",
                ).get("ready") is True,
                "data_gate": beit_gate,
                "manifest": _manifest_status(
                    beit3_dir / "run_manifest.json",
                    "beit3",
                    768,
                    "https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth",
                ),
            },
            "collection": _collection_status(qdrant, "aic_beit3_frames", "beit3", 768),
            "ingestion": _ingestion_status(
                ingestion_root,
                "aic_beit3_frames",
                EXPECTED_FRAMES,
                data_root,
                beit_artifacts,
                beit_fingerprints,
            ),
            "text_encoder": encoder_health.get("beit3", {"ready": False}),
            "offline_identity": _offline_identity((data_gate or {}).get("beit3")),
        },
    }
    models_ready = all(
        all(section.get("ready") is True for section in model.values()) for model in models.values()
    )
    ready = data_gate_ready and compatibility_ready and models_ready
    manager = getattr(encoders, "manager", None)
    memory_ready = manager is None or manager.production_ready
    resource_state = resource_qualification(gate_root)
    peak_worker_rss = 0 if manager is None else int(manager.peak_worker_rss_bytes)
    estimated_peak_total_rss = 0 if manager is None else int(manager.estimated_peak_total_rss_bytes)
    provenance_verified = all(
        model["offline_identity"].get("revision_verified") is True
        for model in models.values()
    )
    return {
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "production_ready": (
            ready
            and provenance_verified
            and memory_ready
            and resource_state.get("production_ready") is True
        ),
        "fail_closed": True,
        "api_rss_bytes": current_process_rss_bytes(),
        "peak_worker_rss_bytes": peak_worker_rss,
        "estimated_peak_total_rss_bytes": estimated_peak_total_rss,
        "data_gate": {"ready": data_gate_ready, "report": data_gate},
        "encoder_compatibility": {"ready": compatibility_ready, "report": compatibility},
        "resource_qualification": resource_state,
        "offline_provenance_verified": provenance_verified,
        "models": models,
    }
