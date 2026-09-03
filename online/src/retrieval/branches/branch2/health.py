"""Fail-closed readiness checks for Branch 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..branch1.health import _collection_status, _gate_fingerprints, _ingestion_status, _offline_identity
from ...encoders.sequential_manager import SequentialBranch1Encoders
from ...encoders.cpu import CpuTextEncoders
from ...infrastructure.qdrant import QdrantHttpClient
from ...infrastructure.resources import current_process_rss_bytes, resource_qualification
from ..branch1.contracts import EXPECTED_FRAMES
from .dense import DamDenseRetriever
from .sparse import DamBm25Index


def branch2_health(
    data_root: Path,
    qdrant: QdrantHttpClient,
    dense: DamDenseRetriever,
    sparse: DamBm25Index,
    bge_encoders: CpuTextEncoders,
    beit_encoders: SequentialBranch1Encoders,
    state_root: Path | None = None,
) -> dict[str, Any]:
    dam_collection = _collection_status(qdrant, "aic_dam_regions", "dam", 1024, expected_count=681_355)
    beit_collection = _collection_status(qdrant, "aic_beit3_frames", "beit3", 768, expected_count=EXPECTED_FRAMES)
    ingestion_root = state_root or (data_root / "visual_embeddings")
    beit_health_raw = beit_encoders.health()
    beit_health = beit_health_raw if isinstance(beit_health_raw, dict) else {}
    encoder = beit_health.get("beit3", {"ready": False})
    dam_data = dense.health()
    bge_health = bge_encoders.health()
    bge_encoder = bge_health if isinstance(bge_health, dict) else {"ready": False}
    expected_bge_revision = str(dam_data.get("online_revision") or "")
    actual_bge_revision = str(bge_encoder.get("revision") or "")
    bge_compatible = bool(expected_bge_revision) and expected_bge_revision == actual_bge_revision
    bge_compatibility = {
        "ready": bge_compatible,
        "expected_revision": expected_bge_revision,
        "actual_revision": actual_bge_revision,
        "warnings": [] if bge_compatible else ["Online BGE-M3 revision does not match the DAM migration manifest"],
    }
    frame_mapping_path = data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
    try:
        frame_stat = frame_mapping_path.stat()
        frame_files_match = (
            frame_stat.st_size == int(dam_data.get("frame_metadata_size", -1))
            and frame_stat.st_mtime_ns == int(dam_data.get("frame_metadata_mtime_ns", -1))
        )
    except (OSError, TypeError, ValueError):
        frame_files_match = False
    frame_mapping = {
        "ready": (
            frame_mapping_path.is_file()
            and int(dam_data.get("frame_metadata_count", 0)) == EXPECTED_FRAMES
            and dam_data.get("frame_metadata_identity_verified") is True
            and frame_files_match
        ),
        "path": str(frame_mapping_path),
        "validated_on_search": True,
        "files_match_manifest": frame_files_match,
    }
    dam_artifacts = (
        data_root / "dense_text_embeddings" / "dam_vectors.f16.npy",
        data_root / "dense_text_embeddings" / "dam_metadata.jsonl",
        data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl",
    )
    dam_fingerprints = {
        path.relative_to(data_root).as_posix(): value
        for path, value in (
            (dam_artifacts[0], dam_data.get("matrix_sha256")),
            (dam_artifacts[1], dam_data.get("metadata_sha256")),
            (dam_artifacts[2], dam_data.get("frame_metadata_sha256")),
        )
        if value
    }
    try:
        gate = json.loads((ingestion_root / "branch1_data_gate.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        gate = None
    beit_artifacts = (
        data_root / "visual_embeddings" / "beit3" / "keyframes_visual_vectors.f16.npy",
        data_root / "visual_embeddings" / "beit3" / "keyframes_metadata.jsonl",
    )
    beit_fingerprints = _gate_fingerprints(gate, data_root, beit_artifacts)
    beit_offline_identity = _offline_identity((gate or {}).get("beit3"))
    parts = {
        "dam_data": dam_data,
        "dam_collection": dam_collection,
        "dam_ingestion": _ingestion_status(
            ingestion_root,
            "aic_dam_regions",
            681_355,
            data_root,
            (
                data_root / "dense_text_embeddings" / "dam_vectors.f16.npy",
                data_root / "dense_text_embeddings" / "dam_metadata.jsonl",
                data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl",
            ),
            dam_fingerprints,
        ),
        "bm25": sparse.health(),
        "bge_text_encoder": bge_encoder,
        "bge_compatibility": bge_compatibility,
        "beit3_collection": beit_collection,
        "beit3_ingestion": _ingestion_status(
            ingestion_root,
            "aic_beit3_frames",
            EXPECTED_FRAMES,
            data_root,
            (
                data_root / "visual_embeddings" / "beit3" / "keyframes_visual_vectors.f16.npy",
                data_root / "visual_embeddings" / "beit3" / "keyframes_metadata.jsonl",
            ),
            beit_fingerprints,
        ),
        "beit3_text_encoder": encoder,
        "beit3_offline_identity": {
            "ready": beit_offline_identity.get("recorded") is True,
            "production_ready": beit_offline_identity.get("revision_verified") is True,
            **beit_offline_identity,
        },
        "frame_mapping": frame_mapping,
    }
    ready = all(value.get("ready") is True for value in parts.values())
    warnings = [
        warning
        for value in parts.values()
        for warning in value.get("warnings", [])
    ]
    managers = [getattr(bge_encoders, "manager", None), getattr(beit_encoders, "manager", None)]
    peak_worker_rss = max(
        (int(manager.peak_worker_rss_bytes) for manager in managers if manager is not None),
        default=0,
    )
    memory_ready = all(
        manager is None or manager.production_ready for manager in managers
    )
    resource_state = resource_qualification(ingestion_root)
    estimated_peak_total_rss = max(
        (int(manager.estimated_peak_total_rss_bytes) for manager in managers if manager is not None),
        default=0,
    )
    production_ready = ready and memory_ready and resource_state.get("production_ready") is True and all(
        value.get("production_ready", value.get("ready")) is True for value in parts.values()
    )
    return {
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "production_ready": production_ready,
        "fail_closed": True,
        "warnings": warnings,
        "api_rss_bytes": current_process_rss_bytes(),
        "resource_qualification": resource_state,
        "peak_worker_rss_bytes": peak_worker_rss,
        "estimated_peak_total_rss_bytes": estimated_peak_total_rss,
        "components": parts,
    }
