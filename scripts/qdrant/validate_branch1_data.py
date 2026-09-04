#!/usr/bin/env python3
"""Strict local data gate for Branch-1 visual embedding artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

EXPECTED_FRAMES = 247_956
IDENTITY_FIELDS = ("point_id", "frame_uid", "video_id", "frame_idx", "keyframe_n")
DATA_GATE_SCHEMA_VERSION = "branch1.data-gate.v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--beit3-dir", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def load_manifest(directory: Path, family: str, dimension: int) -> dict[str, Any]:
    path = directory / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_model_id = {
        "metaclip2": "facebook/metaclip-2-worldwide-huge-quickgelu",
        "beit3": (
            "https://github.com/addf400/files/releases/download/beit3/"
            "beit3_base_patch16_384_coco_retrieval.pth"
        ),
    }.get(family)
    expected = {
        "model_family": family,
        "keyframe_count": EXPECTED_FRAMES,
        "embedding_dimension": dimension,
        "dtype": "float16",
        "l2_normalized": True,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"{path}: {field}={manifest.get(field)!r}, expected {value!r}")
    if expected_model_id is not None and manifest.get("model_id") != expected_model_id:
        raise ValueError(
            f"{path}: model_id={manifest.get('model_id')!r}, expected {expected_model_id!r}"
        )
    return manifest


def validate_matrix(directory: Path, dimension: int) -> dict[str, Any]:
    path = directory / "keyframes_visual_vectors.f16.npy"
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (EXPECTED_FRAMES, dimension):
        raise ValueError(f"{path}: unexpected shape {matrix.shape}")
    if matrix.dtype != np.float16:
        raise ValueError(f"{path}: unexpected dtype {matrix.dtype}")
    min_norm = float("inf")
    max_norm = 0.0
    for start in range(0, EXPECTED_FRAMES, 8192):
        chunk = np.asarray(matrix[start : start + 8192], dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise ValueError(f"{path}: non-finite value near row {start}")
        norms = np.linalg.norm(chunk, axis=1)
        min_norm = min(min_norm, float(norms.min()))
        max_norm = max(max_norm, float(norms.max()))
    if min_norm < 0.995 or max_norm > 1.005:
        raise ValueError(f"{path}: vectors are not L2-normalized ({min_norm}, {max_norm})")
    return {
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "min_norm": min_norm,
        "max_norm": max_norm,
        "finite_verified": True,
        "l2_normalized": True,
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": sha256_file(path),
    }


def validate_index(directory: Path) -> int:
    path = directory / "keyframe_index.csv"
    metadata_path = directory / "keyframes_metadata.jsonl"
    count = 0
    identity_fields = ("point_id", "frame_uid", "video_id", "frame_idx", "keyframe_n")
    with (
        path.open("r", encoding="utf-8", newline="") as csv_handle,
        metadata_path.open("r", encoding="utf-8") as metadata_handle,
    ):
        rows = csv.DictReader(csv_handle)
        for count, pair in enumerate(itertools.zip_longest(rows, metadata_handle), start=1):
            row, metadata_line = pair
            if row is None or metadata_line is None:
                raise ValueError(f"{path}: CSV and metadata row counts differ")
            metadata = json.loads(metadata_line)
            if int(row["global_vector_row"]) != count - 1:
                raise ValueError(f"{path}: global_vector_row mismatch at CSV row {count + 1}")
            if int(row["point_id"]) != count:
                raise ValueError(f"{path}: point_id mismatch at CSV row {count + 1}")
            for field in identity_fields:
                if str(row.get(field)) != str(metadata.get(field)):
                    raise ValueError(f"{path}: {field} mismatch at CSV row {count + 1}")
    if count != EXPECTED_FRAMES:
        raise ValueError(f"{path}: expected {EXPECTED_FRAMES} rows, found {count}")
    return count


def compare_metadata(canonical_path: Path, candidate_path: Path) -> int:
    count = 0
    frame_uids: set[str] = set()
    with (
        canonical_path.open("r", encoding="utf-8") as canonical_handle,
        candidate_path.open("r", encoding="utf-8") as candidate_handle,
    ):
        for row_number, pair in enumerate(
            itertools.zip_longest(canonical_handle, candidate_handle), start=1
        ):
            canonical_line, candidate_line = pair
            if canonical_line is None or candidate_line is None:
                raise ValueError("Metadata files have different row counts")
            canonical = json.loads(canonical_line)
            candidate = json.loads(candidate_line)
            for field in IDENTITY_FIELDS:
                if canonical.get(field) != candidate.get(field):
                    raise ValueError(
                        f"Metadata mismatch at row {row_number}, field {field}: "
                        f"{canonical.get(field)!r} != {candidate.get(field)!r}"
                    )
            frame_uid = str(candidate["frame_uid"])
            if int(candidate.get("point_id", 0)) != row_number:
                raise ValueError(f"{candidate_path}: point_id mismatch at row {row_number}")
            derived_uid = f"{candidate.get('video_id')}:{int(candidate.get('frame_idx', -1))}"
            if frame_uid != derived_uid:
                raise ValueError(f"{candidate_path}: frame_uid mismatch at row {row_number}")
            if frame_uid in frame_uids:
                raise ValueError(f"Duplicate frame_uid at row {row_number}: {frame_uid}")
            frame_uids.add(frame_uid)
            count = row_number
    if count != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} metadata rows, found {count}")
    return count


def validate_self_metadata(path: Path) -> int:
    return compare_metadata(path, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: Path, relative_to: Path | None = None) -> dict[str, Any]:
    display_path = path
    if relative_to is not None:
        try:
            display_path = path.relative_to(relative_to)
        except ValueError:
            display_path = path
    return {
        "path": display_path.as_posix(),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": sha256_file(path),
    }


def _display_path(path: Path, relative_to: Path) -> str:
    try:
        return path.relative_to(relative_to).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def offline_identity(
    *,
    model_id: str | None,
    evidence: str,
    revision: str | None = None,
    checkpoint_sha256: str | None = None,
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Record what the offline artifact actually proves, not what runtime uses."""
    revision_verified = bool(revision and (checkpoint_sha256 or source_revision))
    return {
        "model_id": model_id,
        "immutable_revision": revision,
        "checkpoint_sha256": checkpoint_sha256,
        "source_revision": source_revision,
        "evidence": evidence,
        "revision_verified": revision_verified,
        "unverified_reason": (
            None
            if revision_verified
            else "offline embedding artifact does not contain immutable model revision and cryptographic source/checkpoint evidence"
        ),
    }


def validate_siglip_shards(data_root: Path, canonical_metadata: Path) -> dict[str, Any]:
    """Validate every SigLIP shard against canonical frame row identity."""
    scene_dir = data_root / "scene_embeddings"
    if not scene_dir.is_dir():
        raise FileNotFoundError(scene_dir)
    used: set[Path] = set()
    reports: list[dict[str, Any]] = []
    current_video: str | None = None
    current_matrix: np.ndarray | None = None
    current_path: Path | None = None
    current_rows = 0
    current_used_rows: set[int] = set()
    total_mapped_rows = 0
    with canonical_metadata.open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            item = json.loads(line)
            video_id = str(item["video_id"])
            if video_id != current_video:
                if current_path is not None and current_matrix is not None:
                    if len(current_used_rows) != current_rows:
                        raise ValueError(f"SigLIP shard row mapping is incomplete: {current_path}")
                    reports.append(
                        {
                            "path": current_path.relative_to(data_root).as_posix(),
                            "rows": int(current_matrix.shape[0]),
                            "size": current_path.stat().st_size,
                            "mtime_ns": current_path.stat().st_mtime_ns,
                            "sha256": sha256_file(current_path),
                        }
                    )
                current_path = scene_dir / f"{video_id}.safetensors"
                tensors = load_file(current_path)
                if set(tensors) != {"embeddings"}:
                    raise ValueError(f"Unexpected tensors in {current_path}: {sorted(tensors)}")
                current_matrix = tensors["embeddings"]
                if current_matrix.ndim != 2 or current_matrix.shape[1] != 768:
                    raise ValueError(
                        f"Invalid SigLIP shard shape {current_path}: {current_matrix.shape}"
                    )
                if current_matrix.dtype != np.float16:
                    raise ValueError(
                        f"Invalid SigLIP shard dtype {current_path}: {current_matrix.dtype}"
                    )
                used.add(current_path.resolve())
                current_video = video_id
                current_rows = int(current_matrix.shape[0])
                current_used_rows = set()
            assert current_matrix is not None
            local_row = int(item["vector_row"])
            if not 0 <= local_row < current_rows:
                raise IndexError(f"SigLIP row {local_row} missing for {video_id}")
            if local_row in current_used_rows:
                raise ValueError(f"Duplicate SigLIP vector_row {local_row} for {video_id}")
            current_used_rows.add(local_row)
            total_mapped_rows += 1
            chunk = np.asarray(current_matrix[local_row], dtype=np.float32)
            if not np.isfinite(chunk).all():
                raise ValueError(f"Non-finite SigLIP vector at canonical row {row + 1}")
            norm = float(np.linalg.norm(chunk))
            if norm < 0.995 or norm > 1.005:
                raise ValueError(
                    f"SigLIP vector at canonical row {row + 1} is not normalized: {norm}"
                )
    if current_path is not None and current_matrix is not None:
        if len(current_used_rows) != current_rows:
            raise ValueError(f"SigLIP shard row mapping is incomplete: {current_path}")
        reports.append(
            {
                "path": current_path.relative_to(data_root).as_posix(),
                "rows": int(current_matrix.shape[0]),
                "size": current_path.stat().st_size,
                "mtime_ns": current_path.stat().st_mtime_ns,
                "sha256": sha256_file(current_path),
            }
        )
    all_shards = {path.resolve() for path in scene_dir.glob("*.safetensors")}
    if all_shards != used:
        unused = sorted(str(path) for path in all_shards - used)
        missing = sorted(str(path) for path in used - all_shards)
        raise ValueError(
            f"SigLIP shard set does not match metadata; unused={unused[:5]}, missing={missing[:5]}"
        )
    if (
        total_mapped_rows != EXPECTED_FRAMES
        or sum(int(report["rows"]) for report in reports) != EXPECTED_FRAMES
    ):
        raise ValueError("SigLIP shard row total does not equal canonical frame count")
    return {
        "shard_count": len(reports),
        "shards": reports,
        "metadata_rows": EXPECTED_FRAMES,
        "vector_count": EXPECTED_FRAMES,
        "dimension": 768,
        "dtype": "float16",
        "finite_verified": True,
        "l2_normalized": True,
        "ordering_verified": True,
    }


def build_data_gate_report(data_root: Path, beit3_dir: Path | None = None) -> dict[str, Any]:
    """Build the single canonical Branch-1 data-gate report.

    Both the standalone validator and the preparation command use this
    function.  Keeping report construction in one place prevents the runtime
    health gate from requiring fields that the preparation command forgot to
    publish.
    """
    metaclip_dir = data_root / "visual_embeddings" / "metaclip2"
    beit3_dir = beit3_dir or data_root / "visual_embeddings" / "beit3"
    canonical_metadata = metaclip_dir / "keyframes_metadata.jsonl"
    metaclip_matrix = metaclip_dir / "keyframes_visual_vectors.f16.npy"
    beit3_matrix = beit3_dir / "keyframes_visual_vectors.f16.npy"
    beit3_metadata = beit3_dir / "keyframes_metadata.jsonl"
    metaclip_matrix_report = validate_matrix(metaclip_dir, 1024)
    metaclip_matrix_report["path"] = _display_path(metaclip_matrix, data_root)
    beit3_matrix_report = validate_matrix(beit3_dir, 768)
    beit3_matrix_report["path"] = _display_path(beit3_matrix, data_root)
    siglip_report = validate_siglip_shards(data_root, canonical_metadata)
    siglip_report["offline_identity"] = offline_identity(
        model_id=None,
        evidence="scene_embeddings/*.safetensors has no offline model manifest",
    )
    metaclip_manifest = load_manifest(metaclip_dir, "metaclip2", 1024)
    beit3_manifest = load_manifest(beit3_dir, "beit3", 768)
    return {
        "schema_version": DATA_GATE_SCHEMA_VERSION,
        "status": "ready",
        "siglip2": siglip_report,
        "metaclip2": {
            "manifest": metaclip_manifest,
            "offline_identity": offline_identity(
                model_id=str(metaclip_manifest.get("model_id") or "") or None,
                evidence="visual_embeddings/metaclip2/run_manifest.json",
                revision=(
                    str(metaclip_manifest.get("revision"))
                    if metaclip_manifest.get("revision")
                    else None
                ),
                checkpoint_sha256=(
                    str(metaclip_manifest.get("checkpoint_sha256"))
                    if metaclip_manifest.get("checkpoint_sha256")
                    else None
                ),
            ),
            "matrix": metaclip_matrix_report,
            "vector_count": EXPECTED_FRAMES,
            "dimension": 1024,
            "dtype": "float16",
            "finite_verified": True,
            "l2_normalized": True,
            "metadata_rows": validate_self_metadata(canonical_metadata),
            "metadata": file_fingerprint(canonical_metadata, data_root),
            "index_rows": validate_index(metaclip_dir),
            "ordering_verified": True,
        },
        "beit3": {
            "manifest": beit3_manifest,
            "offline_identity": offline_identity(
                model_id=str(beit3_manifest.get("model_id") or "") or None,
                evidence="visual_embeddings/beit3/run_manifest.json",
                revision=(
                    str(beit3_manifest.get("checkpoint_revision"))
                    if beit3_manifest.get("checkpoint_revision")
                    else None
                ),
                checkpoint_sha256=(
                    str(beit3_manifest.get("checkpoint_sha256"))
                    if beit3_manifest.get("checkpoint_sha256")
                    else None
                ),
                source_revision=(
                    str(beit3_manifest.get("unilm_revision"))
                    if beit3_manifest.get("unilm_revision")
                    else None
                ),
            ),
            "matrix": beit3_matrix_report,
            "vector_count": EXPECTED_FRAMES,
            "dimension": 768,
            "dtype": "float16",
            "finite_verified": True,
            "l2_normalized": True,
            "metadata_rows": compare_metadata(canonical_metadata, beit3_metadata),
            "metadata": file_fingerprint(beit3_metadata, data_root),
            "index_rows": validate_index(beit3_dir),
            "ordering_verified": True,
        },
        "passed": True,
        "keyframe_count": EXPECTED_FRAMES,
        "canonical_metadata": {
            "path": _display_path(canonical_metadata, data_root),
            "size": canonical_metadata.stat().st_size,
            "mtime_ns": canonical_metadata.stat().st_mtime_ns,
            "sha256": sha256_file(canonical_metadata),
            "rows": EXPECTED_FRAMES,
        },
    }


def main() -> int:
    args = parse_args()
    beit3_dir = args.beit3_dir or args.data_root / "visual_embeddings" / "beit3"

    result = build_data_gate_report(args.data_root, beit3_dir)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        staging = args.report.with_suffix(args.report.suffix + ".staging")
        staging.write_text(payload, encoding="utf-8")
        os.replace(staging, args.report)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
