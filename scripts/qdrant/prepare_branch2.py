"""Prepare immutable DAM manifest and atomic BM25 index for Branch 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# Keep the preparation command runnable both as ``python path/to/script.py``
# and from the CPU API image, where the repository root is /app.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from online.src.retrieval.branches.branch1.contracts import EXPECTED_FRAMES  # noqa: E402
from online.src.retrieval.branches.branch2.contracts import EXPECTED_DAM_REGIONS  # noqa: E402
from online.src.retrieval.branches.branch2.sparse import DamBm25Index  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_and_manifest(data_root: Path, state_root: Path) -> dict[str, object]:
    dense_root = data_root / "dense_text_embeddings"
    matrix_path = dense_root / "dam_vectors.f16.npy"
    metadata_path = dense_root / "dam_metadata.jsonl"
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (EXPECTED_DAM_REGIONS, 1024) or matrix.dtype != np.float16:
        raise ValueError(f"Invalid DAM matrix shape/dtype: {matrix.shape} {matrix.dtype}")
    finite = True
    min_norm = float("inf")
    max_norm = 0.0
    for start in range(0, EXPECTED_DAM_REGIONS, 8192):
        chunk = np.asarray(matrix[start : start + 8192], dtype=np.float32)
        if not np.isfinite(chunk).all():
            finite = False
            break
        norms = np.linalg.norm(chunk, axis=1)
        min_norm = min(min_norm, float(norms.min()))
        max_norm = max(max_norm, float(norms.max()))
    if not finite:
        raise ValueError("DAM matrix contains non-finite values")
    if min_norm < 0.995 or max_norm > 1.005:
        raise ValueError(f"DAM vectors are not L2-normalized ({min_norm}, {max_norm})")
    metadata_count = 0
    frame_metadata_path = data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
    frame_records: dict[str, dict[str, object]] = {}
    frame_count = 0
    with frame_metadata_path.open("r", encoding="utf-8") as handle:
        for frame_count, line in enumerate(handle, 1):
            frame = json.loads(line)
            point_id = int(frame.get("point_id", 0))
            frame_uid = str(frame.get("frame_uid", ""))
            derived_uid = f"{frame.get('video_id')}:{int(frame.get('frame_idx', -1))}"
            if point_id != frame_count or not frame_uid or frame_uid != derived_uid:
                raise ValueError(f"Canonical frame metadata identity mismatch at row {frame_count}")
            if frame_uid in frame_records:
                raise ValueError(f"Duplicate canonical frame_uid at row {frame_count}: {frame_uid}")
            frame_records[frame_uid] = frame
    if frame_count != EXPECTED_FRAMES or len(frame_records) != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} canonical frames, found {frame_count}")
    region_ids: set[str] = set()
    with metadata_path.open("r", encoding="utf-8") as handle:
        for metadata_count, line in enumerate(handle, 1):
            item = json.loads(line)
            for field in ("video_id", "frame_idx"):
                if field not in item:
                    raise ValueError(f"DAM metadata row {metadata_count} is missing {field}")
            frame_uid = f"{item['video_id']}:{int(item['frame_idx'])}"
            frame_record = frame_records.get(frame_uid)
            if frame_record is None:
                raise ValueError(f"DAM row {metadata_count} maps to unknown frame {frame_uid}")
            exported_keyframe_n = item.get("keyframe_n")
            canonical_keyframe_n = int(frame_record["keyframe_n"])
            if exported_keyframe_n is not None and int(exported_keyframe_n) != canonical_keyframe_n:
                raise ValueError(
                    f"DAM row {metadata_count} keyframe_n disagrees with canonical metadata"
                )
            region_id = str(item.get("region_id") or "")
            if not region_id or region_id in region_ids:
                raise ValueError(f"DAM row {metadata_count} has a missing or duplicate region_id")
            region_ids.add(region_id)
            if len(item.get("bbox") or []) != 4:
                raise ValueError(f"DAM row {metadata_count} has an invalid bbox")
    if metadata_count != EXPECTED_DAM_REGIONS:
        raise ValueError(f"Expected {EXPECTED_DAM_REGIONS} DAM rows, found {metadata_count}")
    metadata_sha256 = sha256_file(metadata_path)
    metadata_stat = metadata_path.stat()
    frame_metadata_stat = frame_metadata_path.stat()
    frame_metadata_sha256 = sha256_file(frame_metadata_path)
    matrix_stat = matrix_path.stat()
    online_revision = os.environ.get("AIC_BGE_REVISION")
    if not online_revision:
        try:
            query_models = json.loads(
                Path(
                    os.environ.get("AIC_QUERY_MODEL_MANIFEST", "/models/query_models.json")
                ).read_text(encoding="utf-8")
            )
            online_revision = str(query_models["models"]["bge_m3"]["revision"])
        except (OSError, ValueError, TypeError, KeyError):
            online_revision = "local-cache"
    manifest: dict[str, object] = {
        "schema_version": "branch2.dam.v2",
        "passed": True,
        "status": "ready",
        "model_id": "BAAI/bge-m3",
        "online_revision": online_revision,
        "offline_revision": None,
        "revision_verified": False,
        "offline_identity": {
            "model_id": "BAAI/bge-m3",
            "immutable_revision": None,
            "checkpoint_sha256": None,
            "evidence": "legacy DAM export metadata does not record its immutable BGE-M3 revision",
            "revision_verified": False,
            "unverified_reason": "offline DAM embeddings cannot be tied cryptographically to a BGE-M3 snapshot",
        },
        "pooling": "cls",
        "normalization": "l2",
        "dimension": 1024,
        "dtype": "float16",
        "vector_count": EXPECTED_DAM_REGIONS,
        "metadata_count": metadata_count,
        "metadata_sha256": metadata_sha256,
        "metadata_size": metadata_stat.st_size,
        "metadata_mtime_ns": metadata_stat.st_mtime_ns,
        "matrix_sha256": sha256_file(matrix_path),
        "matrix_size": matrix_stat.st_size,
        "matrix_mtime_ns": matrix_stat.st_mtime_ns,
        "frame_metadata_sha256": frame_metadata_sha256,
        "frame_metadata_size": frame_metadata_stat.st_size,
        "frame_metadata_mtime_ns": frame_metadata_stat.st_mtime_ns,
        "frame_metadata_count": frame_count,
        "frame_metadata_identity_verified": True,
        "finite_verified": True,
        "l2_normalized": True,
        "min_norm": min_norm,
        "max_norm": max_norm,
        "frame_mapping_verified": True,
        "region_identity_verified": True,
    }
    state_root.mkdir(parents=True, exist_ok=True)
    manifest_path = state_root / "branch2_dam_manifest.json"
    staging = manifest_path.with_suffix(".staging.json")
    staging.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(staging, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("AIC_DATA_ROOT", "/data"))
    )
    parser.add_argument(
        "--state-root", type=Path, default=Path(os.environ.get("AIC_STATE_ROOT", "/state"))
    )
    parser.add_argument("--skip-bm25", action="store_true")
    args = parser.parse_args()
    args.state_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.state_root / "branch2_dam_manifest.json"
    staging = manifest_path.with_suffix(".staging.json")
    staging.write_text(
        json.dumps(
            {
                "schema_version": "branch2.dam.v2",
                "passed": False,
                "status": "validating",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(staging, manifest_path)
    try:
        manifest = validate_and_manifest(args.data_root, args.state_root)
        if not args.skip_bm25:
            DamBm25Index.prepare(
                args.data_root,
                args.state_root,
                str(manifest["metadata_sha256"]),
                str(manifest["frame_metadata_sha256"]),
            )
    except Exception as error:
        failed = {
            "schema_version": "branch2.dam.v2",
            "passed": False,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        staging.write_text(json.dumps(failed, indent=2), encoding="utf-8")
        os.replace(staging, manifest_path)
        raise
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
