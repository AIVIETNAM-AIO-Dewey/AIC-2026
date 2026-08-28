"""Build the UI index for the 247,956-frame no-fusion experiment.

The source dataset is never modified.  The builder writes into a new sibling
directory and only renames that completed directory into place after every
frame, vector row, and metadata join has been validated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = Path("/Users/macbookpro/Downloads/AIC-HCM-BATCH-1/AIC_HCM_BATCH_1")
EXPECTED_VIDEOS = 873
EXPECTED_FRAMES = 247_956
EXPECTED_DAM_REGIONS = 681_355
SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def _read_map(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"n", "pts_time", "fps", "frame_idx"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Missing required columns in {path}: {sorted(required)}")
    return rows


def _count_jsonl(path: Path) -> int:
    with path.open("rb") as file:
        return sum(1 for line in file if line.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _place_asset(source: Path, destination: Path, mode: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "hardlink":
        os.link(source, destination)
        return
    raise ValueError(f"Unknown asset mode: {mode}")


def _same_frame(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("video_id") == right.get("video_id")
        and int(left.get("keyframe_n", -1)) == int(right.get("keyframe_n", -2))
        and int(left.get("frame_idx", -1)) == int(right.get("frame_idx", -2))
    )


def build_nofusion_index(
    dataset_root: Path,
    output_dir: Path,
    *,
    asset_mode: str = "copy",
    compute_checksums: bool = True,
    expected_videos: int = EXPECTED_VIDEOS,
    expected_frames: int = EXPECTED_FRAMES,
    expected_dam_regions: int = EXPECTED_DAM_REGIONS,
) -> dict[str, Any]:
    """Assemble and validate a UI-compatible index without touching sources."""

    dataset_root = dataset_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    artifacts = dataset_root / "artifacts"
    map_dir = artifacts / "map-keyframes"
    unified_meta_dir = artifacts / "unified_metadata"
    scene_dir = artifacts / "scene_embeddings"
    asr_dir = artifacts / "asr_aligned"
    dense_dir = artifacts / "dense_text_embeddings"

    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_dir}. "
            "Move it aside or choose a different --output-dir."
        )

    map_files = sorted(map_dir.glob("*.csv"))
    if len(map_files) != expected_videos:
        raise ValueError(
            f"Expected {expected_videos} map files, found {len(map_files)} in {map_dir}"
        )
    map_video_ids = {path.stem for path in map_files}
    media_video_ids = {path.stem for path in (artifacts / "media-info").glob("*.json")}
    if media_video_ids != map_video_ids:
        missing = sorted(map_video_ids - media_video_ids)
        extra = sorted(media_video_ids - map_video_ids)
        raise ValueError(
            "Media-info video IDs do not match map-keyframes: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    LOGGER.info("Counting canonical keyframes across %d videos", len(map_files))
    map_rows_by_video: dict[str, list[dict[str, str]]] = {}
    total_frames = 0
    for map_path in map_files:
        rows = _read_map(map_path)
        map_rows_by_video[map_path.stem] = rows
        total_frames += len(rows)
    if total_frames != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, found {total_frames}")

    asr_vectors_source = asr_dir / "keyframes_speech_vectors.f16.npy"
    asr_metadata_source = asr_dir / "keyframes_asr_metadata.jsonl"
    speech_matrix = np.load(asr_vectors_source, mmap_mode="r")
    if speech_matrix.shape != (total_frames, 1024) or speech_matrix.dtype != np.float16:
        raise ValueError(
            f"Unexpected ASR matrix: shape={speech_matrix.shape}, dtype={speech_matrix.dtype}"
        )
    if _count_jsonl(asr_metadata_source) != total_frames:
        raise ValueError("ASR metadata row count does not match canonical frame count")
    del speech_matrix

    dam_vectors_source = dense_dir / "dam_vectors.f16.npy"
    dam_metadata_source = dense_dir / "dam_metadata.jsonl"
    dam_matrix = np.load(dam_vectors_source, mmap_mode="r")
    if dam_matrix.shape != (expected_dam_regions, 1024) or dam_matrix.dtype != np.float16:
        raise ValueError(
            f"Unexpected DAM matrix: shape={dam_matrix.shape}, dtype={dam_matrix.dtype}"
        )
    if _count_jsonl(dam_metadata_source) != expected_dam_regions:
        raise ValueError("DAM metadata row count does not match the DAM vector matrix")
    del dam_matrix

    build_dir = output_dir.with_name(f".{output_dir.name}.building-{uuid.uuid4().hex[:8]}")
    build_dir.mkdir(parents=True, exist_ok=False)
    LOGGER.info("Writing validated index into %s", build_dir)

    visual_output = build_dir / "keyframes_visual_vectors.f16.npy"
    keyframe_metadata_output = build_dir / "keyframes_metadata.jsonl"
    visual_matrix = np.lib.format.open_memmap(
        visual_output,
        mode="w+",
        dtype=np.float16,
        shape=(total_frames, 768),
    )

    asr_iterator = _iter_jsonl(asr_metadata_source)
    global_row = 0
    frames_with_speech = 0
    frames_with_ocr = 0

    with keyframe_metadata_output.open("w", encoding="utf-8") as metadata_file:
        for video_number, map_path in enumerate(map_files, 1):
            video_id = map_path.stem
            map_rows = map_rows_by_video[video_id]
            unified_path = unified_meta_dir / f"{video_id}.jsonl"
            scene_path = scene_dir / f"{video_id}.safetensors"
            if not unified_path.is_file() or not scene_path.is_file():
                raise FileNotFoundError(f"Missing metadata or scene vectors for {video_id}")

            frame_metadata = list(_iter_jsonl(unified_path))
            scene_tensors = load_file(scene_path)
            if set(scene_tensors) != {"embeddings"}:
                raise ValueError(
                    f"Unexpected SafeTensors keys for {video_id}: {scene_tensors.keys()}"
                )
            scene_vectors = scene_tensors["embeddings"]
            expected_shape = (len(map_rows), 768)
            if scene_vectors.shape != expected_shape or scene_vectors.dtype != np.float16:
                raise ValueError(
                    f"Unexpected scene vectors for {video_id}: "
                    f"shape={scene_vectors.shape}, dtype={scene_vectors.dtype}"
                )
            if len(frame_metadata) != len(map_rows):
                raise ValueError(
                    f"Metadata/map mismatch for {video_id}: "
                    f"{len(frame_metadata)} != {len(map_rows)}"
                )

            visual_matrix[global_row : global_row + len(map_rows)] = scene_vectors

            for local_row, (map_row, source_meta) in enumerate(
                zip(map_rows, frame_metadata, strict=True)
            ):
                canonical = {
                    "video_id": video_id,
                    "keyframe_n": int(map_row["n"]),
                    "frame_idx": int(map_row["frame_idx"]),
                }
                if not _same_frame(canonical, source_meta):
                    raise ValueError(f"Map/unified metadata mismatch at {video_id} row {local_row}")
                if int(source_meta.get("embedding_row", -1)) != local_row:
                    raise ValueError(f"Scene embedding row mismatch at {video_id} row {local_row}")

                try:
                    asr_meta = next(asr_iterator)
                except StopIteration as error:
                    raise ValueError("ASR metadata ended before canonical frames") from error
                if not _same_frame(canonical, asr_meta):
                    raise ValueError(
                        f"ASR/canonical mismatch at global row {global_row}: "
                        f"{canonical} != {asr_meta}"
                    )
                if int(asr_meta.get("speech_vector_row", -1)) != global_row:
                    raise ValueError(f"ASR vector row mismatch at global row {global_row}")

                merged = dict(source_meta)
                for key, value in asr_meta.items():
                    if key not in {
                        "point_id",
                        "video_id",
                        "keyframe_n",
                        "frame_idx",
                        "pts_time_s",
                        "fps",
                        "frame_uid",
                        "image_relpath",
                    }:
                        merged[key] = value
                merged.update(
                    {
                        "point_id": global_row + 1,
                        "video_id": video_id,
                        "keyframe_n": canonical["keyframe_n"],
                        "frame_idx": canonical["frame_idx"],
                        "pts_time_s": float(map_row["pts_time"]),
                        "fps": float(map_row["fps"]),
                        "frame_uid": f"{video_id}:{canonical['frame_idx']}",
                        "visual_vector_row": global_row,
                        "speech_vector_row": global_row,
                        "embedding_row": global_row,
                        "embedding_matrix_relpath": visual_output.name,
                    }
                )
                image_path = artifacts / str(merged.get("image_relpath", ""))
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Missing keyframe image at global row {global_row}: {image_path}"
                    )
                metadata_file.write(json.dumps(merged, ensure_ascii=False, separators=(",", ":")))
                metadata_file.write("\n")

                frames_with_speech += int(bool(merged.get("has_speech")))
                frames_with_ocr += int(bool(str(merged.get("ocr_text", "")).strip()))
                global_row += 1

            if video_number % 50 == 0 or video_number == len(map_files):
                LOGGER.info(
                    "Validated %d/%d videos (%d/%d frames)",
                    video_number,
                    len(map_files),
                    global_row,
                    total_frames,
                )

    try:
        extra_asr_row = next(asr_iterator)
    except StopIteration:
        extra_asr_row = None
    if extra_asr_row is not None or global_row != total_frames:
        raise ValueError("Canonical metadata and ASR metadata did not end together")

    visual_matrix.flush()
    del visual_matrix

    placed_assets = {
        "keyframes_speech_vectors.f16.npy": asr_vectors_source,
        "dam_vectors.f16.npy": dam_vectors_source,
        "dam_metadata.jsonl": dam_metadata_source,
    }
    for filename, source in placed_assets.items():
        LOGGER.info("Placing %s via %s", filename, asset_mode)
        _place_asset(source, build_dir / filename, asset_mode)

    summary = {
        "schema_version": "aic26.nofusion_ui_index.v1",
        "dataset_root": str(dataset_root),
        "videos": len(map_files),
        "media_info_files": len(media_video_ids),
        "total_keyframes": total_frames,
        "frames_with_speech": frames_with_speech,
        "silent_frames": total_frames - frames_with_speech,
        "frames_with_ocr": frames_with_ocr,
        "frames_without_ocr": total_frames - frames_with_ocr,
        "visual_vectors_shape": [total_frames, 768],
        "speech_vectors_shape": [total_frames, 1024],
        "dam_vectors_shape": [expected_dam_regions, 1024],
        "matrix_dtype": "float16",
        "asset_mode": asset_mode,
        "siglip_model_id": SIGLIP_MODEL_ID,
        "siglip_revision": SIGLIP_REVISION,
    }
    (build_dir / "unified_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_files: dict[str, dict[str, Any]] = {}
    for path in sorted(build_dir.iterdir()):
        if path.name == "manifest.json":
            continue
        record: dict[str, Any] = {"size_bytes": path.stat().st_size}
        if compute_checksums:
            LOGGER.info("Hashing %s", path.name)
            record["sha256"] = _sha256(path)
        manifest_files[path.name] = record
    manifest = {
        "schema_version": "aic26.nofusion_ui_index.manifest.v1",
        "source_dataset": str(dataset_root),
        "source_data_modified": False,
        "files": manifest_files,
    }
    (build_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    build_dir.rename(output_dir)
    LOGGER.info("Completed no-fusion UI index at %s", output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--asset-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to place already-built ASR and DAM artifacts in the output",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip SHA-256 generation for a faster local build",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.dataset_root / "unified_index"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_nofusion_index(
        args.dataset_root,
        output_dir,
        asset_mode=args.asset_mode,
        compute_checksums=not args.skip_checksums,
    )


if __name__ == "__main__":
    main()
