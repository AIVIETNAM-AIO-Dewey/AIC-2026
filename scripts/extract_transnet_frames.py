#!/usr/bin/env python3
"""Stage 0: TransNetV2 Shot Boundary Detection & Adaptive Keyframe Extraction.

Replaces organizer keyframes by running PyTorch TransNetV2 on raw video,
adaptively sampling keyframes based on shot duration, extracting high-quality JPEGs
in a single FFmpeg pass, and emitting both canonical frame manifests (FrameRef)
and local organizer-compatible map-keyframes CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aic2026.common.io import iter_jsonl, write_jsonl_atomic  # noqa: E402
from aic2026.common.manifest import (  # noqa: E402
    complete_manifest,
    create_manifest,
    fail_manifest,
    prepare_resume,
    write_manifest,
)
from aic2026.contracts import FrameRef  # noqa: E402
from aic2026.frame_extraction.ffmpeg import extract_frames_by_index, probe_video  # noqa: E402
from aic2026.frame_extraction.sampling import (  # noqa: E402
    FrameSampleCandidate,
    adaptive_samples_from_shots,
    dedupe_samples,
    fallback_samples,
)
from aic2026.frame_extraction.transnetv2 import run_transnetv2_inference  # noqa: E402
from _common import read_config, resolve_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True, help="Video identifier (e.g. L21_V001)")
    parser.add_argument("--video-path", type=Path, required=True, help="Path to input video MP4")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root directory")
    parser.add_argument("--config", type=Path, help="Path to YAML configuration")
    parser.add_argument("--device", default="auto", help="Inference device (auto, cuda, cpu)")
    parser.add_argument("--limit", type=int, help="Limit number of keyframes (smoke testing)")
    parser.add_argument("--jpeg-quality", type=int, default=2, help="FFmpeg JPEG quality (1=best, 31=worst)")
    parser.add_argument("--no-resume", action="store_true", help="Force overwrite existing manifest")
    return parser


def _valid_image_size(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    try:
        with Image.open(path) as img:
            size = img.size
            img.verify()
        return size
    except (OSError, UnidentifiedImageError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_id = args.video_id
    video_path = args.video_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    config = read_config(args.config)
    seed = int(config.get("seed", 2026))
    device = resolve_device(args.device, config)
    run_id = str(config.get("run", {}).get("run_id", "frame-extraction-v1"))

    # Adaptive sampling policy
    sampling_cfg = config.get("adaptive_sampling", {})
    single_frame_before_s = float(sampling_cfg.get("single_frame_before_s", 2.0))
    cadence_from_s = float(sampling_cfg.get("cadence_from_s", 4.0))
    extreme_shot_from_s = float(sampling_cfg.get("extreme_shot_from_s", 7.0))
    cadence_s = float(sampling_cfg.get("cadence_s", 1.5))
    max_frames_per_shot = int(sampling_cfg.get("max_frames_per_shot", 10))
    dedupe_tolerance_s = float(sampling_cfg.get("dedupe_tolerance_s", 0.5))

    # Manifest and target paths
    manifest_path = output_root / "frame_manifests" / f"{video_id}.jsonl"
    manifest_meta_path = manifest_path.with_suffix(".manifest.json")
    frames_dir = output_root / "keyframes" / video_id
    map_csv_path = output_root / "map-keyframes" / f"{video_id}.csv"

    resolved_config = {
        "schema_version": config.get("schema_version", "1.0"),
        "video_id": video_id,
        "seed": seed,
        "device": device,
        "limit": args.limit,
        "jpeg_quality": args.jpeg_quality,
        "adaptive_sampling": {
            "single_frame_before_s": single_frame_before_s,
            "cadence_from_s": cadence_from_s,
            "extreme_shot_from_s": extreme_shot_from_s,
            "cadence_s": cadence_s,
            "max_frames_per_shot": max_frames_per_shot,
            "dedupe_tolerance_s": dedupe_tolerance_s,
        },
    }

    manifest = create_manifest(
        run_id=run_id,
        stage="frame_manifest",
        config=resolved_config,
        seed=seed,
        input_paths=[("video", video_path)],
        repo_root=REPO_ROOT,
    )
    manifest, complete = prepare_resume(
        manifest_path=manifest_meta_path,
        output_path=manifest_path,
        proposed=manifest,
        resume=not args.no_resume,
    )
    if complete and map_csv_path.is_file():
        records = [FrameRef.model_validate(val) for val in iter_jsonl(manifest_path)]
        print(
            json.dumps(
                {
                    "status": "already_complete",
                    "video_id": video_id,
                    "frames": len(records),
                    "manifest": str(manifest_path),
                    "map_csv": str(map_csv_path),
                }
            )
        )
        return 0
    write_manifest(manifest_meta_path, manifest)

    # 1. Probe video metadata
    probe = probe_video(video_path)
    if probe.fps <= 0 or not math.isfinite(probe.fps):
        raise ValueError(f"Invalid video FPS: {probe.fps}")

    # 2. Run PyTorch TransNetV2 Shot Boundary Inference
    print(f"[transnetv2] Probing and detecting shots for {video_id} ({video_path.name}) ...", file=sys.stderr, flush=True)
    shot_result = run_transnetv2_inference(
        video_path=video_path,
        video_id=video_id,
        fps=probe.fps,
        device=device,
    )
    print(f"[transnetv2] Detected {len(shot_result.scenes)} shots in {video_id}", file=sys.stderr, flush=True)

    # 3. Adaptive keyframe candidate generation
    candidates = adaptive_samples_from_shots(
        shot_result.shots,
        single_frame_before_s=single_frame_before_s,
        cadence_from_s=cadence_from_s,
        extreme_shot_from_s=extreme_shot_from_s,
        cadence_s=cadence_s,
        max_frames_per_shot=max_frames_per_shot,
    )
    candidates = dedupe_samples(candidates, tolerance_s=dedupe_tolerance_s)

    if not candidates:
        print(f"[transnetv2] Warning: No candidates generated, falling back to interval samples", file=sys.stderr)
        candidates = fallback_samples(fps=probe.fps, limit=args.limit or 5)
    elif args.limit is not None and args.limit > 0:
        candidates = candidates[: args.limit]

    # Re-index sample_n sequentially from 1..N
    reindexed_candidates: list[FrameSampleCandidate] = [
        FrameSampleCandidate(
            sample_n=idx,
            pts_time_s=c.pts_time_s,
            fps=probe.fps,
            frame_idx=c.frame_idx,
            sampling_source="transnetv2",
            keyframe_n=idx,
            shot_id=c.shot_id,
            shot_start_idx=c.shot_start_idx,
            shot_end_idx=c.shot_end_idx,
        )
        for idx, c in enumerate(candidates, start=1)
    ]

    # 4. Extract missing frames in a single FFmpeg pass
    frames_dir.mkdir(parents=True, exist_ok=True)
    missing_extract: list[tuple[int, Path]] = []
    reused = 0
    for cand in reindexed_candidates:
        frame_file = frames_dir / f"{cand.frame_idx:08d}.jpg"
        if _valid_image_size(frame_file) is None:
            missing_extract.append((cand.frame_idx, frame_file))
        else:
            reused += 1

    if missing_extract:
        print(f"[transnetv2] Extracting {len(missing_extract)} frames (reused {reused}) via FFmpeg ...", file=sys.stderr, flush=True)
        extract_frames_by_index(
            video_path=video_path,
            outputs=missing_extract,
            jpeg_quality=args.jpeg_quality,
        )

    # 5. Write FrameRef manifest JSONL
    records: list[dict] = []
    for cand in reindexed_candidates:
        frame_file = frames_dir / f"{cand.frame_idx:08d}.jpg"
        img_size = _valid_image_size(frame_file)
        if img_size is None:
            raise RuntimeError(f"Extracted keyframe is missing or invalid: {frame_file}")
        w, h = img_size

        frame_ref = FrameRef(
            video_id=video_id,
            frame_uid=f"{video_id}:{cand.frame_idx}",
            keyframe_n=cand.sample_n,
            frame_idx=cand.frame_idx,
            pts_time_s=cand.pts_time_s,
            fps=probe.fps,
            frame_relpath=frame_file.relative_to(output_root).as_posix(),
            width=w,
            height=h,
        )
        records.append(frame_ref.model_dump())

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(manifest_path, records)

    # 6. Generate local organizer-compatible map-keyframes CSV
    map_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with map_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["n", "pts_time", "fps", "frame_idx"])
        for cand in reindexed_candidates:
            writer.writerow([cand.sample_n, f"{cand.pts_time_s:.4f}", f"{probe.fps:.2f}", cand.frame_idx])

    # 7. Write completion manifest metadata
    counters = {"frames": len(records), "reused": reused}
    manifest = complete_manifest(
        manifest,
        counters=counters,
        shard=video_id,
        output_paths=[("frame_manifest", manifest_path), ("map_csv", map_csv_path)],
    )
    write_manifest(manifest_meta_path, manifest)

    print(
        json.dumps(
            {
                "status": "completed",
                "video_id": video_id,
                "shots": len(shot_result.scenes),
                "frames": len(records),
                "reused": reused,
                "manifest": str(manifest_path),
                "map_csv": str(map_csv_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
