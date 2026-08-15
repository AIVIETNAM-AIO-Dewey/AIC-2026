#!/usr/bin/env python3
"""Run PhoWhisper ASR pipeline to extract audio speech transcripts from video files."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _common import add_common_arguments, read_config, resolve_device  # noqa: E402

from aic2026.asr.backend import create_asr_backend  # noqa: E402
from aic2026.asr.pipeline import process_video  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_phowhisper_asr")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Directory containing .mp4 video files (or parent of video folders).",
    )
    parser.add_argument(
        "--map-csv-dir",
        type=Path,
        help="Directory containing map-keyframes/*.csv files.",
    )
    parser.add_argument(
        "--engine",
        choices=["faster_whisper", "huggingface"],
        help="ASR backend engine (default from config or faster_whisper).",
    )
    parser.add_argument(
        "--model-id",
        help="Model ID or CT2 path (default from config or vinai/PhoWhisper-large).",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="CTranslate2 compute type: float16, int8, float32.",
    )
    parser.add_argument(
        "--rclone-dest",
        help="Remote rclone destination (e.g. gdrive:AIC_HCM/artifacts/asr_segments/).",
    )
    return parser


def rclone_sync_file(local_path: Path, rclone_dest: str) -> bool:
    """Sync a single completed file to rclone remote destination."""
    cmd = ["rclone", "copy", str(local_path), rclone_dest]
    try:
        res = subprocess.run(cmd, capture_output=True, check=True)
        logger.info("rclone synced %s -> %s", local_path.name, rclone_dest)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("rclone sync failed for %s: %s", local_path.name, exc)
        return False


def find_video_files(video_dir: Path, explicit_video_id: str | None = None) -> list[tuple[str, Path]]:
    """Discover video files (video_id, video_path) under video_dir."""
    videos: list[tuple[str, Path]] = []

    # If video_dir (e.g. /kaggle/input/aic2026-data/videos) does not exist, fallback to parent or /kaggle/input
    if not video_dir.exists():
        if video_dir.parent.exists():
            logger.info("Video directory %s not found, falling back to parent %s", video_dir, video_dir.parent)
            video_dir = video_dir.parent
        else:
            raise FileNotFoundError(f"Video directory not found: {video_dir}")

    # Case 1: video_dir contains .mp4 files directly or nested
    for p in sorted(video_dir.glob("**/*.mp4")):
        v_id = p.stem
        if explicit_video_id and v_id != explicit_video_id:
            continue
        videos.append((v_id, p))

    return videos


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = args.config or (REPO_ROOT / "configs" / "offline" / "asr.yaml")
    config = read_config(config_path if config_path.exists() else None)

    device = resolve_device(args.device, config)
    engine = args.engine or config.get("engine", "faster_whisper")
    model_id = args.model_id or (
        config.get("ct2_model_id") if engine == "faster_whisper" else config.get("model_id")
    ) or "vinai/PhoWhisper-large"
    compute_type = args.compute_type or config.get("compute_type", "float16")

    output_root = args.output_root or Path(os.environ.get("AIC_ARTIFACT_ROOT", REPO_ROOT / "artifacts"))
    output_dir = output_root / "asr_segments"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = args.data_root or Path(os.environ.get("AIC_DATA_ROOT", REPO_ROOT))
    video_dir = args.video_dir or (data_root / "videos")
    map_csv_dir = args.map_csv_dir or (data_root / "map-keyframes")

    rclone_dest = args.rclone_dest or config.get("rclone_dest")

    logger.info("Initializing ASR backend engine=%s model=%s device=%s", engine, model_id, device)
    backend = create_asr_backend(
        engine=engine,
        model_id=model_id,
        device=device,
        compute_type=compute_type,
    )

    # Find videos
    video_list = find_video_files(video_dir, explicit_video_id=args.video_id)
    if not video_list and args.video_id:
        # Check if video_path can be inferred
        possible_mp4 = video_dir / f"{args.video_id}.mp4"
        if possible_mp4.exists():
            video_list = [(args.video_id, possible_mp4)]

    if not video_list:
        logger.error("No video files found in %s matching criteria.", video_dir)
        return 1

    if args.limit:
        video_list = video_list[: args.limit]

    logger.info("Found %d videos to process.", len(video_list))

    window_size_s = float(config.get("window_size_s", 15.0))
    stride_s = float(config.get("stride_s", 7.5))
    initial_prompt = config.get(
        "initial_prompt",
        "Bản tin thời sự, tin tức Việt Nam, YouTube, Facebook, iPhone, AI, video, online",
    )

    completed_count = 0
    skipped_count = 0

    for video_id, video_path in video_list:
        csv_path = map_csv_dir / f"{video_id}.csv"
        if not csv_path.exists():
            logger.warning("Missing map-keyframes CSV for %s at %s. Skipping.", video_id, csv_path)
            continue

        manifest_path = output_dir / f"{video_id}.manifest.json"
        jsonl_path = output_dir / f"{video_id}.jsonl"

        if args.resume and manifest_path.exists() and jsonl_path.exists():
            logger.info("Skipping already completed video: %s (--resume)", video_id)
            skipped_count += 1
            continue

        logger.info("Processing video: %s (%s)", video_id, video_path.name)
        manifest = process_video(
            video_id=video_id,
            video_path=video_path,
            keyframe_csv_path=csv_path,
            output_dir=output_dir,
            backend=backend,
            window_size_s=window_size_s,
            stride_s=stride_s,
            initial_prompt=initial_prompt,
            vad_filter=bool(config.get("vad_filter", True)),
            vad_min_silence_duration_ms=int(config.get("vad_min_silence_duration_ms", 500)),
            dedup_time_overlap_threshold=float(config.get("dedup_time_overlap_threshold", 0.80)),
            dedup_text_similarity_threshold=float(config.get("dedup_text_similarity_threshold", 0.85)),
            merge_gap_ms=int(config.get("merge_gap_ms", 500)),
        )

        completed_count += 1

        if rclone_dest and manifest.status in ("completed", "skipped"):
            rclone_sync_file(jsonl_path, rclone_dest)
            rclone_sync_file(manifest_path, rclone_dest)

    logger.info(
        "ASR processing finished: %d completed, %d skipped out of %d total.",
        completed_count,
        skipped_count,
        len(video_list),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
