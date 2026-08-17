#!/usr/bin/env python3
"""Multi-worker batch runner for DAM object description with atomic rclone sync."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("dam_batch_runner")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/offline/object_description.yaml")
    parser.add_argument("--keyframes-root", type=Path, required=True, help="Root directory containing keyframe images (aic-26-video)")
    parser.add_argument("--objects-root", type=Path, required=True, help="Root directory containing object bounding boxes")
    parser.add_argument("--map-keyframes-root", type=Path, required=True, help="Root directory containing map-keyframes CSVs")
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/aic2026-artifacts"))
    parser.add_argument("--cache-root", type=Path, default=Path("/kaggle/working/aic2026-model-cache"))
    parser.add_argument("--device", default="cuda", help="Execution device (cuda or auto)")
    parser.add_argument("--worker-id", type=int, default=0, help="0-indexed worker ID (0 to num_workers-1)")
    parser.add_argument("--num-workers", type=int, default=8, help="Total number of workers")
    parser.add_argument("--rclone-dest", help="Remote rclone destination (e.g. gdrive:AIC_HCM/artifacts/dam_descriptions/)")
    parser.add_argument("--master-video-list", type=Path, default=REPO_ROOT / "configs/master_video_list.txt")
    parser.add_argument("--limit", type=int, help="Optional frame limit per video for smoke testing")
    parser.add_argument("--dry-run", action="store_true", help="Validate assignments and input paths without running models")
    parser.add_argument("--no-resume", action="store_true", help="Force re-running already completed videos")
    return parser


def load_master_video_list(list_path: Path, objects_root: Path) -> list[str]:
    """Load sorted master list of all 873 video IDs."""
    if list_path.exists():
        with list_path.open("r", encoding="utf-8") as fp:
            vids = [line.strip() for line in fp if line.strip() and not line.startswith("#")]
        if vids:
            return sorted(set(vids))
    
    # Fallback to scanning objects_root
    logger.info("Master list file not found at %s, discovering from %s", list_path, objects_root)
    discovered = []
    for entry in objects_root.iterdir():
        if entry.is_dir() and entry.name.startswith("L"):
            discovered.append(entry.name)
        elif entry.is_file() and entry.suffix == ".json" and entry.stem.startswith("L"):
            discovered.append(entry.stem)
    if not discovered:
        raise FileNotFoundError(f"Could not load or discover master video list from {objects_root}")
    return sorted(set(discovered))


class PathResolver:
    """Fast, safe path resolution engine for custom dataset folder hierarchies."""
    def __init__(self, keyframes_root: Path, objects_root: Path, map_keyframes_root: Path) -> None:
        self.keyframes_root = keyframes_root.expanduser().resolve()
        self.objects_root = objects_root.expanduser().resolve()
        self.map_keyframes_root = map_keyframes_root.expanduser().resolve()
        self._keyframe_index: dict[str, Path] | None = None
        self._objects_index: dict[str, Path] | None = None
        self._map_csv_index: dict[str, Path] | None = None

    def _build_keyframe_index(self) -> dict[str, Path]:
        if self._keyframe_index is not None:
            return self._keyframe_index
        logger.info("Fast-indexing keyframe directories under %s ...", self.keyframes_root)
        index: dict[str, Path] = {}
        for root, dirs, _ in os.walk(self.keyframes_root):
            for d in list(dirs):
                if d.startswith("L") and "_" in d:
                    index[d] = Path(root) / d
                    dirs.remove(d)  # Do NOT recurse into video directory (skip 177K jpgs)
        logger.info("Indexed %d keyframe video directories.", len(index))
        self._keyframe_index = index
        return index

    def _build_objects_index(self) -> dict[str, Path]:
        if self._objects_index is not None:
            return self._objects_index
        logger.info("Fast-indexing object directories under %s ...", self.objects_root)
        index: dict[str, Path] = {}
        for root, dirs, _ in os.walk(self.objects_root):
            for d in list(dirs):
                if d.startswith("L") and "_" in d:
                    index[d] = Path(root) / d
                    dirs.remove(d)  # Do NOT recurse into object directory (skip jsons)
        logger.info("Indexed %d object directories.", len(index))
        self._objects_index = index
        return index

    def _build_map_csv_index(self) -> dict[str, Path]:
        if self._map_csv_index is not None:
            return self._map_csv_index
        logger.info("Fast-indexing map-keyframes CSVs under %s ...", self.map_keyframes_root)
        index: dict[str, Path] = {}
        for root_to_scan in (self.map_keyframes_root, REPO_ROOT / "data" / "map-keyframes"):
            if root_to_scan.exists():
                for root, _, files in os.walk(root_to_scan):
                    for f in files:
                        if f.endswith(".csv") and f.startswith("L"):
                            p = Path(root) / f
                            index[p.stem] = p
        logger.info("Indexed %d map CSV files.", len(index))
        self._map_csv_index = index
        return index

    def resolve_frames_dir(self, video_id: str) -> Path:
        batch = video_id.split("_")[0]  # e.g. L21
        candidates = [
            self.keyframes_root / f"Keyframes_{batch}" / "keyframes" / video_id,
            self.keyframes_root / f"Keyframes_{batch}" / video_id,
            self.keyframes_root / "Keyframes" / f"Keyframes_{batch}" / "keyframes" / video_id,
            self.keyframes_root / "Keyframes" / "Keyframes" / f"Keyframes_{batch}" / "keyframes" / video_id,
            self.keyframes_root / "keyframes" / video_id,
            self.keyframes_root / video_id,
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate

        # Search in-memory index
        index = self._build_keyframe_index()
        if video_id in index and index[video_id].is_dir():
            return index[video_id]

        raise FileNotFoundError(f"Keyframe directory for {video_id} not found under {self.keyframes_root}")

    def resolve_objects_dir(self, video_id: str) -> Path:
        candidates = [
            self.objects_root / video_id,
            self.objects_root / "objects" / video_id,
            self.objects_root / "data" / "objects" / video_id,
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate

        index = self._build_objects_index()
        if video_id in index and index[video_id].is_dir():
            return index[video_id]

        raise FileNotFoundError(f"Objects directory for {video_id} not found under {self.objects_root}")

    def resolve_map_csv(self, video_id: str) -> Path:
        candidates = [
            self.map_keyframes_root / f"{video_id}.csv",
            self.map_keyframes_root / "map-keyframes" / f"{video_id}.csv",
            self.map_keyframes_root / "data" / "map-keyframes" / f"{video_id}.csv",
            REPO_ROOT / "data" / "map-keyframes" / f"{video_id}.csv",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        index = self._build_map_csv_index()
        if video_id in index and index[video_id].is_file():
            return index[video_id]

        raise FileNotFoundError(f"Map CSV for {video_id} not found under {self.map_keyframes_root}")


def rclone_sync_file(local_path: Path, rclone_dest: str, max_retries: int = 3) -> bool:
    """Sync a single file to rclone remote with exponential backoff retries."""
    cmd = ["rclone", "copy", str(local_path), rclone_dest]
    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(cmd, capture_output=True, check=True, text=True)
            logger.info("  [RCLONE] Synced %s -> %s", local_path.name, rclone_dest)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("  [RCLONE] Attempt %d/%d failed for %s: %s", attempt, max_retries, local_path.name, exc)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return False


def is_video_completed(description_artifact: Path, description_manifest: Path) -> bool:
    """Check if a video description shard is already completely generated."""
    if not description_artifact.exists() or not description_manifest.exists():
        return False
    if description_artifact.stat().st_size == 0 or description_manifest.stat().st_size == 0:
        return False
    try:
        import json
        with description_manifest.open("r", encoding="utf-8") as fp:
            manifest_data = json.load(fp)
            return manifest_data.get("status") == "completed"
    except Exception:
        return False


def run_pipeline_step(cmd: list[str], step_name: str) -> None:
    """Execute a single pipeline CLI step synchronously with immediate log flushing."""
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        logger.error("Pipeline failure in %s (exit code %d):\n%s", step_name, res.returncode, res.stdout)
        raise RuntimeError(f"Step {step_name} failed with exit code {res.returncode}")
    # Print the last meaningful line of output
    lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    if lines:
        logger.info("    -> %s: %s", step_name, lines[-1])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError(f"worker_id must be within [0, {args.num_workers - 1}]")

    output_root = args.output_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)

    # 1. Load Master Video List and Partition
    all_videos = load_master_video_list(args.master_video_list, args.objects_root)
    assigned_videos = all_videos[args.worker_id::args.num_workers]

    print("=" * 80)
    print(f" 🛰️  KAGGLE DAM DISTRIBUTED WORKER INITIALIZATION")
    print("=" * 80)
    print(f" Worker Index:         {args.worker_id} / {args.num_workers}")
    print(f" Total Corpus Videos:  {len(all_videos)}")
    print(f" Assigned Workload:    {len(assigned_videos)} videos")
    print(f" Keyframes Source:     {args.keyframes_root}")
    print(f" Objects Source:       {args.objects_root}")
    print(f" Map-Keyframes Source: {args.map_keyframes_root}")
    print(f" Output Root:          {output_root}")
    print(f" Model Cache:          {cache_root}")
    if args.rclone_dest:
        print(f" Rclone Destination:   {args.rclone_dest}")
    if args.limit:
        print(f" Frame Limit:          {args.limit} (Smoke Mode)")
    print("-" * 80)
    print(f" Assigned Videos ({len(assigned_videos)}):")
    print(f"   {assigned_videos[:10]} ... {assigned_videos[-5:] if len(assigned_videos) > 10 else ''}")
    print("=" * 80)
    sys.stdout.flush()

    # 2. Path Resolver
    resolver = PathResolver(
        keyframes_root=args.keyframes_root,
        objects_root=args.objects_root,
        map_keyframes_root=args.map_keyframes_root,
    )

    if args.dry_run:
        logger.info("🔍 DRY-RUN: Verifying paths for all %d assigned videos...", len(assigned_videos))
        missing_count = 0
        from aic2026.object_description import load_organizer_detections, filter_detections, FilterConfig
        import yaml
        cfg_dict: dict[str, Any] = {}
        if args.config and args.config.exists():
            try:
                with open(args.config, "r", encoding="utf-8") as cf:
                    cfg_dict = yaml.safe_load(cf) or {}
            except Exception:
                pass
        sample_config = FilterConfig(
            minimum_score=float(cfg_dict.get("score_threshold", 0.30)),
            minimum_area_ratio=float(cfg_dict.get("min_area_ratio", 0.005)),
            maximum_area_ratio=float(cfg_dict.get("max_area_ratio", 0.85)),
            same_class_iou=float(cfg_dict.get("class_nms_iou", 0.45)),
            cross_label_duplicate_iou=float(cfg_dict.get("cross_label_iou", 0.60)),
            maximum_regions=int(cfg_dict.get("max_regions", 3)),
        )
        for idx, vid in enumerate(assigned_videos, start=1):
            try:
                f_dir = resolver.resolve_frames_dir(vid)
                o_dir = resolver.resolve_objects_dir(vid)
                m_csv = resolver.resolve_map_csv(vid)
                if idx == 1:
                    sample_json = next(o_dir.glob("*.json"), None)
                    if sample_json and sample_json.is_file():
                        raw_dets = load_organizer_detections(sample_json)
                        filtered = filter_detections(raw_dets, sample_config)
                        labels = [d.class_entity for d in filtered]
                        logger.info("  🎯 Sample Box Filter Test (%s/%s): %d raw boxes -> %d distinct objects: %s", vid, sample_json.name, len(raw_dets), len(filtered), labels)
                if idx <= 5 or idx == len(assigned_videos):
                    logger.info("  [%d/%d] %s -> Frames: %s | Objects: %s | Map: %s", idx, len(assigned_videos), vid, f_dir.name, o_dir.name, m_csv.name)
            except FileNotFoundError as error:
                logger.error("  ❌ Missing path for %s: %s", vid, error)
                missing_count += 1
        if missing_count == 0:
            logger.info("✅ DRY-RUN SUCCESS: All %d assigned videos have valid verified paths!", len(assigned_videos))
            return 0
        else:
            logger.error("❌ DRY-RUN FAILED: %d videos have missing source paths.", missing_count)
            return 1

    # 3. Main Batch Execution Loop
    total_assigned = len(assigned_videos)
    completed_in_run = 0
    skipped_count = 0
    start_time_all = time.time()

    remote_completed_videos: set[str] = set()
    if args.rclone_dest and not args.no_resume:
        try:
            res = subprocess.run(
                ["rclone", "lsf", args.rclone_dest, "--tpslimit", "5"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if res.returncode == 0:
                for fname in res.stdout.splitlines():
                    fname = fname.strip()
                    if fname.endswith(".jsonl"):
                        remote_completed_videos.add(fname[:-6])
                if remote_completed_videos:
                    logger.info("📡 Pre-fetched %d completed videos from Google Drive destination!", len(remote_completed_videos))
        except Exception as exc:
            logger.warning("Could not pre-fetch remote Google Drive file list: %s", exc)

    for idx, video_id in enumerate(assigned_videos, start=1):
        video_start_time = time.time()
        pct = (idx / total_assigned) * 100
        logger.info("[Video %d/%d (%.1f%%)] Starting %s ...", idx, total_assigned, pct, video_id)

        # Artifact target paths
        frame_manifest = output_root / "frame_manifests" / f"{video_id}.jsonl"
        mask_artifact = output_root / "object_description" / "masks" / f"{video_id}.jsonl"
        description_artifact = output_root / "object_description" / "descriptions" / f"{video_id}.jsonl"
        description_manifest = description_artifact.with_suffix(".manifest.json")

        # Check if already completed (Resume from local SSD or Google Drive)
        if not args.no_resume and (video_id in remote_completed_videos or is_video_completed(description_artifact, description_manifest)):
            logger.info("  [SKIP - ALREADY COMPLETED IN GDRIVE/LOCAL] %s", video_id)
            skipped_count += 1
            if args.rclone_dest and is_video_completed(description_artifact, description_manifest):
                rclone_sync_file(description_artifact, args.rclone_dest)
                rclone_sync_file(description_manifest, args.rclone_dest)
            continue

        try:
            frames_dir = resolver.resolve_frames_dir(video_id)
            objects_dir = resolver.resolve_objects_dir(video_id)
            map_csv = resolver.resolve_map_csv(video_id)
        except FileNotFoundError as error:
            logger.error("  ❌ Skipping %s due to missing path: %s", video_id, error)
            continue

        # Stage 0: Build Frame Manifest
        limit_args = ["--limit", str(args.limit)] if args.limit else []
        resume_args = ["--resume"] if not args.no_resume else []

        run_pipeline_step(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/build_frame_manifest.py"),
                "--config", str(args.config),
                "--video-id", video_id,
                "--data-root", str(resolver.keyframes_root),
                "--output-root", str(output_root),
                "--cache-root", str(cache_root),
                "--map-csv", str(map_csv),
                "--frames-dir", str(frames_dir),
                "--output", str(frame_manifest),
                *resume_args,
                *limit_args,
            ],
            step_name="build_frame_manifest",
        )

        # Stage 1: SAM Masks
        run_pipeline_step(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/prepare_object_masks.py"),
                "--config", str(args.config),
                "--video-id", video_id,
                "--data-root", str(resolver.keyframes_root),
                "--output-root", str(output_root),
                "--cache-root", str(cache_root),
                "--frame-manifest", str(frame_manifest),
                "--objects-dir", str(objects_dir),
                "--output", str(mask_artifact),
                "--device", args.device,
                *resume_args,
                *limit_args,
            ],
            step_name="prepare_object_masks",
        )

        # Stage 2: DAM Descriptions
        run_pipeline_step(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/run_dam_descriptions.py"),
                "--config", str(args.config),
                "--video-id", video_id,
                "--data-root", str(resolver.keyframes_root),
                "--output-root", str(output_root),
                "--cache-root", str(cache_root),
                "--mask-artifact", str(mask_artifact),
                "--output", str(description_artifact),
                "--device", args.device,
                *resume_args,
                *limit_args,
            ],
            step_name="run_dam_descriptions",
        )

        # Sync to Google Drive
        if args.rclone_dest and is_video_completed(description_artifact, description_manifest):
            rclone_sync_file(description_artifact, args.rclone_dest)
            rclone_sync_file(description_manifest, args.rclone_dest)

        elapsed_vid = time.time() - video_start_time
        completed_in_run += 1
        elapsed_total = time.time() - start_time_all
        avg_time = elapsed_total / (completed_in_run + skipped_count)
        remaining_vids = total_assigned - idx
        eta_seconds = remaining_vids * avg_time
        eta_hours = eta_seconds / 3600.0

        logger.info(
            "  ✅ %s completed in %.1fs | Worker Progress: %d/%d (%.1f%%) | ETA: %.1fh",
            video_id,
            elapsed_vid,
            idx,
            total_assigned,
            pct,
            eta_hours,
        )
        sys.stdout.flush()

    logger.info("=" * 80)
    logger.info("🎉 WORKER %d BATCH COMPLETE! Processed: %d, Skipped: %d, Total: %d", args.worker_id, completed_in_run, skipped_count, total_assigned)
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
