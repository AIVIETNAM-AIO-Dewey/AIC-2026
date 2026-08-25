#!/usr/bin/env python3
"""Multi-worker batch runner for DAM object description with atomic rclone sync."""

from __future__ import annotations

import argparse
import logging
import os
import random
import subprocess
import sys
import time
import zipfile
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


def create_keyframe_zip(keyframes_dir: Path, zip_output_path: Path) -> Path | None:
    """Create a compressed .zip archive of all keyframes for a video using ZIP_STORED (ultra fast)."""
    if not keyframes_dir.is_dir():
        return None
    jpg_files = sorted(keyframes_dir.glob("*.jpg"))
    if not jpg_files:
        return None

    zip_output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_zip = zip_output_path.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            for jpg in jpg_files:
                zf.write(jpg, arcname=f"{keyframes_dir.name}/{jpg.name}")
        temp_zip.replace(zip_output_path)
        return zip_output_path
    except Exception as exc:
        logger.warning("  ⚠️ Failed to create keyframe zip archive for %s: %s", keyframes_dir.name, exc)
        if temp_zip.exists():
            temp_zip.unlink()
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/offline/object_description.yaml")
    parser.add_argument("--videos-root", type=Path, default=None, help="Root directory containing raw video MP4s (e.g. Videos/ or Videos_L21/video/)")
    parser.add_argument("--keyframes-root", type=Path, default=None, help="Root directory containing keyframe images (optional with TransNetV2)")
    parser.add_argument("--objects-root", type=Path, default=None, help="Root directory containing object bounding boxes (optional with YOLO-World)")
    parser.add_argument("--map-keyframes-root", type=Path, default=None, help="Root directory containing map-keyframes CSVs (optional with TransNetV2)")
    parser.add_argument("--frame-extractor", choices=["transnetv2", "organizer"], default="transnetv2", help="Keyframe extraction method")
    parser.add_argument("--detector", choices=["yolo-world", "organizer"], default="yolo-world", help="Object detector backend")
    parser.add_argument("--detector-model", type=str, default="yolov8x-worldv2.pt", help="YOLO-World checkpoint or path")
    parser.add_argument("--enable-ocr", action="store_true", default=True, help="Enable Stage 3 OCR extraction (default: True)")
    parser.add_argument("--no-ocr", action="store_false", dest="enable_ocr", help="Disable Stage 3 OCR extraction")
    parser.add_argument("--ocr-backend", default="auto", choices=["auto", "easyocr", "paddleocr"], help="OCR engine backend")
    parser.add_argument("--ocr-threshold", type=float, default=0.30, help="Confidence threshold for OCR")
    parser.add_argument("--enable-siglip", action="store_true", default=True, help="Enable Stage 4 SigLIP2 embedding (default: True)")
    parser.add_argument("--no-siglip", action="store_false", dest="enable_siglip", help="Disable Stage 4 SigLIP2 embedding")
    parser.add_argument("--siglip-model", default="google/siglip2-base-patch16-224", help="SigLIP2 model ID or path")
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


def load_master_video_list(list_path: Path, objects_root: Path | None = None, keyframes_root: Path | None = None) -> list[str]:
    """Load sorted master list of all 873 video IDs."""
    if list_path.exists():
        with list_path.open("r", encoding="utf-8") as fp:
            vids = [line.strip() for line in fp if line.strip() and not line.startswith("#")]
        if vids:
            return sorted(set(vids))
    
    # Fallback to scanning objects_root or keyframes_root
    scan_root = objects_root if (objects_root and objects_root.exists()) else keyframes_root
    if scan_root and scan_root.exists():
        logger.info("Master list file not found at %s, discovering from %s", list_path, scan_root)
        discovered = []
        for entry in scan_root.iterdir():
            if entry.is_dir() and entry.name.startswith("L"):
                discovered.append(entry.name)
            elif entry.is_file() and entry.suffix == ".json" and entry.stem.startswith("L"):
                discovered.append(entry.stem)
        if discovered:
            return sorted(set(discovered))

    raise FileNotFoundError(f"Could not load or discover master video list from {list_path}")


class PathResolver:
    """Fast, safe path resolution engine for custom dataset folder hierarchies."""
    def __init__(
        self,
        videos_root: Path | None = None,
        keyframes_root: Path | None = None,
        map_keyframes_root: Path | None = None,
        objects_root: Path | None = None,
    ) -> None:
        self.videos_root = videos_root.expanduser().resolve() if videos_root else None
        self.keyframes_root = keyframes_root.expanduser().resolve() if keyframes_root else None
        self.map_keyframes_root = map_keyframes_root.expanduser().resolve() if map_keyframes_root else None
        self.objects_root = objects_root.expanduser().resolve() if objects_root else None
        self._video_index: dict[str, Path] | None = None
        self._keyframe_index: dict[str, Path] | None = None
        self._objects_index: dict[str, Path] | None = None
        self._map_csv_index: dict[str, Path] | None = None

    def _build_video_index(self) -> dict[str, Path]:
        if self._video_index is not None:
            return self._video_index
        index: dict[str, Path] = {}
        scan_roots: list[Path] = []
        if self.videos_root and self.videos_root.exists():
            scan_roots.append(self.videos_root)
        for fallback in (
            Path("/kaggle/input/datasets/lyduchoang/aic-26-video/Video"),
            Path("/kaggle/input/aic-26-video/Video"),
            Path("/kaggle/input"),
            REPO_ROOT / "data" / "videos",
            REPO_ROOT / "data",
        ):
            if fallback.exists() and fallback not in scan_roots:
                scan_roots.append(fallback)

        for scan_root in scan_roots:
            logger.info("Fast-indexing video files under %s ...", scan_root)
            for root, dirs, files in os.walk(scan_root):
                # Skip heavy keyframe/image/object folders to prevent slow scans
                dirs[:] = [d for d in dirs if not d.lower().startswith(("keyframe", "frame", "object", "mask", "desc", "ocr", "map-key"))]
                for f in files:
                    if f.endswith(".mp4") and f.startswith("L"):
                        p = Path(root) / f
                        if p.stem not in index:
                            index[p.stem] = p
                if len(index) >= 873:
                    break
            if len(index) >= 873:
                break

        logger.info("Indexed %d video MP4 files.", len(index))
        self._video_index = index
        return index

    def resolve_video_path(self, video_id: str) -> Path:
        index = self._build_video_index()
        if video_id in index and index[video_id].is_file():
            return index[video_id]

        if self.videos_root:
            batch = video_id.split("_")[0]  # e.g. L21
            candidates = [
                self.videos_root / f"Videos_{batch}" / "video" / f"{video_id}.mp4",
                self.videos_root / f"Videos_{batch}" / f"{video_id}.mp4",
                self.videos_root / "Videos" / f"Videos_{batch}" / "video" / f"{video_id}.mp4",
                self.videos_root / "Videos" / f"{video_id}.mp4",
                self.videos_root / "video" / f"{video_id}.mp4",
                self.videos_root / f"{video_id}.mp4",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate

        raise FileNotFoundError(f"Video file {video_id}.mp4 not found in index or under {self.videos_root}")

    def _build_keyframe_index(self) -> dict[str, Path]:
        if self.keyframes_root is None:
            return {}
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
        if self.objects_root is None:
            return {}
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
        if self.map_keyframes_root is None:
            return {}
        if self._map_csv_index is not None:
            return self._map_csv_index
        logger.info("Fast-indexing map-keyframes CSVs under %s ...", self.map_keyframes_root)
        index: dict[str, Path] = {}
        for root_to_scan in (self.map_keyframes_root, REPO_ROOT / "data" / "map-keyframes"):
            if root_to_scan and root_to_scan.exists():
                for root, _, files in os.walk(root_to_scan):
                    for f in files:
                        if f.endswith(".csv") and f.startswith("L"):
                            p = Path(root) / f
                            index[p.stem] = p
        logger.info("Indexed %d map CSV files.", len(index))
        self._map_csv_index = index
        return index

    def resolve_frames_dir(self, video_id: str) -> Path:
        if self.keyframes_root is None:
            raise FileNotFoundError("keyframes_root is not configured")
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

    def resolve_objects_dir(self, video_id: str) -> Path | None:
        if self.objects_root is None:
            return None
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

        return None

    def resolve_map_csv(self, video_id: str) -> Path:
        if self.map_keyframes_root is None:
            raise FileNotFoundError("map_keyframes_root is not configured")
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


def find_rclone_config() -> str | None:
    """Find the first available rclone configuration file."""
    candidates = [
        os.environ.get("RCLONE_CONFIG"),
        str(Path.home() / ".config" / "rclone" / "rclone.conf"),
        str(Path.home() / ".rclone.conf"),
        "/root/.config/rclone/rclone.conf",
        "/root/.rclone.conf",
        "/kaggle/working/.rclone.conf",
        "/tmp/rclone.conf",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    return None


def rclone_sync_file(local_path: Path, rclone_dest: str, max_retries: int = 10) -> bool:
    """Sync a single file to rclone remote with 10-attempt exponential backoff retries and jitter."""
    if not local_path.is_file():
        logger.error("  [RCLONE] Source file does not exist: %s", local_path)
        return False

    dest_url = rclone_dest if rclone_dest.endswith("/") else rclone_dest + "/"
    config_path = find_rclone_config()

    base_cmd = ["rclone", "copyto", str(local_path), dest_url + local_path.name]
    if config_path:
        base_cmd.extend(["--config", config_path])
    base_cmd.extend([
        "--retries", "10",
        "--retries-sleep", "3s",
        "--low-level-retries", "20",
        "--timeout", "5m",
        "--contimeout", "60s",
        "--drive-chunk-size", "64M",
        "--tpslimit", "5",
    ])

    backoff_delays = [5, 10, 15, 25, 35, 45, 60, 90, 120, 180]
    for attempt in range(1, max_retries + 1):
        try:
            res = subprocess.run(
                base_cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if res.returncode == 0:
                logger.info("  [RCLONE] Synced %s -> %s", local_path.name, dest_url)
                return True
            else:
                err_out = (res.stderr or res.stdout or "").strip()
                logger.warning(
                    "  [RCLONE] Attempt %d/%d failed for %s (exit code %d): %s",
                    attempt,
                    max_retries,
                    local_path.name,
                    res.returncode,
                    err_out[:300] if err_out else "No output",
                )
        except subprocess.TimeoutExpired:
            logger.warning("  [RCLONE] Attempt %d/%d timed out (300s) for %s", attempt, max_retries, local_path.name)
        except Exception as exc:
            logger.warning("  [RCLONE] Attempt %d/%d exception for %s: %s", attempt, max_retries, local_path.name, exc)

        if attempt < max_retries:
            base_delay = backoff_delays[min(attempt - 1, len(backoff_delays) - 1)]
            delay = base_delay + random.uniform(1.0, 5.0)
            logger.info("  [RCLONE] Retrying upload in %.1fs (attempt %d/%d)...", delay, attempt + 1, max_retries)
            time.sleep(delay)

    return False


def rclone_sync_dir(local_dir: Path, rclone_dest: str) -> bool:
    """Bulk sync an entire local directory to rclone remote."""
    if not local_dir.is_dir():
        return False
    config_path = find_rclone_config()
    cmd = ["rclone", "copy", str(local_dir), rclone_dest]
    if config_path:
        cmd.extend(["--config", config_path])
    cmd.extend([
        "--retries", "5",
        "--retries-sleep", "3s",
        "--low-level-retries", "10",
        "--timeout", "5m",
        "--contimeout", "60s",
        "--tpslimit", "5",
    ])
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return res.returncode == 0
    except Exception:
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
    print(f" Frame Extractor:      {args.frame_extractor.upper()}")
    print(f" Videos Source:        {args.videos_root}")
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
        videos_root=args.videos_root,
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
                if args.frame_extractor == "transnetv2":
                    v_path = resolver.resolve_video_path(vid)
                    source_desc = f"Video: {v_path.name}"
                else:
                    f_dir = resolver.resolve_frames_dir(vid)
                    m_csv = resolver.resolve_map_csv(vid)
                    source_desc = f"Frames: {f_dir.name} | Map: {m_csv.name}"

                o_dir = resolver.resolve_objects_dir(vid)
                if idx == 1 and o_dir is not None:
                    sample_json = next(o_dir.glob("*.json"), None)
                    if sample_json and sample_json.is_file():
                        raw_dets = load_organizer_detections(sample_json)
                        filtered = filter_detections(raw_dets, sample_config)
                        labels = [d.class_entity for d in filtered]
                        logger.info("  🎯 Sample Box Filter Test (%s/%s): %d raw boxes -> %d distinct objects: %s", vid, sample_json.name, len(raw_dets), len(filtered), labels)
                if idx <= 5 or idx == len(assigned_videos):
                    obj_info = o_dir.name if o_dir else f"(Dynamic {args.detector.upper()})"
                    logger.info("  [%d/%d] %s -> %s | Objects: %s", idx, len(assigned_videos), vid, source_desc, obj_info)
            except FileNotFoundError as error:
                logger.error("  ❌ Missing path for %s: %s", vid, error)
                missing_count += 1
        if missing_count == 0:
            logger.info("✅ DRY-RUN SUCCESS: All %d assigned videos have valid verified paths!", len(assigned_videos))
            return 0
        else:
            logger.error("❌ DRY-RUN FAILED: %d videos have missing source paths.", missing_count)
            return 1

    # 3. Remote Destination Startup Pre-Flight Check
    if args.rclone_dest and not args.dry_run:
        config_path = find_rclone_config()
        test_cmd = ["rclone", "mkdir", args.rclone_dest]
        if config_path:
            test_cmd.extend(["--config", config_path])
        try:
            t_res = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
            if t_res.returncode == 0:
                logger.info("📡 Rclone destination %s verified reachable and writable!", args.rclone_dest)
            else:
                logger.warning("⚠️ Rclone destination check warning (code %d): %s", t_res.returncode, t_res.stderr.strip()[:200])
        except Exception as exc:
            logger.warning("⚠️ Rclone destination check exception: %s", exc)

    # 4. Main Batch Execution Loop
    total_assigned = len(assigned_videos)
    completed_in_run = 0
    skipped_count = 0
    start_time_all = time.time()

    remote_completed_videos: set[str] = set()
    if args.rclone_dest and not args.no_resume:
        config_path = find_rclone_config()
        lsf_cmd = ["rclone", "lsf", args.rclone_dest, "-R", "--include", "*.jsonl", "--fast-list", "--tpslimit", "5"]
        if config_path:
            lsf_cmd.extend(["--config", config_path])
        for attempt in range(1, 11):
            try:
                res = subprocess.run(
                    lsf_cmd,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        fname = Path(line.strip())
                        if fname.suffix == ".jsonl" and not fname.stem.endswith(".manifest"):
                            remote_completed_videos.add(fname.stem)
                    if remote_completed_videos:
                        logger.info("📡 Pre-fetched %d completed videos from Google Drive destination!", len(remote_completed_videos))
                    break
                else:
                    logger.warning("Remote pre-fetch attempt %d/10 warning (code %d). Retrying in 3s...", attempt, res.returncode)
                    time.sleep(3 + random.uniform(1.0, 3.0))
            except Exception as exc:
                logger.warning("Remote pre-fetch attempt %d/10 exception: %s. Retrying in 3s...", attempt, exc)
                time.sleep(3 + random.uniform(1.0, 3.0))

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
        is_done = video_id in remote_completed_videos or is_video_completed(description_artifact, description_manifest)
        if not args.no_resume and not is_done and not remote_completed_videos and args.rclone_dest:
            # Fallback direct check for this specific video if global pre-fetch was empty
            config_path = find_rclone_config()
            chk_cmd = ["rclone", "lsf", f"{args.rclone_dest.rstrip('/')}/{video_id}.jsonl"]
            if config_path:
                chk_cmd.extend(["--config", config_path])
            try:
                chk_res = subprocess.run(chk_cmd, capture_output=True, text=True, timeout=15)
                if chk_res.returncode == 0 and video_id in chk_res.stdout:
                    is_done = True
                    remote_completed_videos.add(video_id)
            except Exception:
                pass

        if not args.no_resume and is_done:
            logger.info("  [SKIP - ALREADY COMPLETED IN GDRIVE/LOCAL] %s", video_id)
            skipped_count += 1
            continue

        objects_dir = resolver.resolve_objects_dir(video_id)
        limit_args = ["--limit", str(args.limit)] if args.limit else []
        resume_args = ["--resume"] if not args.no_resume else []

        # Stage 0: Frame Manifest / Keyframe Extraction
        if args.frame_extractor == "transnetv2":
            try:
                video_path = resolver.resolve_video_path(video_id)
            except FileNotFoundError as error:
                logger.error("  ❌ Skipping %s due to missing video: %s", video_id, error)
                continue

            stage0_cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts/extract_transnet_frames.py"),
                "--config", str(args.config),
                "--video-id", video_id,
                "--video-path", str(video_path),
                "--output-root", str(output_root),
                "--device", args.device,
                *limit_args,
            ]
            if args.no_resume:
                stage0_cmd.append("--no-resume")
            run_pipeline_step(stage0_cmd, step_name="extract_transnet_frames")
            data_root = output_root
        else:
            try:
                frames_dir = resolver.resolve_frames_dir(video_id)
                map_csv = resolver.resolve_map_csv(video_id)
            except FileNotFoundError as error:
                logger.error("  ❌ Skipping %s due to missing path: %s", video_id, error)
                continue

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
            data_root = resolver.keyframes_root

        # Stage 1: SAM Masks / Object Detection
        stage1_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/prepare_object_masks.py"),
            "--config", str(args.config),
            "--video-id", video_id,
            "--data-root", str(data_root),
            "--output-root", str(output_root),
            "--cache-root", str(cache_root),
            "--frame-manifest", str(frame_manifest),
            "--detector", args.detector,
            "--detector-model", args.detector_model,
            "--output", str(mask_artifact),
            "--device", args.device,
            *resume_args,
            *limit_args,
        ]
        if objects_dir is not None:
            stage1_cmd.extend(["--objects-dir", str(objects_dir)])
        run_pipeline_step(stage1_cmd, step_name="prepare_object_masks")

        # Stage 2: DAM Descriptions
        run_pipeline_step(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/run_dam_descriptions.py"),
                "--config", str(args.config),
                "--video-id", video_id,
                "--data-root", str(data_root),
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

        # Stage 3: OCR Text Extraction (Optional / Default True)
        ocr_artifact = output_root / "ocr" / "transcripts" / f"{video_id}.jsonl"
        ocr_manifest = ocr_artifact.with_suffix(".manifest.json")
        if args.enable_ocr:
            ocr_cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts/run_ocr_extraction.py"),
                "--config", str(args.config),
                "--video-id", video_id,
                "--frame-manifest", str(frame_manifest),
                "--data-root", str(data_root),
                "--output", str(ocr_artifact),
                "--device", args.device,
                "--backend", args.ocr_backend,
                "--threshold", str(args.ocr_threshold),
            ]
            if args.no_resume:
                ocr_cmd.append("--no-resume")
            if args.limit:
                ocr_cmd.extend(["--limit", str(args.limit)])
            run_pipeline_step(ocr_cmd, step_name="run_ocr_extraction")

        # Stage 4: SigLIP2 Dense Scene Embeddings (Optional / Default True)
        emb_artifact = output_root / "scene_embeddings" / f"{video_id}.jsonl"
        if args.enable_siglip:
            siglip_cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts/run_scene_embeddings.py"),
                "--config", str(args.config),
                "--video-id", video_id,
                "--frame-manifest", str(frame_manifest),
                "--data-root", str(data_root),
                "--output", str(emb_artifact),
                "--device", args.device,
                "--matrix-format", "safetensors",
            ]
            if args.no_resume:
                siglip_cmd.append("--no-resume")
            if args.limit:
                siglip_cmd.extend(["--limit", str(args.limit)])
            run_pipeline_step(siglip_cmd, step_name="run_scene_embeddings")

        # Stage 5: Multi-Modal Metadata Fusion
        unified_artifact = output_root / "unified_metadata" / f"{video_id}.jsonl"
        fusion_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/build_unified_frame_metadata.py"),
            "--video-id", video_id,
            "--frame-manifest", str(frame_manifest),
            "--descriptions", str(description_artifact),
            "--output", str(unified_artifact),
        ]
        if args.enable_ocr and ocr_artifact.is_file():
            fusion_cmd.extend(["--ocr-transcripts", str(ocr_artifact)])
        if args.enable_siglip and emb_artifact.is_file():
            fusion_cmd.extend(["--embeddings", str(emb_artifact)])
        run_pipeline_step(fusion_cmd, step_name="build_unified_frame_metadata")

        # Atomic & Fault-Tolerant Google Drive Upload
        if args.rclone_dest:
            rclone_base = args.rclone_dest.rstrip('/')
            try:
                # 1. Package keyframes into zip archive
                keyframes_dir = output_root / "keyframes" / video_id
                keyframe_zip = output_root / "keyframes_zips" / f"{video_id}.zip"
                if create_keyframe_zip(keyframes_dir, keyframe_zip):
                    rclone_sync_file(keyframe_zip, f"{rclone_base}/keyframes_zips/")

                # 2. Sync DAM descriptions
                if description_artifact.is_file():
                    rclone_sync_file(description_artifact, f"{rclone_base}/descriptions/")
                if description_manifest.is_file():
                    rclone_sync_file(description_manifest, f"{rclone_base}/descriptions/")

                # 3. Sync OCR transcripts
                if ocr_artifact.is_file():
                    rclone_sync_file(ocr_artifact, f"{rclone_base}/ocr_transcripts/")

                # 4. Sync SigLIP2 Scene Embeddings
                safetensors_mat = output_root / "scene_embeddings" / f"{video_id}.safetensors"
                if safetensors_mat.is_file():
                    rclone_sync_file(safetensors_mat, f"{rclone_base}/scene_embeddings/")
                if emb_artifact.is_file():
                    rclone_sync_file(emb_artifact, f"{rclone_base}/scene_embeddings/")

                # 5. Sync map-keyframes CSV
                map_csv = output_root / "map-keyframes" / f"{video_id}.csv"
                if map_csv.is_file():
                    rclone_sync_file(map_csv, f"{rclone_base}/map-keyframes/")

                # 6. LAST FILE: Sync search-ready Unified Metadata (acts as atomic completion lock)
                if unified_artifact.is_file():
                    rclone_sync_file(unified_artifact, f"{rclone_base}/unified_metadata/")
                    remote_completed_videos.add(video_id)
            except Exception as rclone_exc:
                logger.error("  ❌ [RCLONE] Exception during sync for %s: %s (will continue to next video)", video_id, rclone_exc)

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
