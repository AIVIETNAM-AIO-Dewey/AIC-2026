#!/usr/bin/env python3
"""Validate a multi-video organizer-compatible self-cut keyframe package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aic2026.common.frame_manifest import read_frame_map  # noqa: E402
from aic2026.common.io import iter_jsonl  # noqa: E402
from aic2026.contracts import FrameSampleRecord  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--video-id", action="append", required=True, dest="video_ids")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package_root = args.package_root.expanduser().resolve()
    benchmark_path = package_root / "benchmark" / "t4x2_summary.json"
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"T4x2 benchmark summary is missing: {benchmark_path}")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if set(benchmark.get("videos", {})) != set(args.video_ids):
        raise ValueError("Benchmark summary does not contain exactly the requested videos")
    report: dict[str, object] = {
        "package_root": str(package_root),
        "benchmark": str(benchmark_path),
        "videos": {},
    }
    total_images = 0
    for video_id in args.video_ids:
        batch = video_id.split("_", maxsplit=1)[0]
        frames_dir = package_root / f"Keyframes_{batch}" / "keyframes" / video_id
        map_path = package_root / "map-keyframes" / f"{video_id}.csv"
        manifest_path = package_root / "manifests" / f"{video_id}.jsonl"
        rows = read_frame_map(map_path)
        records = [
            FrameSampleRecord.model_validate(value)
            for value in iter_jsonl(manifest_path)
        ]
        images = sorted(frames_dir.glob("*.jpg"))
        expected_names = [f"{index:06d}.jpg" for index in range(1, len(rows) + 1)]
        if [path.name for path in images] != expected_names:
            raise ValueError(f"Image numbering is not contiguous for {video_id}")
        if len(rows) != len(records) or len(rows) != len(images):
            raise ValueError(f"CSV/manifest/image counts differ for {video_id}")
        for row, record, image_path in zip(rows, records, images, strict=True):
            if (
                row.keyframe_n != record.keyframe_n
                or row.frame_idx != record.frame_idx
                or abs(row.pts_time_s - record.pts_time_s) > 1e-6
            ):
                raise ValueError(f"Map/manifest mismatch for {video_id} n={row.keyframe_n}")
            if record.extraction_method != "frame-index-select":
                raise ValueError(f"Unexpected extraction method: {record.extraction_method}")
            with Image.open(image_path) as image:
                if image.size != (record.width, record.height):
                    raise ValueError(f"Image dimensions mismatch: {image_path}")
                image.verify()
        report["videos"][video_id] = {
            "images": len(images),
            "first_frame_idx": rows[0].frame_idx,
            "last_frame_idx": rows[-1].frame_idx,
            "map_csv": str(map_path),
            "manifest": str(manifest_path),
        }
        total_images += len(images)
        print(
            f"[package_validation] video_id={video_id} images={len(images)} result=passed",
            file=sys.stderr,
            flush=True,
        )
    report.update({"status": "completed", "total_images": total_images})
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
