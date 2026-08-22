#!/usr/bin/env python3
"""Build a TransNet-only keyframe package in organizer-compatible layout."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aic2026.common.frame_manifest import read_frame_map  # noqa: E402
from aic2026.common.io import iter_jsonl, write_jsonl_atomic  # noqa: E402
from aic2026.contracts import FrameSampleRecord  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _validate_source_records(
    records: list[FrameSampleRecord],
    *,
    output_root: Path,
) -> list[FrameSampleRecord]:
    if not records:
        raise ValueError("Adaptive frame manifest is empty")
    ordered = sorted(records, key=lambda record: (record.frame_idx, record.pts_time_s))
    frame_indices = [record.frame_idx for record in ordered]
    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError("Adaptive frame manifest contains duplicate frame_idx values")
    for record in ordered:
        if record.sampling_source != "transnetv2":
            raise ValueError(
                f"TransNet-only package cannot include source={record.sampling_source!r}"
            )
        expected_idx = round(record.pts_time_s * record.fps)
        if record.frame_idx != expected_idx:
            raise ValueError(
                f"Non-canonical frame_idx for {record.frame_uid}: "
                f"got {record.frame_idx}, expected {expected_idx}"
            )
        image_path = output_root / record.frame_relpath
        if not image_path.is_file():
            raise FileNotFoundError(f"Adaptive frame is missing: {image_path}")
        with Image.open(image_path) as image:
            if image.size != (record.width, record.height):
                raise ValueError(f"Image dimensions do not match manifest: {image_path}")
            image.verify()
    return ordered


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    output_root = args.output_root.expanduser().resolve()
    source_manifest = (
        args.source_manifest.expanduser().resolve()
        if args.source_manifest
        else output_root
        / "frame_extraction"
        / "adaptive_manifests"
        / f"{args.video_id}.jsonl"
    )
    package_root = (
        args.package_root.expanduser().resolve()
        if args.package_root
        else output_root / "gdrive_export" / "transnetv2-only"
    )
    records = [
        FrameSampleRecord.model_validate(value)
        for value in iter_jsonl(source_manifest)
    ]
    print(f"[keyframe_package] source_manifest={source_manifest}", file=sys.stderr, flush=True)
    print(f"[keyframe_package] source_records={len(records)}", file=sys.stderr, flush=True)
    print(f"[keyframe_package] package_root={package_root}", file=sys.stderr, flush=True)
    records = _validate_source_records(records, output_root=output_root)
    print("[keyframe_package] source_validation=passed", file=sys.stderr, flush=True)

    batch = args.video_id.split("_", maxsplit=1)[0]
    frames_dir = package_root / f"Keyframes_{batch}" / "keyframes" / args.video_id
    temporary_frames_dir = frames_dir.with_name(frames_dir.name + ".tmp")
    map_path = package_root / "map-keyframes" / f"{args.video_id}.csv"
    manifest_path = package_root / "manifests" / f"{args.video_id}.jsonl"
    if temporary_frames_dir.exists():
        shutil.rmtree(temporary_frames_dir)
    temporary_frames_dir.mkdir(parents=True)

    package_records: list[FrameSampleRecord] = []
    for keyframe_n, record in enumerate(records, start=1):
        source_image = output_root / record.frame_relpath
        image_name = f"{keyframe_n:06d}.jpg"
        destination = temporary_frames_dir / image_name
        _link_or_copy(source_image, destination)
        payload = record.model_dump(mode="json")
        payload.update(
            {
                "sample_n": keyframe_n,
                "keyframe_n": keyframe_n,
                "frame_relpath": (
                    Path(f"Keyframes_{batch}")
                    / "keyframes"
                    / args.video_id
                    / image_name
                ).as_posix(),
            }
        )
        package_records.append(FrameSampleRecord.model_validate(payload))
        if (
            keyframe_n == 1
            or keyframe_n % args.progress_every == 0
            or keyframe_n == len(records)
        ):
            print(
                f"[keyframe_package] packaged={keyframe_n}/{len(records)}",
                file=sys.stderr,
                flush=True,
            )

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    temporary_frames_dir.replace(frames_dir)

    map_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_map = map_path.with_suffix(map_path.suffix + ".tmp")
    with temporary_map.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("n", "pts_time", "fps", "frame_idx"),
            lineterminator="\n",
        )
        writer.writeheader()
        for record in package_records:
            writer.writerow(
                {
                    "n": record.keyframe_n,
                    "pts_time": f"{record.pts_time_s:.6f}",
                    "fps": f"{record.fps:.6f}",
                    "frame_idx": record.frame_idx,
                }
            )
    temporary_map.replace(map_path)
    write_jsonl_atomic(manifest_path, package_records)

    print("[keyframe_package] package_validation=running", file=sys.stderr, flush=True)
    mapped = read_frame_map(map_path)
    expected_names = [f"{index:06d}.jpg" for index in range(1, len(records) + 1)]
    actual_names = sorted(path.name for path in frames_dir.glob("*.jpg"))
    if actual_names != expected_names:
        raise ValueError("Packaged image names do not match organizer keyframe numbering")
    if len(mapped) != len(package_records):
        raise ValueError("Packaged map row count does not match frame count")
    for row, record in zip(mapped, package_records, strict=True):
        if (
            row.keyframe_n != record.keyframe_n
            or row.frame_idx != record.frame_idx
            or abs(row.pts_time_s - record.pts_time_s) > 1e-5
        ):
            raise ValueError(f"Packaged map does not match frame record n={record.keyframe_n}")
    print("[keyframe_package] package_validation=passed", file=sys.stderr, flush=True)

    print(
        json.dumps(
            {
                "status": "completed",
                "video_id": args.video_id,
                "frames": len(package_records),
                "package_root": str(package_root),
                "frames_dir": str(frames_dir),
                "map_csv": str(map_path),
                "manifest": str(manifest_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
