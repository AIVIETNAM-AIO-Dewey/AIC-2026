#!/usr/bin/env python3
"""Decode canonical dense TRAKE frames at 5 FPS without renumbering frame_idx."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _common import add_common_arguments, read_config  # noqa: E402

from aic2026.common.io import write_jsonl_atomic  # noqa: E402
from aic2026.contracts import FrameRef  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args(argv)
    config = read_config(args.config or REPO_ROOT / "configs" / "offline" / "dense_frames.yaml")
    try:
        import cv2
    except ImportError as error:
        raise SystemExit(
            "Install requirements/easyocr.txt (opencv-python-headless) first"
        ) from error
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise SystemExit("Video reports invalid FPS")
    every = max(1, round(fps / float(config.get("sampling_fps", 5.0))))
    video_id = args.video_id or args.video.stem
    output_root = args.output_root or REPO_ROOT / "artifacts"
    output_dir = output_root / "dense_frames" / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[FrameRef] = []
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % every == 0:
            height, width = frame.shape[:2]
            relative = Path("dense_frames") / video_id / f"{frame_idx}.jpg"
            target = output_root / relative
            if not cv2.imwrite(
                str(target),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, int(config.get("thumbnail_jpeg_quality", 90))],
            ):
                raise SystemExit(f"Cannot write thumbnail: {target}")
            records.append(
                FrameRef(
                    video_id=video_id,
                    frame_uid=f"{video_id}:{frame_idx}",
                    keyframe_n=frame_idx + 1,
                    frame_idx=frame_idx,
                    pts_time_s=frame_idx / fps,
                    fps=fps,
                    frame_relpath=relative.as_posix(),
                    width=width,
                    height=height,
                )
            )
        frame_idx += 1
        if args.limit and len(records) >= args.limit:
            break
    capture.release()
    output = output_root / "dense_scene_embeddings" / f"{video_id}.frames.jsonl"
    write_jsonl_atomic(output, records)
    print(json.dumps({"video_id": video_id, "frames": len(records), "manifest": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
