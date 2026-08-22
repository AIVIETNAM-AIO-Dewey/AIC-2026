#!/usr/bin/env python3
"""Extract all adaptive frame candidates into canonical frame-index paths."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aic2026.common.io import iter_jsonl, write_jsonl_atomic  # noqa: E402
from aic2026.contracts import FrameSampleRecord  # noqa: E402
from aic2026.frame_extraction.ffmpeg import extract_frames_by_index, probe_video  # noqa: E402
from aic2026.frame_extraction.sampling import FrameSampleCandidate  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jpeg-quality", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def _read_candidates(path: Path) -> list[FrameSampleCandidate]:
    field_names = {field.name for field in fields(FrameSampleCandidate)}
    candidates = [
        FrameSampleCandidate(**{key: value for key, value in raw.items() if key in field_names})
        for raw in iter_jsonl(path)
    ]
    if not candidates:
        raise ValueError(f"Adaptive candidate manifest is empty: {path}")
    frame_indices = [candidate.frame_idx for candidate in candidates]
    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError(f"Adaptive candidate manifest has duplicate frame_idx values: {path}")
    return candidates


def _valid_image(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            size = image.size
            image.verify()
        return size
    except (OSError, UnidentifiedImageError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    video_path = args.video_path.expanduser().resolve()
    candidates_path = args.candidates.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    candidates = _read_candidates(candidates_path)
    probe = probe_video(video_path)
    frames_dir = output_root / "adaptive_keyframes" / args.video_id
    manifest_path = (
        output_root / "frame_extraction" / "adaptive_manifests" / f"{args.video_id}.jsonl"
    )
    records: list[FrameSampleRecord] = []

    print(f"[adaptive_extraction] video={video_path}", file=sys.stderr, flush=True)
    print(f"[adaptive_extraction] candidates={len(candidates)}", file=sys.stderr, flush=True)
    print(f"[adaptive_extraction] frames_dir={frames_dir}", file=sys.stderr, flush=True)
    missing: list[tuple[int, Path]] = []
    reused = 0
    for candidate in candidates:
        expected_idx = round(candidate.pts_time_s * probe.fps)
        if candidate.frame_idx != expected_idx:
            raise ValueError(
                f"Non-canonical frame_idx at candidate {sample_n}: "
                f"got {candidate.frame_idx}, expected {expected_idx}"
            )
        frame_path = frames_dir / f"{candidate.frame_idx:08d}.jpg"
        size = _valid_image(frame_path)
        if size is None:
            missing.append((candidate.frame_idx, frame_path))
        else:
            reused += 1

    print(
        f"[adaptive_extraction] exact_index_batch missing={len(missing)} reused={reused}",
        file=sys.stderr,
        flush=True,
    )
    extract_frames_by_index(
        video_path=video_path,
        outputs=missing,
        jpeg_quality=args.jpeg_quality,
    )

    for sample_n, candidate in enumerate(candidates, start=1):
        frame_path = frames_dir / f"{candidate.frame_idx:08d}.jpg"
        size = _valid_image(frame_path)
        if size is None:
            raise ValueError(f"Extracted image is invalid: {frame_path}")
        width, height = size
        records.append(
            FrameSampleRecord(
                video_id=args.video_id,
                frame_uid=f"{args.video_id}:{candidate.frame_idx}",
                sample_n=sample_n,
                frame_idx=candidate.frame_idx,
                pts_time_s=candidate.pts_time_s,
                fps=probe.fps,
                frame_relpath=frame_path.relative_to(output_root).as_posix(),
                width=width,
                height=height,
                source_video=str(video_path),
                sampling_source="transnetv2",
                extraction_method="frame-index-select",
                shot_id=candidate.shot_id,
                shot_start_idx=candidate.shot_start_idx,
                shot_end_idx=candidate.shot_end_idx,
            )
        )
        if sample_n == 1 or sample_n % args.progress_every == 0 or sample_n == len(candidates):
            print(
                f"[adaptive_extraction] frames={sample_n}/{len(candidates)} reused={reused}",
                file=sys.stderr,
                flush=True,
            )

    write_jsonl_atomic(manifest_path, records)
    print(
        json.dumps(
            {
                "status": "completed",
                "frames": len(records),
                "reused": reused,
                "manifest": str(manifest_path),
                "frames_dir": str(frames_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
