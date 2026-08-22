#!/usr/bin/env python3
"""Build TransNetV2 adaptive frame candidates without organizer deduplication."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aic2026.common.io import iter_jsonl, write_jsonl_atomic  # noqa: E402
from aic2026.contracts import ShotRecord  # noqa: E402
from aic2026.frame_extraction.sampling import adaptive_samples_from_shots  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cadence-s", type=float, default=1.5)
    parser.add_argument("--max-frames-per-shot", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shots = [
        ShotRecord.model_validate(value)
        for value in iter_jsonl(args.shots.expanduser().resolve())
    ]
    if not shots:
        raise ValueError(f"Shot manifest is empty: {args.shots}")
    if any(shot.video_id != args.video_id for shot in shots):
        raise ValueError("Shot manifest contains a different video_id")
    candidates = adaptive_samples_from_shots(
        shots,
        cadence_s=args.cadence_s,
        max_frames_per_shot=args.max_frames_per_shot,
    )
    frame_indices = [candidate.frame_idx for candidate in candidates]
    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError("Adaptive sampling produced duplicate frame_idx values")
    output = (
        args.output.expanduser().resolve()
        if args.output
        else args.output_root.expanduser().resolve()
        / "frame_extraction"
        / "adaptive_candidates"
        / f"{args.video_id}.jsonl"
    )
    write_jsonl_atomic(output, (asdict(candidate) for candidate in candidates))
    print(
        json.dumps(
            {
                "status": "completed",
                "video_id": args.video_id,
                "shots": len(shots),
                "candidates": len(candidates),
                "output": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
