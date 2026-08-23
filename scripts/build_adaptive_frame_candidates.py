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
from _common import read_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--single-frame-before-s", type=float)
    parser.add_argument("--cadence-from-s", type=float)
    parser.add_argument("--extreme-shot-from-s", type=float)
    parser.add_argument("--cadence-s", type=float)
    parser.add_argument("--max-frames-per-shot", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = read_config(args.config)
    sampling_config = config.get("adaptive_sampling", {})
    if not isinstance(sampling_config, dict):
        raise ValueError("config.adaptive_sampling must be a mapping")
    policy = {
        "single_frame_before_s": (
            args.single_frame_before_s
            if args.single_frame_before_s is not None
            else float(sampling_config.get("single_frame_before_s", 2.0))
        ),
        "cadence_from_s": (
            args.cadence_from_s
            if args.cadence_from_s is not None
            else float(sampling_config.get("cadence_from_s", 4.0))
        ),
        "extreme_shot_from_s": (
            args.extreme_shot_from_s
            if args.extreme_shot_from_s is not None
            else float(sampling_config.get("extreme_shot_from_s", 7.0))
        ),
        "cadence_s": (
            args.cadence_s
            if args.cadence_s is not None
            else float(sampling_config.get("cadence_s", 1.5))
        ),
        "max_frames_per_shot": (
            args.max_frames_per_shot
            if args.max_frames_per_shot is not None
            else int(sampling_config.get("max_frames_per_shot", 10))
        ),
    }
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
        **policy,
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
                "sampling_policy": policy,
                "output": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
