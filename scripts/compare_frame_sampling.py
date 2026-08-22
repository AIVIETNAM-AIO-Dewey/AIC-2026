#!/usr/bin/env python3
"""Compare organizer keyframes with TransNetV2 adaptive frame candidates."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from bisect import bisect_left
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aic2026.common.io import atomic_write_json, iter_jsonl, write_jsonl_atomic  # noqa: E402
from aic2026.contracts import ShotRecord  # noqa: E402
from aic2026.frame_extraction.ffmpeg import extract_frame  # noqa: E402
from aic2026.frame_extraction.sampling import (  # noqa: E402
    FrameSampleCandidate,
    adaptive_samples_from_shots,
    dedupe_samples,
    map_keyframe_samples,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--map-csv", type=Path, required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tolerance-s", type=float, default=0.5)
    parser.add_argument("--preview-pairs", type=int, default=8)
    parser.add_argument("--preview-selection", choices=("evenly", "page"), default="evenly")
    parser.add_argument("--preview-page", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=2)
    return parser


def _nearest(
    sample: FrameSampleCandidate,
    candidates: list[FrameSampleCandidate],
    candidate_times: list[float],
) -> tuple[FrameSampleCandidate, float]:
    position = bisect_left(candidate_times, sample.pts_time_s)
    indices = {max(0, position - 1), min(len(candidates) - 1, position)}
    nearest = min(indices, key=lambda index: abs(candidate_times[index] - sample.pts_time_s))
    candidate = candidates[nearest]
    return candidate, abs(candidate.pts_time_s - sample.pts_time_s)


def _temporal_gap_stats(samples: list[FrameSampleCandidate]) -> dict[str, float | None]:
    times = sorted(sample.pts_time_s for sample in samples)
    gaps = [current - previous for previous, current in zip(times, times[1:], strict=False)]
    if not gaps:
        return {"median_s": None, "p95_s": None, "max_s": None}
    ordered = sorted(gaps)
    p95_index = round(0.95 * (len(ordered) - 1))
    return {
        "median_s": statistics.median(gaps),
        "p95_s": ordered[p95_index],
        "max_s": max(gaps),
    }


def _select_previews(
    samples: list[FrameSampleCandidate],
    *,
    count: int,
    selection: str,
    page: int,
) -> list[tuple[int, FrameSampleCandidate]]:
    if not samples or count <= 0:
        return []
    if selection == "page":
        start = (page - 1) * count
        if start >= len(samples):
            page_count = (len(samples) + count - 1) // count
            raise ValueError(f"--preview-page must be between 1 and {page_count}")
        return [
            (index + 1, samples[index])
            for index in range(start, min(start + count, len(samples)))
        ]
    if count >= len(samples):
        return list(enumerate(samples, start=1))
    if count <= 0:
        return []
    if count == 1:
        index = len(samples) // 2
        return [(index + 1, samples[index])]
    indices = {
        round(index * (len(samples) - 1) / (count - 1))
        for index in range(count)
    }
    return [(index + 1, samples[index]) for index in sorted(indices)]


def _sample_payload(sample: FrameSampleCandidate) -> dict[str, object]:
    return asdict(sample)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tolerance_s < 0:
        raise ValueError("--tolerance-s must be non-negative")
    if args.preview_pairs < 0:
        raise ValueError("--preview-pairs must be non-negative")
    if args.preview_page < 1:
        raise ValueError("--preview-page must be positive")
    if args.preview_pairs and args.video_path is None:
        raise ValueError("--video-path is required when --preview-pairs is positive")

    organizer = map_keyframe_samples(args.map_csv.expanduser().resolve())
    shots = [
        ShotRecord.model_validate(value)
        for value in iter_jsonl(args.shots.expanduser().resolve())
    ]
    if not shots:
        raise ValueError(f"Shot manifest is empty: {args.shots}")
    adaptive = adaptive_samples_from_shots(shots)
    merged = dedupe_samples([*organizer, *adaptive], tolerance_s=args.tolerance_s)
    additions = [sample for sample in merged if sample.sampling_source == "transnetv2"]

    organizer_times = [sample.pts_time_s for sample in organizer]
    adaptive_times = [sample.pts_time_s for sample in adaptive]
    adaptive_matches = [_nearest(sample, organizer, organizer_times)[1] for sample in adaptive]
    organizer_matches = [_nearest(sample, adaptive, adaptive_times)[1] for sample in organizer]
    short_shots = sum(
        ((shot.shot_end_idx - shot.shot_start_idx + 1) / shot.fps) <= 3.0
        for shot in shots
    )

    output_dir = (
        args.output_root.expanduser().resolve()
        / "frame_extraction"
        / "comparison"
        / args.video_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    organizer_path = output_dir / "organizer_samples.jsonl"
    adaptive_path = output_dir / "transnetv2_adaptive_samples.jsonl"
    additions_path = output_dir / "transnetv2_additions.jsonl"
    merged_path = output_dir / "merged_samples.jsonl"
    preview_path = output_dir / "preview_pairs.jsonl"
    write_jsonl_atomic(organizer_path, map(_sample_payload, organizer))
    write_jsonl_atomic(adaptive_path, map(_sample_payload, adaptive))
    write_jsonl_atomic(additions_path, map(_sample_payload, additions))
    write_jsonl_atomic(merged_path, map(_sample_payload, merged))

    preview_records: list[dict[str, object]] = []
    selected = _select_previews(
        additions,
        count=args.preview_pairs,
        selection=args.preview_selection,
        page=args.preview_page,
    )
    if selected:
        assert args.video_path is not None
        video_path = args.video_path.expanduser().resolve()
        preview_dir = output_dir / "preview"
        for rendered_n, (addition_position, adaptive_sample) in enumerate(selected, start=1):
            original, delta_s = _nearest(adaptive_sample, organizer, organizer_times)
            pair_dir = preview_dir / f"addition_{addition_position:04d}"
            original_image = pair_dir / f"organizer_{original.frame_idx:08d}.jpg"
            adaptive_image = pair_dir / f"transnetv2_{adaptive_sample.frame_idx:08d}.jpg"
            if not original_image.is_file():
                extract_frame(
                    video_path=video_path,
                    pts_time_s=original.pts_time_s,
                    output_path=original_image,
                    jpeg_quality=args.jpeg_quality,
                )
            if not adaptive_image.is_file():
                extract_frame(
                    video_path=video_path,
                    pts_time_s=adaptive_sample.pts_time_s,
                    output_path=adaptive_image,
                    jpeg_quality=args.jpeg_quality,
                )
            preview_records.append(
                {
                    "pair_n": rendered_n,
                    "addition_position": addition_position,
                    "delta_s": delta_s,
                    "organizer": _sample_payload(original),
                    "transnetv2": _sample_payload(adaptive_sample),
                    "organizer_image": str(original_image),
                    "transnetv2_image": str(adaptive_image),
                }
            )
            print(
                f"[sampling_comparison] preview={rendered_n}/{len(selected)} "
                f"addition={addition_position}/{len(additions)} "
                f"organizer={original.pts_time_s:.3f}s "
                f"transnetv2={adaptive_sample.pts_time_s:.3f}s delta={delta_s:.3f}s",
                file=sys.stderr,
                flush=True,
            )
    write_jsonl_atomic(preview_path, preview_records)

    summary = {
        "video_id": args.video_id,
        "fps": shots[0].fps,
        "shots": len(shots),
        "short_shots_le_3s": short_shots,
        "long_shots_gt_3s": len(shots) - short_shots,
        "organizer_frames": len(organizer),
        "adaptive_candidates_before_dedupe": len(adaptive),
        "adaptive_candidates_overlapping_organizer": sum(
            delta <= args.tolerance_s for delta in adaptive_matches
        ),
        "transnetv2_additions_after_dedupe": len(additions),
        "merged_frames": len(merged),
        "organizer_frames_covered_by_adaptive": sum(
            delta <= args.tolerance_s for delta in organizer_matches
        ),
        "dedupe_tolerance_s": args.tolerance_s,
        "preview_selection": args.preview_selection,
        "preview_page": args.preview_page,
        "preview_page_size": args.preview_pairs,
        "preview_page_count": (
            (len(additions) + args.preview_pairs - 1) // args.preview_pairs
            if args.preview_pairs
            else 0
        ),
        "organizer_gap_stats": _temporal_gap_stats(organizer),
        "merged_gap_stats": _temporal_gap_stats(merged),
        "artifacts": {
            "organizer": str(organizer_path),
            "adaptive": str(adaptive_path),
            "additions": str(additions_path),
            "merged": str(merged_path),
            "preview_pairs": str(preview_path),
        },
    }
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, summary)

    print("\nOrganizer vs TransNetV2 sampling comparison")
    print(f"  shots: {summary['shots']}")
    print(f"  organizer frames: {summary['organizer_frames']}")
    print(f"  adaptive candidates: {summary['adaptive_candidates_before_dedupe']}")
    print(f"  overlapping organizer (+/-{args.tolerance_s:.1f}s): {summary['adaptive_candidates_overlapping_organizer']}")
    print(f"  new TransNetV2 additions: {summary['transnetv2_additions_after_dedupe']}")
    print(f"  merged frames: {summary['merged_frames']}")
    print(
        f"  preview: selection={args.preview_selection} page={args.preview_page}/"
        f"{summary['preview_page_count']} size={args.preview_pairs}"
    )
    print(f"  organizer max gap: {summary['organizer_gap_stats']['max_s']:.3f}s")
    print(f"  merged max gap: {summary['merged_gap_stats']['max_s']:.3f}s")
    print(json.dumps({"status": "completed", "summary": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
