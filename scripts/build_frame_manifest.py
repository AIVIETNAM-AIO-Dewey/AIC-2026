#!/usr/bin/env python3
"""Build canonical frame JSONL from an organizer map-keyframes CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _common import add_common_arguments, read_config, resolve_seed, runtime_roots  # noqa: E402

from aic2026.common import (  # noqa: E402
    complete_manifest,
    create_manifest,
    fail_manifest,
    prepare_resume,
    write_jsonl_atomic,
    write_manifest,
)
from aic2026.common.frame_manifest import build_frame_refs, read_frame_map  # noqa: E402
from aic2026.common.io import iter_jsonl  # noqa: E402
from aic2026.contracts import FrameRef  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--map-csv", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--drop-duplicate-frame-idx",
        action="store_true",
        help=(
            "When several keyframes share one frame_idx, index only the middle one and "
            "discard the rest. No frame_idx is ever rewritten, so every indexed frame "
            "keeps the value the organizer shipped. The run manifest records the count."
        ),
    )
    return parser


def _configured_path(explicit: Path | None, config: dict, key: str, data_root: Path) -> Path | None:
    if explicit:
        return explicit.expanduser().resolve()
    raw = config.get("inputs", {}).get(key)
    if raw is None:
        return None
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (data_root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = read_config(args.config)
    seed = resolve_seed(args.seed, config)
    roots = runtime_roots(args, config, required=("data_root", "output_root"))
    video_id = args.video_id or config.get("video_id")
    if not video_id:
        raise SystemExit("--video-id is required")
    map_csv = _configured_path(args.map_csv, config, "map_csv", roots["data_root"])
    frames_dir = _configured_path(args.frames_dir, config, "frames_dir", roots["data_root"])
    if map_csv is None or frames_dir is None:
        raise SystemExit("--map-csv and --frames-dir are required")
    output = (
        args.output.expanduser().resolve()
        if args.output
        else roots["output_root"] / "frame_manifests" / f"{video_id}.jsonl"
    )
    manifest_path = output.with_suffix(".manifest.json")

    resolved = {
        "schema_version": config.get("schema_version", "1.0"),
        "video_id": video_id,
        "limit": args.limit,
        "drop_duplicate_frame_idx": args.drop_duplicate_frame_idx,
    }
    run_id = str(config.get("run", {}).get("run_id", "object-description-v1"))
    manifest = create_manifest(
        run_id=run_id,
        stage="frame_manifest",
        config=resolved,
        seed=seed,
        input_paths=[("map_csv", map_csv), ("frames", frames_dir)],
        repo_root=REPO_ROOT,
    )
    manifest, complete = prepare_resume(
        manifest_path=manifest_path,
        output_path=output,
        proposed=manifest,
        resume=args.resume,
    )
    if complete:
        records = [FrameRef.model_validate(value) for value in iter_jsonl(output)]
        print(
            json.dumps(
                {"status": "already_complete", "frames": len(records), "output": str(output)}
            )
        )
        return 0
    write_manifest(manifest_path, manifest)
    try:
        refs = build_frame_refs(
            video_id=video_id,
            map_csv=map_csv,
            frames_dir=frames_dir,
            data_root=roots["data_root"],
            limit=args.limit,
            drop_duplicate_frame_idx=args.drop_duplicate_frame_idx,
        )
        discarded_keyframes = sum(
            len(row.discarded_keyframe_ns)
            for row in read_frame_map(
                map_csv, drop_duplicate_frame_idx=args.drop_duplicate_frame_idx
            )
        )
        recovered_final = output.exists()
        if recovered_final:
            published = [FrameRef.model_validate(value) for value in iter_jsonl(output)]
            if published != refs:
                raise ValueError(
                    "Published frame manifest differs from deterministic reconstruction"
                )
        else:
            write_jsonl_atomic(output, refs)
        counters = {"frames": len(refs), "discarded_duplicate_keyframes": discarded_keyframes}
        if recovered_final:
            counters["recovered_final"] = 1
        manifest = complete_manifest(
            manifest,
            counters=counters,
            shard=video_id,
            output_paths=[("frame_manifest", output)],
        )
        write_manifest(manifest_path, manifest)
    except BaseException as error:
        if not isinstance(error, KeyboardInterrupt | SystemExit):
            write_manifest(manifest_path, fail_manifest(manifest, error))
        raise
    print(json.dumps({"status": "completed", **counters, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
