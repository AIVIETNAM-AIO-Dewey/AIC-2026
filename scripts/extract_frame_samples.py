#!/usr/bin/env python3
"""Extract sampled frames from a raw video for offline smoke/adaptive indexing."""

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
from aic2026.common.io import iter_jsonl  # noqa: E402
from aic2026.contracts import FrameSampleRecord  # noqa: E402
from aic2026.frame_extraction import extract_frame_samples, locate_inputs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--search-root", type=Path)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--map-csv", type=Path)
    parser.add_argument("--media-info", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _config_list(config: dict, key: str, default: list[float]) -> list[float]:
    raw = config.get(key, default)
    if not isinstance(raw, list):
        raise ValueError(f"config.{key} must be a list")
    return [float(value) for value in raw]


def _configured_limit(args_limit: int | None, config: dict) -> int | None:
    if args_limit is not None:
        return args_limit
    raw = config.get("limit", 10)
    return None if raw is None else int(raw)


def _search_root(args: argparse.Namespace, config: dict, roots: dict[str, Path]) -> Path | None:
    if args.search_root is not None:
        return args.search_root.expanduser().resolve()
    if "data_root" in roots:
        return roots["data_root"]
    raw = config.get("search_root")
    if raw is None:
        return None
    return Path(raw).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = read_config(args.config)
    seed = resolve_seed(args.seed, config)
    roots = runtime_roots(args, config, required=("output_root",))
    video_id = args.video_id or str(config.get("video_id", "L21_V001"))
    limit = _configured_limit(args.limit, config)
    output_root = roots["output_root"]
    search_root = _search_root(args, config, roots)
    inputs = locate_inputs(
        video_id=video_id,
        search_root=search_root,
        video_path=args.video_path,
        map_csv=args.map_csv,
        media_info=args.media_info,
    )
    frames_dir = (
        args.frames_dir.expanduser().resolve()
        if args.frames_dir
        else output_root / "frame_extraction" / "keyframes" / video_id
    )
    output = (
        args.output.expanduser().resolve()
        if args.output
        else output_root / "frame_extraction" / "manifests" / f"{video_id}.jsonl"
    )
    manifest_path = output.with_suffix(".manifest.json")
    jpeg_quality = int(config.get("jpeg_quality", 2))
    fallback_timestamps = _config_list(config, "fallback_timestamps_s", [0, 1, 2, 3, 4])
    resolved = {
        "schema_version": config.get("schema_version", "1.0"),
        "video_id": video_id,
        "limit": limit,
        "jpeg_quality": jpeg_quality,
        "fallback_timestamps_s": fallback_timestamps,
        "frames_dir": str(frames_dir),
        "output": str(output),
        "search_root": str(search_root) if search_root is not None else None,
    }
    run_id = str(config.get("run", {}).get("run_id", "frame-extraction-v1"))
    input_paths = [("video", inputs.video_path)]
    if inputs.map_csv is not None:
        input_paths.append(("map_csv", inputs.map_csv))
    if inputs.media_info is not None:
        input_paths.append(("media_info", inputs.media_info))
    manifest = create_manifest(
        run_id=run_id,
        stage="frame_extraction",
        config=resolved,
        seed=seed,
        input_paths=input_paths,
        repo_root=REPO_ROOT,
    )
    manifest, complete = prepare_resume(
        manifest_path=manifest_path,
        output_path=output,
        proposed=manifest,
        resume=args.resume,
    )
    if complete:
        records = [FrameSampleRecord.model_validate(value) for value in iter_jsonl(output)]
        print(
            json.dumps(
                {"status": "already_complete", "frames": len(records), "output": str(output)}
            )
        )
        return 0
    write_manifest(manifest_path, manifest)
    try:
        recovered_final = output.exists()
        if recovered_final:
            records = [FrameSampleRecord.model_validate(value) for value in iter_jsonl(output)]
        else:
            records = extract_frame_samples(
                video_id=video_id,
                video_path=inputs.video_path,
                output_root=output_root,
                frames_dir=frames_dir,
                map_csv=inputs.map_csv,
                limit=limit,
                fallback_timestamps_s=fallback_timestamps,
                jpeg_quality=jpeg_quality,
            )
            write_jsonl_atomic(output, records)
        counters = {"frames": len(records)}
        if inputs.map_csv is None:
            counters["fallback"] = 1
        if recovered_final:
            counters["recovered_final"] = 1
        manifest = complete_manifest(
            manifest,
            counters=counters,
            shard=video_id,
            output_paths=[("frame_samples", output), ("frames_dir", frames_dir)],
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
