#!/usr/bin/env python3
"""Convert external TransNetV2 scenes into versioned shot records."""

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
from aic2026.contracts import ShotRecord  # noqa: E402
from aic2026.frame_extraction.discovery import locate_inputs  # noqa: E402
from aic2026.frame_extraction.ffmpeg import probe_video  # noqa: E402
from aic2026.frame_extraction.transnetv2 import (  # noqa: E402
    build_shot_records,
    parse_scenes_txt,
    run_transnetv2_inference,
    run_transnetv2_pytorch_inference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--search-root", type=Path)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--scenes-file", type=Path)
    parser.add_argument("--entrypoint", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--backend", choices=("pytorch", "tensorflow"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float)
    return parser


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
    backend = args.backend or str(config.get("shot_detection", {}).get("backend", "pytorch"))
    roots = runtime_roots(args, config, required=("output_root",))
    output_root = roots["output_root"]
    video_id = args.video_id or str(config.get("video_id", "L21_V001"))
    inputs = locate_inputs(
        video_id=video_id,
        search_root=_search_root(args, config, roots),
        video_path=args.video_path,
    )
    print(f"[shot_detection] video_id={video_id}", file=sys.stderr, flush=True)
    print(f"[shot_detection] video={inputs.video_path}", file=sys.stderr, flush=True)
    output = (
        args.output.expanduser().resolve()
        if args.output
        else output_root / "shot_detection" / f"{video_id}.jsonl"
    )
    manifest_path = output.with_suffix(".manifest.json")
    run_id = str(config.get("run", {}).get("run_id", "transnetv2-shot-detection-v1"))
    input_paths = [("video", inputs.video_path)]
    if args.scenes_file is not None:
        input_paths.append(("scenes", args.scenes_file.expanduser().resolve()))
    if args.entrypoint is not None and args.entrypoint.exists():
        input_paths.append(("transnetv2_entrypoint", args.entrypoint.expanduser().resolve()))
    if args.weights is not None and args.weights.exists():
        input_paths.append(("transnetv2_weights", args.weights.expanduser().resolve()))
    resolved = {
        "schema_version": config.get("schema_version", "1.0"),
        "video_id": video_id,
        "scenes_file": str(args.scenes_file) if args.scenes_file else None,
        "entrypoint": str(args.entrypoint) if args.entrypoint else None,
        "weights": str(args.weights) if args.weights else None,
        "backend": backend,
        "output": str(output),
    }
    manifest = create_manifest(
        run_id=run_id,
        stage="shot_detection",
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
        records = [ShotRecord.model_validate(value) for value in iter_jsonl(output)]
        print(
            json.dumps(
                {"status": "already_complete", "shots": len(records), "output": str(output)}
            )
        )
        return 0
    write_manifest(manifest_path, manifest)
    try:
        recovered_final = output.exists()
        if recovered_final:
            records = [ShotRecord.model_validate(value) for value in iter_jsonl(output)]
            scenes_path = args.scenes_file
        else:
            if args.scenes_file is None:
                if args.entrypoint is None:
                    raise ValueError("--entrypoint is required when --scenes-file is not provided")
                print(
                    f"[shot_detection] running TransNetV2 backend={backend} entrypoint={args.entrypoint}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.weights is not None:
                    print(
                        f"[shot_detection] weights={args.weights}",
                        file=sys.stderr,
                        flush=True,
                    )
                work_dir = output_root / "shot_detection" / "transnetv2_work" / video_id
                if backend == "pytorch":
                    if args.weights is None:
                        raise ValueError("--weights is required for the PyTorch backend")
                    scenes_path = run_transnetv2_pytorch_inference(
                        video_path=inputs.video_path,
                        model_module=args.entrypoint,
                        weights=args.weights,
                        work_dir=work_dir,
                    )
                else:
                    scenes_path = run_transnetv2_inference(
                        video_path=inputs.video_path,
                        entrypoint=args.entrypoint,
                        weights=args.weights,
                        work_dir=work_dir,
                    )
            else:
                scenes_path = args.scenes_file.expanduser().resolve()
                print(
                    f"[shot_detection] reading scenes_file={scenes_path}",
                    file=sys.stderr,
                    flush=True,
                )
            fps = args.fps if args.fps is not None else probe_video(inputs.video_path).fps
            print(f"[shot_detection] fps={fps:.6f}", file=sys.stderr, flush=True)
            scenes = parse_scenes_txt(scenes_path)
            print(f"[shot_detection] parsed_scenes={len(scenes)}", file=sys.stderr, flush=True)
            records = build_shot_records(
                video_id=video_id,
                scenes=scenes,
                fps=fps,
                source_video=inputs.video_path,
            )
            write_jsonl_atomic(output, records)
            print(f"[shot_detection] wrote_records={output}", file=sys.stderr, flush=True)
        counters = {"shots": len(records)}
        if recovered_final:
            counters["recovered_final"] = 1
        output_paths = [("shot_records", output)]
        if scenes_path is not None and Path(scenes_path).exists():
            output_paths.append(("scenes", Path(scenes_path)))
        manifest = complete_manifest(
            manifest,
            counters=counters,
            shard=video_id,
            output_paths=output_paths,
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
