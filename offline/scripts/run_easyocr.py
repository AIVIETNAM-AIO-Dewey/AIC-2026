#!/usr/bin/env python3
"""Run EasyOCR on the shared canonical keyframe manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _common import (  # noqa: E402
    add_common_arguments,
    read_config,
    resolve_device,
    resolve_seed,
    runtime_roots,
    seed_everything,
)
from aic2026.common import (  # noqa: E402
    complete_manifest,
    create_manifest,
    fail_manifest,
    prepare_resume,
    require_prepared_video,
    write_manifest,
)
from aic2026.ocr import EasyOcrReader, extract_ocr_frames  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = read_config(args.config or REPO_ROOT / "configs" / "offline" / "ocr.yaml")
    roots = runtime_roots(args, config)
    video_id = args.video_id or config.get("video_id")
    if not video_id:
        raise SystemExit("--video-id is required")
    require_prepared_video(roots["data_root"], video_id)
    seed = resolve_seed(args.seed, config)
    device = resolve_device(args.device, config)
    run_id = str(config.get("run", {}).get("run_id", "easyocr-v1"))
    revision = str(config.get("model_revision", "easyocr-1.7.2"))
    frame_manifest = args.frame_manifest.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else roots["output_root"] / "ocr" / f"{video_id}.jsonl"
    )
    manifest_path = output.with_suffix(".manifest.json")
    manifest = create_manifest(
        run_id=run_id,
        stage="ocr",
        config={
            "video_id": video_id,
            "device": device,
            "languages": config.get("languages", ["vi", "en"]),
            "model_revision": revision,
            "limit": args.limit,
        },
        seed=seed,
        input_paths=[("frame_manifest", frame_manifest)],
        models=[{"model_id": "EasyOCR", "revision": revision, "license": "Apache-2.0"}],
        repo_root=REPO_ROOT,
    )
    manifest, complete = prepare_resume(
        manifest_path=manifest_path,
        output_path=output,
        proposed=manifest,
        resume=args.resume,
    )
    if complete:
        print(json.dumps({"status": "already_complete", "output": str(output)}))
        return 0
    write_manifest(manifest_path, manifest)
    try:
        seed_everything(seed)
        reader = EasyOcrReader.create(gpu=device.startswith("cuda"))
        counters = extract_ocr_frames(
            frame_manifest=frame_manifest,
            data_root=roots["data_root"],
            output=output,
            run_id=run_id,
            reader=reader,
            limit=args.limit,
            resume=args.resume,
        )
        manifest = complete_manifest(
            manifest,
            counters=counters,
            shard=video_id,
            output_paths=[("ocr", output)],
        )
        write_manifest(manifest_path, manifest)
    except BaseException as error:
        if not isinstance(error, KeyboardInterrupt | SystemExit):
            write_manifest(manifest_path, fail_manifest(manifest, error))
        raise
    print(json.dumps({"status": "completed", "output": str(output), **counters}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
