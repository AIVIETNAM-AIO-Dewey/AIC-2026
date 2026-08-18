#!/usr/bin/env python3
"""Run pinned offline PP-OCRv6-small on the canonical keyframe manifest."""

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
from aic2026.ocr import PaddleOcrV6Reader, extract_ocr_frames, verify_ppocrv6  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--frame-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify packages and local model checksums without constructing PaddleOCR.",
    )
    args = parser.parse_args(argv)
    config = read_config(args.config or REPO_ROOT / "configs" / "offline" / "ocr_ppocrv6.yaml")
    model = config.get("model", {})
    requested_device = args.device or config.get("device", "cpu")
    if requested_device != model.get("device"):
        raise SystemExit("--device must equal the pinned PP-OCRv6 model device")

    if args.preflight_only:
        roots = runtime_roots(args, config, required=("cache_root",))
        evidence = verify_ppocrv6(config, roots["cache_root"])
        print(json.dumps({"status": "preflight_pass", **evidence}, ensure_ascii=False))
        return 0

    if args.frame_manifest is None:
        raise SystemExit("--frame-manifest is required unless --preflight-only is used")
    roots = runtime_roots(args, config)
    video_id = args.video_id or config.get("video_id")
    if not video_id:
        raise SystemExit("--video-id is required")
    require_prepared_video(roots["data_root"], str(video_id))
    seed = resolve_seed(args.seed, config)
    run_id = str(config.get("run", {}).get("run_id", "ppocrv6-small-v1"))
    revision = str(model.get("revision", ""))
    frame_manifest = args.frame_manifest.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else roots["output_root"] / "ocr" / f"{video_id}.jsonl"
    )
    manifest_path = output.with_suffix(".manifest.json")
    resolved_config = {
        "video_id": str(video_id),
        "device": requested_device,
        "model_id": model.get("id"),
        "model_revision": revision,
        "source_registry_sha256": model.get("source_registry_sha256"),
        "confidence_threshold": model.get("confidence_threshold"),
        "offline_only": True,
        "download_allowed": False,
        "fallback": False,
        "ensemble": False,
        "limit": args.limit,
    }
    manifest = create_manifest(
        run_id=run_id,
        stage="ocr",
        config=resolved_config,
        seed=seed,
        input_paths=[("frame_manifest", frame_manifest)],
        models=[{"model_id": "PP-OCRv6-small", "revision": revision, "license": "Apache-2.0"}],
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
        reader = PaddleOcrV6Reader.create(config=config, cache_root=roots["cache_root"])
        counters = extract_ocr_frames(
            frame_manifest=frame_manifest,
            data_root=roots["data_root"],
            output=output,
            run_id=run_id,
            reader=reader,
            limit=args.limit,
        )
        manifest = complete_manifest(
            manifest,
            counters=counters,
            shard=str(video_id),
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
