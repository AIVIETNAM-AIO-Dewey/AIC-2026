#!/usr/bin/env python3
"""Decode 5 FPS TRAKE frames by PTS and generate pinned SigLIP2 embeddings."""

from __future__ import annotations

import argparse
import json
import os
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
    write_jsonl_atomic,
    write_manifest,
)
from aic2026.scene_embedding import SiglipEncoder, embed_frames, matrix_path_for  # noqa: E402
from aic2026.scene_embedding.dense_frames import decode_dense_frames  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = read_config(args.config or REPO_ROOT / "configs" / "offline" / "dense_frames.yaml")
    roots = runtime_roots(args, config)
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"Video does not exist: {video}")
    video_id = args.video_id or video.stem
    require_prepared_video(roots["data_root"], video_id)

    seed = resolve_seed(args.seed, config)
    device = resolve_device(args.device, config)
    run_id = str(config.get("run", {}).get("run_id", "dense-frames-5fps-v1"))
    sampling_fps = float(config.get("sampling_fps", 5.0))
    jpeg_quality = int(config.get("thumbnail_jpeg_quality", 90))
    batch_size = int(args.batch_size or config.get("batch_size", 32))
    model_id = str(config.get("siglip_model_id", "google/siglip2-base-patch16-224"))
    revision = str(config.get("siglip_revision", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"))
    output_index = roots["output_root"] / "dense_scene_embeddings" / f"{video_id}.jsonl"
    output_matrix = matrix_path_for(output_index, "float16")
    frame_manifest = roots["output_root"] / "dense_frame_manifests" / f"{video_id}.jsonl"
    manifest_path = output_index.with_suffix(".manifest.json")
    resolved = {
        "video_id": video_id,
        "sampling_fps": sampling_fps,
        "thumbnail_jpeg_quality": jpeg_quality,
        "siglip_model_id": model_id,
        "siglip_revision": revision,
        "batch_size": batch_size,
        "device": device,
        "limit": args.limit,
    }
    manifest = create_manifest(
        run_id=run_id,
        stage="dense_scene_embeddings",
        config=resolved,
        seed=seed,
        input_paths=[("video", video)],
        models=[{"model_id": model_id, "revision": revision, "license": "Apache-2.0"}],
        repo_root=REPO_ROOT,
    )
    manifest, complete = prepare_resume(
        manifest_path=manifest_path,
        output_path=output_index,
        proposed=manifest,
        resume=args.resume,
    )
    if complete:
        print(json.dumps({"status": "already_complete", "output": str(output_index)}))
        return 0
    write_manifest(manifest_path, manifest)
    try:
        refs = decode_dense_frames(
            video_path=video,
            video_id=video_id,
            output_root=roots["output_root"],
            sampling_fps=sampling_fps,
            jpeg_quality=jpeg_quality,
            limit=args.limit,
        )
        write_jsonl_atomic(frame_manifest, refs)
        seed_everything(seed)
        cache_dir = roots["cache_root"] / "huggingface"
        os.environ.setdefault("HF_HOME", str(cache_dir))
        backend = SiglipEncoder.from_pretrained(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            device=device,
            compute_dtype=str(config.get("compute_dtype", "float32")),
        )
        counters = embed_frames(
            frame_manifest=frame_manifest,
            data_root=roots["output_root"],
            output_index=output_index,
            output_matrix=output_matrix,
            run_id=run_id,
            backend=backend,
            matrix_dtype="float16",
            batch_size=batch_size,
        )
        manifest = complete_manifest(
            manifest,
            counters={**counters, "sampling_fps": int(sampling_fps)},
            shard=video_id,
            output_paths=[
                ("dense_frame_manifest", frame_manifest),
                ("embedding_index", output_index),
                ("embedding_matrix", output_matrix),
            ],
        )
        write_manifest(manifest_path, manifest)
    except BaseException as error:
        if not isinstance(error, KeyboardInterrupt | SystemExit):
            write_manifest(manifest_path, fail_manifest(manifest, error))
        raise
    print(json.dumps({"status": "completed", "output": str(output_index), **counters}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
