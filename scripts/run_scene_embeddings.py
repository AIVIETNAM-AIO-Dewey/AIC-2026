#!/usr/bin/env python3
"""Stage 4: SigLIP2 Dense Visual Scene Embedding.

Embeds video keyframes using SigLIP2 vision encoder, producing 768-dimensional
unit L2-normalized vectors saved into compact `.safetensors` / `.npy` matrices
along with positional JSONL index manifests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aic2026.common.io import iter_jsonl  # noqa: E402
from aic2026.common.manifest import (  # noqa: E402
    complete_manifest,
    create_manifest,
    prepare_resume,
    write_manifest,
)
from aic2026.contracts import FrameRef  # noqa: E402
from aic2026.scene_embedding import (  # noqa: E402
    SiglipEncoder,
    embed_frames,
    matrix_path_for,
    validate_embedding_stage_inputs,
    validate_published_embeddings,
)
from _common import read_config, resolve_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True, help="Video identifier (e.g. L21_V001)")
    parser.add_argument("--frame-manifest", type=Path, required=True, help="Path to frame manifest JSONL")
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory containing keyframe images")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL index path")
    parser.add_argument("--config", type=Path, help="Path to YAML configuration")
    parser.add_argument("--device", default="auto", help="Execution device (auto, cuda, cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--limit", type=int, help="Optional frame limit for smoke testing")
    parser.add_argument("--matrix-format", default="safetensors", choices=["safetensors", "npy"], help="Matrix storage format")
    parser.add_argument("--no-resume", action="store_true", help="Force overwrite existing embedding artifact")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_id = args.video_id
    manifest_path = args.frame_manifest.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_index = args.output.expanduser().resolve()
    meta_path = output_index.with_suffix(".manifest.json")

    config = read_config(args.config)
    seed = int(config.get("seed", 2026))
    device = resolve_device(args.device, config)
    run_id = str(config.get("run", {}).get("run_id", "scene-embedding-v1"))
    model_id = str(config.get("siglip_model_id", "google/siglip2-base-patch16-224"))
    revision = str(config.get("siglip_revision", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"))
    compute_dtype = str(config.get("compute_dtype", "float32"))
    matrix_dtype = str(config.get("matrix_dtype", "float16"))
    batch_size = int(args.batch_size or config.get("batch_size", 32))

    matrix_path = matrix_path_for(output_index, dtype=matrix_dtype, format=args.matrix_format)

    resolved_config = {
        "schema_version": config.get("schema_version", "1.0"),
        "video_id": video_id,
        "device": device,
        "model_id": model_id,
        "revision": revision,
        "compute_dtype": compute_dtype,
        "matrix_dtype": matrix_dtype,
        "batch_size": batch_size,
        "matrix_format": args.matrix_format,
        "limit": args.limit,
    }

    manifest = create_manifest(
        run_id=run_id,
        stage="scene_embeddings",
        config=resolved_config,
        seed=seed,
        input_paths=[("frame_manifest", manifest_path)],
        models=[{"model_id": model_id, "revision": revision, "license": "Apache-2.0"}],
        repo_root=REPO_ROOT,
    )
    manifest, complete = prepare_resume(
        manifest_path=meta_path,
        output_path=output_index,
        proposed=manifest,
        resume=not args.no_resume,
    )
    if complete and matrix_path.is_file():
        records = list(iter_jsonl(output_index))
        print(
            json.dumps(
                {
                    "status": "already_complete",
                    "video_id": video_id,
                    "frames": len(records),
                    "index": str(output_index),
                    "matrix": str(matrix_path),
                }
            )
        )
        return 0
    write_manifest(meta_path, manifest)

    # 1. Validate Input Manifest
    validate_embedding_stage_inputs(
        frame_manifest=manifest_path,
        data_root=data_root,
        video_id=video_id,
        limit=args.limit,
    )

    # 2. Load SigLIP2 Model
    print(f"[siglip2] Loading {model_id} on {device} ...", file=sys.stderr, flush=True)
    encoder = SiglipEncoder.from_pretrained(
        model_id=model_id,
        revision=revision,
        device=device,
        compute_dtype=compute_dtype,
    )

    # 3. Batch Embed Keyframes
    start_time = time.time()
    counters = embed_frames(
        frame_manifest=manifest_path,
        data_root=data_root,
        output_index=output_index,
        output_matrix=matrix_path,
        run_id=run_id,
        backend=encoder,
        matrix_dtype=matrix_dtype,
        batch_size=batch_size,
        limit=args.limit,
    )
    elapsed = time.time() - start_time

    # 4. Finalize Manifest
    manifest = complete_manifest(
        manifest,
        counters={**counters, "elapsed_s": round(elapsed, 2)},
        shard=video_id,
        output_paths=[("embedding_index", output_index), ("embedding_matrix", matrix_path)],
    )
    write_manifest(meta_path, manifest)

    print(
        json.dumps(
            {
                "status": "completed",
                "video_id": video_id,
                "frames": counters["frames"],
                "batches": counters["batches"],
                "embedding_dim": counters["embedding_dim"],
                "elapsed_s": round(elapsed, 2),
                "index": str(output_index),
                "matrix": str(matrix_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
