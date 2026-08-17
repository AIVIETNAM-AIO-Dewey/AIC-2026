#!/usr/bin/env python3
"""Embed keyframes with SigLIP2 into a per-video matrix plus a JSONL index."""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import islice
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
    iter_jsonl,
    prepare_resume,
    require_prepared_video,
    validate_upstream_manifest,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Index JSONL path; matrix is derived from it.")
    parser.add_argument("--batch-size", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = read_config(args.config)
    seed = resolve_seed(args.seed, config)
    roots = runtime_roots(args, config)
    video_id = args.video_id or config.get("video_id")
    if not video_id:
        raise SystemExit("--video-id is required")
    require_prepared_video(roots["data_root"], video_id)
    device = resolve_device(args.device, config)
    if device == "cpu":
        print(
            "warning: running SigLIP2 on cpu; pass --device mps on Apple Silicon",
            file=sys.stderr,
        )
    frame_manifest = args.frame_manifest.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else roots["output_root"] / "scene_embeddings" / f"{video_id}.jsonl"
    )
    manifest_path = output.with_suffix(".manifest.json")
    run_id = str(config.get("run", {}).get("run_id", "scene-embedding-v1"))
    model_id = str(config.get("siglip_model_id", "google/siglip2-base-patch16-224"))
    revision = str(config.get("siglip_revision", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"))
    compute_dtype = str(config.get("compute_dtype", "float32"))
    matrix_dtype = str(config.get("matrix_dtype", "float16"))
    batch_size = int(args.batch_size or config.get("batch_size", 32))
    matrix_path = matrix_path_for(output, matrix_dtype)

    resolved = {
        "schema_version": config.get("schema_version", "1.0"),
        "video_id": video_id,
        "seed": seed,
        "device": device,
        "limit": args.limit,
        "siglip_model_id": model_id,
        "siglip_revision": revision,
        "compute_dtype": compute_dtype,
        "matrix_dtype": matrix_dtype,
        "batch_size": batch_size,
        "l2_normalized": True,
        "image_processor": "fast",
    }
    input_paths = [("frame_manifest", frame_manifest)]
    frame_manifest_sidecar = frame_manifest.with_suffix(".manifest.json")
    if frame_manifest_sidecar.exists():
        input_paths.append(("frame_manifest_run", frame_manifest_sidecar))
    manifest = create_manifest(
        run_id=run_id,
        stage="scene_embeddings",
        config=resolved,
        seed=seed,
        input_paths=input_paths,
        models=[{"model_id": model_id, "revision": revision, "license": "Apache-2.0"}],
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
    # prepare_resume only guards the index; an orphan matrix means a crash between the
    # two publishes and must not be silently reused.
    if matrix_path.exists() and not output.exists():
        raise SystemExit(f"Embedding matrix exists without its index: {matrix_path}")
    write_manifest(manifest_path, manifest)
    cache_dir = roots["cache_root"] / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_dir))
    try:
        if frame_manifest_sidecar.exists():
            upstream_manifest = validate_upstream_manifest(
                manifest_path=frame_manifest_sidecar,
                artifact_path=frame_manifest,
                expected_stage="frame_manifest",
            )
            if upstream_manifest.run_id != run_id:
                raise ValueError("Frame-manifest run_id does not match embedding-stage run_id")
        validate_embedding_stage_inputs(
            frame_manifest=frame_manifest,
            data_root=roots["data_root"],
            video_id=video_id,
            limit=args.limit,
        )
        seed_everything(seed)
        expected_raw = iter_jsonl(frame_manifest)
        if args.limit is not None:
            expected_raw = islice(expected_raw, args.limit)
        expected_frame_uids = [FrameRef.model_validate(raw).frame_uid for raw in expected_raw]

        if output.exists():
            summary = validate_published_embeddings(
                index_path=output,
                matrix_path=matrix_path,
                video_id=video_id,
                expected_frame_uids=expected_frame_uids,
                expected_run_id=run_id,
            )
            counters = {**summary, "recovered_final": 1}
            manifest = complete_manifest(
                manifest,
                counters=counters,
                shard=video_id,
                output_paths=[("embedding_index", output), ("embedding_matrix", matrix_path)],
            )
            write_manifest(manifest_path, manifest)
            print(json.dumps({"status": "recovered", "output": str(output), **counters}))
            return 0

        backend = SiglipEncoder.from_pretrained(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            device=device,
            compute_dtype=compute_dtype,
        )
        counters = embed_frames(
            frame_manifest=frame_manifest,
            data_root=roots["data_root"],
            output_index=output,
            output_matrix=matrix_path,
            run_id=run_id,
            backend=backend,
            matrix_dtype=matrix_dtype,
            batch_size=batch_size,
            limit=args.limit,
        )
        validate_published_embeddings(
            index_path=output,
            matrix_path=matrix_path,
            video_id=video_id,
            expected_frame_uids=expected_frame_uids,
            expected_run_id=run_id,
        )
        manifest = complete_manifest(
            manifest,
            counters=counters,
            shard=video_id,
            output_paths=[("embedding_index", output), ("embedding_matrix", matrix_path)],
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
