#!/usr/bin/env python3
"""Generate deterministic English DAM captions for persisted region masks."""

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
from aic2026.contracts import ObjectFrameRecord  # noqa: E402
from aic2026.object_description import (  # noqa: E402
    DAM_PROMPT,
    DamCaptioner,
    run_descriptions,
    validate_description_settings,
    validate_description_stage_inputs,
    validate_published_object_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--mask-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path)
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
    if not device.startswith("cuda"):
        raise SystemExit("DAM-3B requires --device cuda; CPU fallback is not supported")
    mask_artifact = args.mask_artifact.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else roots["output_root"] / "object_description" / "descriptions" / f"{video_id}.jsonl"
    )
    manifest_path = output.with_suffix(".manifest.json")
    run_id = str(config.get("run", {}).get("run_id", "object-description-v1"))
    model_id = str(config.get("dam_model_id", "nvidia/DAM-3B"))
    revision = str(config.get("dam_revision", "0797bedd98d645cd021379a4661ee233da279bba"))
    code_revision = str(config.get("dam_code_revision", "153ad3d33c29324e9197f565547c6bc8500da02d"))
    max_words = int(config.get("max_words", 20))
    max_new_tokens = int(config.get("max_new_tokens", 48))
    oom_retry_tokens = int(config.get("oom_retry_max_new_tokens", 32))
    maximum_consecutive_oom = int(config.get("maximum_consecutive_oom", 3))
    maximum_consecutive_errors = int(config.get("maximum_consecutive_region_errors", 3))
    validate_description_settings(
        maximum_words=max_words,
        max_new_tokens=max_new_tokens,
        oom_retry_max_new_tokens=oom_retry_tokens,
        maximum_consecutive_oom=maximum_consecutive_oom,
        maximum_consecutive_errors=maximum_consecutive_errors,
    )
    resolved = {
        "schema_version": config.get("schema_version", "1.0"),
        "video_id": video_id,
        "seed": seed,
        "device": device,
        "limit": args.limit,
        "dam_model_id": model_id,
        "dam_revision": revision,
        "dam_code_revision": code_revision,
        "max_words": max_words,
        "max_new_tokens": max_new_tokens,
        "oom_retry_max_new_tokens": oom_retry_tokens,
        "maximum_consecutive_oom": maximum_consecutive_oom,
        "maximum_consecutive_region_errors": maximum_consecutive_errors,
        "prompt_mode": "full+focal_crop",
        "prompt": DAM_PROMPT,
        "temperature": 0,
        "num_beams": 1,
    }
    input_paths = [("mask_artifact", mask_artifact)]
    mask_manifest_sidecar = mask_artifact.with_suffix(".manifest.json")
    if mask_manifest_sidecar.exists():
        input_paths.append(("mask_artifact_run", mask_manifest_sidecar))
    manifest = create_manifest(
        run_id=run_id,
        stage="dam_descriptions",
        config=resolved,
        seed=seed,
        input_paths=input_paths,
        models=[
            {
                "model_id": model_id,
                "revision": revision,
                "license": "NVIDIA Noncommercial License",
            },
            {
                "model_id": "github.com/NVlabs/describe-anything",
                "revision": code_revision,
                "license": "Apache-2.0",
            },
        ],
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
    cache_dir = roots["cache_root"] / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TORCH_HOME", str(roots["cache_root"] / "torch"))
    try:
        if mask_manifest_sidecar.exists():
            upstream_manifest = validate_upstream_manifest(
                manifest_path=mask_manifest_sidecar,
                artifact_path=mask_artifact,
                expected_stage="sam_masks",
            )
            if upstream_manifest.run_id != run_id:
                raise ValueError("Mask-stage run_id does not match DAM-stage run_id")
        validate_description_stage_inputs(
            mask_artifact=mask_artifact,
            data_root=roots["data_root"],
            video_id=video_id,
            limit=args.limit,
            expected_run_id=run_id,
        )
        seed_everything(seed)
        if output.exists():
            expected_raw = iter_jsonl(mask_artifact)
            if args.limit is not None:
                expected_raw = islice(expected_raw, args.limit)
            expected_frame_uids = [
                ObjectFrameRecord.model_validate(raw).frame_uid for raw in expected_raw
            ]
            summary = validate_published_object_artifact(
                artifact=output,
                data_root=roots["data_root"],
                video_id=video_id,
                expected_frame_uids=expected_frame_uids,
                caption_mode="finished",
                expected_run_id=run_id,
            )
            counters = {
                "frames": summary["frames"],
                "captions_ok": summary["captions_ok"],
                "captions_error": summary["captions_error"],
                "captions_oom": summary["captions_oom"],
                "oom_retries": 0,
                "oom_retries_unknown_after_recovery": 1,
                "recovered_final": 1,
            }
            manifest = complete_manifest(
                manifest,
                counters=counters,
                shard=video_id,
                output_paths=[("description_artifact", output)],
            )
            write_manifest(manifest_path, manifest)
            print(json.dumps({"status": "recovered", "output": str(output), **counters}))
            return 0
        backend = DamCaptioner.from_pretrained(
            model_id=model_id,
            revision=revision,
            code_revision=code_revision,
            cache_dir=cache_dir,
        )
        counters = run_descriptions(
            mask_artifact=mask_artifact,
            data_root=roots["data_root"],
            output=output,
            caption_backend=backend,
            resume=args.resume,
            limit=args.limit,
            max_new_tokens=max_new_tokens,
            oom_retry_max_new_tokens=oom_retry_tokens,
            maximum_words=max_words,
            maximum_consecutive_oom=maximum_consecutive_oom,
            maximum_consecutive_errors=maximum_consecutive_errors,
        )
        manifest = complete_manifest(
            manifest,
            counters=counters,
            shard=video_id,
            output_paths=[("description_artifact", output)],
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
