#!/usr/bin/env python3
"""Filter organizer bboxes and convert them to SAM masks."""

from __future__ import annotations

import argparse
import json
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
    validate_upstream_manifest,
    write_manifest,
)
from aic2026.contracts import FrameRef  # noqa: E402
from aic2026.object_description import (  # noqa: E402
    BboxMaskGenerator,
    FilterConfig,
    SamMaskGenerator,
    YoloWorldDetector,
    prepare_masks,
    validate_mask_stage_inputs,
    validate_published_object_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--objects-dir", type=Path, default=None, help="Directory containing organizer JSON files")
    parser.add_argument("--detector", choices=["yolo-world", "organizer"], default=None, help="Object detector backend")
    parser.add_argument("--detector-model", type=str, default=None, help="YOLO-World checkpoint or path (default: yolov8x-worldv2.pt)")
    parser.add_argument("--detector-conf", type=float, default=None, help="Detector confidence threshold")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--use-bbox",
        action="store_true",
        help="Use fast direct bounding box masks instead of loading SAM (zero GPU overhead)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = read_config(args.config)
    seed = resolve_seed(args.seed, config)
    roots = runtime_roots(args, config)
    video_id = args.video_id or config.get("video_id")
    if not video_id:
        raise SystemExit("--video-id is required")
    frame_manifest = args.frame_manifest.expanduser().resolve()
    objects_dir = args.objects_dir.expanduser().resolve() if args.objects_dir else None
    detector_type = args.detector or config.get("detector", "yolo-world")
    detector_model = args.detector_model or config.get("detector_model", "yolov8x-worldv2.pt")
    detector_conf = args.detector_conf if args.detector_conf is not None else float(config.get("detector_conf", 0.20))
    if detector_type == "organizer" and objects_dir is None:
        raise SystemExit("--objects-dir is required when using organizer detector")
    output = (
        args.output.expanduser().resolve()
        if args.output
        else roots["output_root"] / "object_description" / "masks" / f"{video_id}.jsonl"
    )
    manifest_path = output.with_suffix(".manifest.json")
    run_id = str(config.get("run", {}).get("run_id", "object-description-v1"))
    device = resolve_device(args.device, config)
    filter_config = FilterConfig(
        minimum_score=float(config.get("score_threshold", 0.10)),
        minimum_area_ratio=float(config.get("min_area_ratio", 0.001)),
        maximum_area_ratio=float(config.get("max_area_ratio", 0.90)),
        same_class_iou=float(config.get("class_nms_iou", 0.70)),
        cross_label_duplicate_iou=float(config.get("cross_label_iou", 0.95)),
        maximum_regions=int(config.get("max_regions", 20)),
        fallback_scene=bool(config.get("fallback_scene", True)),
    )
    sam_id = str(config.get("sam_model_id", "facebook/sam-vit-base"))
    sam_revision = str(config.get("sam_revision", "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f"))
    resolved = {
        "schema_version": config.get("schema_version", "1.0"),
        "video_id": video_id,
        "seed": seed,
        "device": device,
        "limit": args.limit,
        "detector": detector_type,
        "detector_model": detector_model if detector_type == "yolo-world" else None,
        "detector_conf": detector_conf if detector_type == "yolo-world" else None,
        "region_filter": {
            "minimum_score": filter_config.minimum_score,
            "minimum_area_ratio": filter_config.minimum_area_ratio,
            "maximum_area_ratio": filter_config.maximum_area_ratio,
            "same_class_iou": filter_config.same_class_iou,
            "cross_label_duplicate_iou": filter_config.cross_label_duplicate_iou,
            "maximum_regions": filter_config.maximum_regions,
            "fallback_scene": filter_config.fallback_scene,
        },
        "sam_model_id": sam_id,
        "sam_revision": sam_revision,
    }
    input_paths = [("frame_manifest", frame_manifest)]
    if objects_dir is not None:
        input_paths.append(("objects", objects_dir))
    frame_manifest_sidecar = frame_manifest.with_suffix(".manifest.json")
    if frame_manifest_sidecar.exists():
        input_paths.append(("frame_manifest_run", frame_manifest_sidecar))
    
    models = [{"model_id": sam_id, "revision": sam_revision, "license": "Apache-2.0"}]
    if detector_type == "yolo-world":
        models.append({"model_id": detector_model, "revision": "latest", "license": "AGPL-3.0"})

    manifest = create_manifest(
        run_id=run_id,
        stage="sam_masks",
        config=resolved,
        seed=seed,
        input_paths=input_paths,
        models=models,
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
        if frame_manifest_sidecar.exists():
            upstream_manifest = validate_upstream_manifest(
                manifest_path=frame_manifest_sidecar,
                artifact_path=frame_manifest,
                expected_stage="frame_manifest",
            )
            if upstream_manifest.run_id != run_id:
                raise ValueError("Frame-manifest run_id does not match mask-stage run_id")
        validate_mask_stage_inputs(
            frame_manifest=frame_manifest,
            objects_dir=objects_dir,
            data_root=roots["data_root"],
            video_id=video_id,
            limit=args.limit,
        )
        seed_everything(seed)
        if output.exists():
            expected_raw = iter_jsonl(frame_manifest)
            if args.limit is not None:
                expected_raw = islice(expected_raw, args.limit)
            expected_frame_uids = [FrameRef.model_validate(raw).frame_uid for raw in expected_raw]
            summary = validate_published_object_artifact(
                artifact=output,
                data_root=roots["data_root"],
                video_id=video_id,
                expected_frame_uids=expected_frame_uids,
                caption_mode="pending",
                expected_run_id=run_id,
            )
            counters = {
                "frames": summary["frames"],
                "regions": summary["regions"],
                "bbox_fallbacks": summary["bbox_fallbacks"],
                "recovered_final": 1,
            }
            manifest = complete_manifest(
                manifest,
                counters=counters,
                shard=video_id,
                output_paths=[("mask_artifact", output)],
            )
            write_manifest(manifest_path, manifest)
            print(json.dumps({"status": "recovered", "output": str(output), **counters}))
            return 0
        use_bbox = args.use_bbox or str(config.get("mask_backend", "")).lower() == "bbox"
        if use_bbox:
            backend = BboxMaskGenerator()
        else:
            backend = SamMaskGenerator.from_pretrained(
                model_id=sam_id,
                revision=sam_revision,
                cache_dir=roots["cache_root"] / "huggingface",
                device=device,
            )
        detector_instance = (
            YoloWorldDetector(
                model_id_or_path=detector_model,
                device=device,
                conf=detector_conf,
            )
            if detector_type == "yolo-world"
            else None
        )
        counters = prepare_masks(
            frame_manifest=frame_manifest,
            data_root=roots["data_root"],
            output=output,
            run_id=run_id,
            mask_backend=backend,
            objects_dir=objects_dir,
            detector=detector_instance,
            filter_config=filter_config,
            resume=args.resume,
            limit=args.limit,
        )
        manifest = complete_manifest(
            manifest,
            counters=counters,
            shard=video_id,
            output_paths=[("mask_artifact", output)],
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
