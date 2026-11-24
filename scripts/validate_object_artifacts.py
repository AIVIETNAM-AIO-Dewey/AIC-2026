#!/usr/bin/env python3
"""Validate frame/object JSONL contracts and optional run manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _common import add_common_arguments  # noqa: E402

from aic2026.common import iter_jsonl, sha256_path  # noqa: E402
from aic2026.contracts import FrameRef, ObjectFrameRecord, RunManifest  # noqa: E402
from aic2026.object_description.rle import decode_mask  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--require-captions", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report: dict[str, object] = {"schema_version": "aic26.validation.v1", "artifacts": []}
    total_frames = 0
    total_regions = 0
    caption_statuses: dict[str, int] = {}
    global_uids: set[str] = set()
    artifact_paths: list[Path] = []
    artifact_reports: list[dict[str, object]] = []

    for raw_path in args.artifact:
        path = raw_path.expanduser().resolve()
        file_frames = 0
        file_regions = 0
        file_uids: set[str] = set()
        file_run_ids: set[str] = set()
        artifact_type: str | None = None
        for raw in iter_jsonl(path):
            if raw.get("schema_version") == "aic26.object_regions.v1":
                record = ObjectFrameRecord.model_validate(raw)
                current_type = "object_regions"
                file_run_ids.add(record.run_id)
                file_regions += len(record.regions)
                for region in record.regions:
                    mask = decode_mask(region.segmentation.mask_rle.model_dump(mode="json"))
                    if not mask.any():
                        raise ValueError(f"Decoded mask is empty: {region.region_id}")
                    status = region.caption.status
                    caption_statuses[status] = caption_statuses.get(status, 0) + 1
                    if args.require_captions and status != "ok":
                        raise ValueError(
                            f"{path}: {region.region_id} caption status is {status!r}, "
                            "expected 'ok'"
                        )
            else:
                record = FrameRef.model_validate(raw)
                current_type = "frame_manifest"
            if artifact_type is not None and current_type != artifact_type:
                raise ValueError(f"Mixed record types in one artifact: {path}")
            artifact_type = current_type
            if record.frame_uid in file_uids:
                raise ValueError(f"Duplicate frame_uid in {path}: {record.frame_uid}")
            file_uids.add(record.frame_uid)
            file_frames += 1
        if file_frames == 0:
            raise ValueError(f"Artifact is empty: {path}")
        overlap = global_uids.intersection(file_uids)
        if overlap:
            raise ValueError(f"Duplicate frame_uid across artifacts: {sorted(overlap)[0]}")
        global_uids.update(file_uids)
        total_frames += file_frames
        total_regions += file_regions
        artifact_paths.append(path)
        artifact_report: dict[str, object] = {
            "path": str(path),
            "type": artifact_type,
            "frames": file_frames,
            "regions": file_regions,
            "checksum_verified": False,
        }
        if file_run_ids:
            if len(file_run_ids) != 1:
                raise ValueError(f"Object artifact contains multiple run_id values: {path}")
            artifact_report["run_id"] = next(iter(file_run_ids))
        artifact_reports.append(artifact_report)

    if args.manifest and len(args.manifest) != len(artifact_paths):
        raise ValueError("Provide exactly one --manifest per --artifact, in the same order")

    manifests = []
    for index, raw_path in enumerate(args.manifest):
        path = raw_path.expanduser().resolve()
        manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        artifact_path = artifact_paths[index]
        artifact_report = artifact_reports[index]
        if manifest.status != "completed":
            raise ValueError(f"Run manifest is not completed: {path}")
        artifact_type = artifact_report["type"]
        if artifact_type == "frame_manifest" and manifest.stage != "frame_manifest":
            raise ValueError(f"Frame artifact requires a frame_manifest sidecar: {path}")
        if artifact_type == "object_regions":
            allowed_stages = (
                {"dam_descriptions"}
                if args.require_captions
                else {
                    "sam_masks",
                    "dam_descriptions",
                }
            )
            if manifest.stage not in allowed_stages:
                raise ValueError(
                    f"Object artifact is incompatible with manifest stage {manifest.stage!r}"
                )
            if artifact_report.get("run_id") != manifest.run_id:
                raise ValueError(f"Artifact run_id does not match manifest: {artifact_path}")
        actual_sha256 = sha256_path(artifact_path)
        matching_outputs = [
            item
            for item in manifest.outputs
            if item.source_id == artifact_path.name and item.sha256 == actual_sha256
        ]
        if not matching_outputs:
            raise ValueError(f"Artifact checksum does not match run manifest: {artifact_path}")
        artifact_report["checksum_verified"] = True
        artifact_report["manifest"] = str(path)
        manifests.append(
            {
                "path": str(path),
                "run_id": manifest.run_id,
                "stage": manifest.stage,
                "status": manifest.status,
            }
        )
    report.update(
        {
            "ok": True,
            "artifacts": artifact_reports,
            "frames": total_frames,
            "regions": total_regions,
            "caption_statuses": caption_statuses,
            "manifests": manifests,
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
