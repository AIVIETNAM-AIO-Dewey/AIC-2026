#!/usr/bin/env python3
"""Stage 3: On-Screen Text (OCR) Extraction with Vietnamese Normalization.

Runs lightweight, high-accuracy OCR on video keyframes, extracting on-screen text
(e.g., news banners, location titles, logos, subtitles) and normalizing Vietnamese diacritics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aic2026.common.io import iter_jsonl, write_jsonl_atomic  # noqa: E402
from aic2026.common.manifest import (  # noqa: E402
    complete_manifest,
    create_manifest,
    prepare_resume,
    write_manifest,
)
from aic2026.contracts import FrameRef  # noqa: E402
from aic2026.ocr import OcrReader  # noqa: E402
from _common import read_config, resolve_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True, help="Video identifier (e.g. L21_V001)")
    parser.add_argument("--frame-manifest", type=Path, required=True, help="Path to frame manifest JSONL")
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory containing keyframe images")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path for OCR transcripts")
    parser.add_argument("--config", type=Path, help="Path to YAML configuration")
    parser.add_argument("--device", default="auto", help="Execution device (auto, cuda, cpu)")
    parser.add_argument("--backend", default="auto", choices=["auto", "easyocr", "paddleocr"], help="OCR backend")
    parser.add_argument("--threshold", type=float, default=0.30, help="Confidence threshold for text lines")
    parser.add_argument("--limit", type=int, help="Optional frame limit for smoke testing")
    parser.add_argument("--no-resume", action="store_true", help="Force overwrite existing transcript artifact")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_id = args.video_id
    manifest_path = args.frame_manifest.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    meta_path = output_path.with_suffix(".manifest.json")

    config = read_config(args.config)
    seed = int(config.get("seed", 2026))
    device = resolve_device(args.device, config)
    run_id = str(config.get("run", {}).get("run_id", "ocr-extraction-v1"))

    resolved_config = {
        "schema_version": config.get("schema_version", "1.0"),
        "video_id": video_id,
        "device": device,
        "backend": args.backend,
        "threshold": args.threshold,
        "limit": args.limit,
    }

    manifest = create_manifest(
        run_id=run_id,
        stage="frame_manifest",
        config=resolved_config,
        seed=seed,
        input_paths=[("frame_manifest", manifest_path)],
        repo_root=REPO_ROOT,
    )
    manifest, complete = prepare_resume(
        manifest_path=meta_path,
        output_path=output_path,
        proposed=manifest,
        resume=not args.no_resume,
    )
    if complete:
        records = list(iter_jsonl(output_path))
        print(
            json.dumps(
                {
                    "status": "already_complete",
                    "video_id": video_id,
                    "frames": len(records),
                    "output": str(output_path),
                }
            )
        )
        return 0
    write_manifest(meta_path, manifest)

    # 1. Load Frame Manifest
    frame_refs = [FrameRef.model_validate(val) for val in iter_jsonl(manifest_path)]
    if not frame_refs:
        raise ValueError(f"Frame manifest is empty: {manifest_path}")
    if args.limit is not None and args.limit > 0:
        frame_refs = frame_refs[: args.limit]

    # 2. Initialize OCR Engine
    print(f"[ocr] Initializing {args.backend.upper()} on {device} (threshold={args.threshold}) ...", file=sys.stderr, flush=True)
    reader = OcrReader.create(
        backend=args.backend,
        device=device,
        threshold=args.threshold,
        languages=["vi", "en"],
    )

    # 3. Process Each Frame
    output_records: list[dict] = []
    total_lines = 0
    start_time = time.time()

    for idx, frame in enumerate(frame_refs, start=1):
        image_path = (data_root / frame.frame_relpath).resolve()
        if not image_path.is_file():
            # Fallback search if relative root shifted
            candidates = list(data_root.glob(f"**/{frame.video_id}/*{frame.frame_idx:08d}.jpg"))
            if candidates:
                image_path = candidates[0]
            else:
                raise FileNotFoundError(f"Keyframe image not found: {image_path}")

        with Image.open(image_path) as img:
            image_rgb = img.convert("RGB")
            ocr_res = reader.extract(image_rgb, image_path=image_path)

        spans_json = [
            {
                "line_id": span.line_id,
                "raw_text": span.raw_text,
                "normalized_text": span.normalized_text,
                "confidence": round(span.confidence, 4),
                "polygon_norm": [[round(x, 4), round(y, 4)] for x, y in span.normalized_polygon_xy],
                "reading_order": span.reading_order,
            }
            for span in ocr_res.spans
        ]
        total_lines += len(spans_json)

        output_records.append(
            {
                "video_id": frame.video_id,
                "frame_uid": frame.frame_uid,
                "keyframe_n": frame.keyframe_n,
                "frame_idx": frame.frame_idx,
                "pts_time_s": frame.pts_time_s,
                "full_text": ocr_res.full_text,
                "spans": spans_json,
            }
        )

        if idx == 1 or idx % 25 == 0 or idx == len(frame_refs):
            print(f"[ocr] Processed {idx}/{len(frame_refs)} frames ({total_lines} text lines extracted)", file=sys.stderr, flush=True)

    # 4. Save Transcripts Atomic
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_path, output_records)

    # 5. Finalize Manifest
    manifest = complete_manifest(
        manifest,
        counters={"frames": len(output_records), "text_lines": total_lines},
        shard=video_id,
        output_paths=[("ocr_transcripts", output_path)],
    )
    write_manifest(meta_path, manifest)

    elapsed = time.time() - start_time
    print(
        json.dumps(
            {
                "status": "completed",
                "video_id": video_id,
                "frames": len(output_records),
                "text_lines": total_lines,
                "elapsed_s": round(elapsed, 2),
                "output": str(output_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
