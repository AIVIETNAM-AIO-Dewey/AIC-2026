#!/usr/bin/env python3
"""Validate ASR segment JSONL artifacts and companion manifests against Pydantic contracts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _common import add_common_arguments  # noqa: E402

from aic2026.asr.validation import validate_jsonl, validate_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        help="Path to an asr_segments/<video_id>.jsonl file to validate (can be passed multiple times).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Directory containing asr_segments/*.jsonl files to validate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    artifacts: list[Path] = []
    if args.artifact:
        artifacts.extend([p.expanduser().resolve() for p in args.artifact])
    if args.artifact_dir:
        art_dir = args.artifact_dir.expanduser().resolve()
        if art_dir.exists():
            artifacts.extend(sorted(art_dir.glob("*.jsonl")))

    if not artifacts:
        print("Error: No JSONL artifacts specified. Use --artifact or --artifact-dir.", file=sys.stderr)
        return 1

    total_valid = 0
    total_invalid = 0

    for jsonl_path in artifacts:
        res = validate_jsonl(jsonl_path)
        manifest_path = jsonl_path.with_name(f"{jsonl_path.stem}.manifest.json")
        manifest_errors = []
        if manifest_path.exists():
            manifest_errors = validate_manifest(manifest_path, jsonl_result=res)

        if res.is_valid and not manifest_errors:
            print(res.summary())
            total_valid += 1
        else:
            print(f"FAILED: {jsonl_path.name}")
            for err in res.errors:
                print(f"  Line {err['line']}: {err['error']}")
            for m_err in manifest_errors:
                print(f"  Manifest Error: {m_err}")
            total_invalid += 1

    print("-" * 50)
    print(f"Validation summary: {total_valid} PASSED, {total_invalid} FAILED out of {len(artifacts)} artifacts.")
    return 0 if total_invalid == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
