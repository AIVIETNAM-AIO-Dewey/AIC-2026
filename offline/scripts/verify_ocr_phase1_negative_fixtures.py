#!/usr/bin/env python3
"""Verify OCR Phase 1 negative source fixtures without constructing a detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _common import read_config  # noqa: E402
from aic2026.common import atomic_write_json, sha256_file  # noqa: E402
from aic2026.ocr import (  # noqa: E402
    canonical_config_sha256,
    verify_negative_fixture_suite,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "offline" / "ocr_phase1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--negative-manifest", type=Path, required=True)
    parser.add_argument("--negative-data-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = read_config(args.config)
        if config.get("schema_version") != "aic26.ocr_phase1.config.v1":
            raise ValueError("unsupported OCR Phase 1 config schema")
        config_hash = canonical_config_sha256(config)
        receipt_path = args.receipt.expanduser().resolve()
        if receipt_path.exists():
            raise FileExistsError("negative-suite receipt already exists")
        receipt, _baseline = verify_negative_fixture_suite(
            args.negative_manifest.expanduser().resolve(),
            args.negative_data_root.expanduser().resolve(),
            config_sha256=config_hash,
        )
        atomic_write_json(receipt_path, receipt.model_dump(mode="json"))
        print(
            json.dumps(
                {
                    "status": "negative_fixture_suite_pass",
                    "fixtures": receipt.fixture_count,
                    "negative_manifest_sha256": receipt.negative_manifest_sha256,
                    "receipt_sha256": sha256_file(receipt_path),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        print(f"negative fixture suite failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
