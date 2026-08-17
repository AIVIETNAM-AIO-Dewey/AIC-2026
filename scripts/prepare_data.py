#!/usr/bin/env python3
"""Verify organizer ZIP files and atomically prepare one complete data subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aic2026.common.data_prep import discover_archives, prepare_subset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument(
        "--archive-config",
        type=Path,
        default=REPO_ROOT / "configs" / "data" / "aic25-b1.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.archive_config.read_text(encoding="utf-8"))
    specs = discover_archives(args.raw_root.resolve(), config)
    inventory = prepare_subset(
        specs=specs,
        prepared_root=args.prepared_root,
        subset=args.subset,
        resume=args.resume,
        expected_counts=config.get("subsets", {}).get(args.subset),
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
