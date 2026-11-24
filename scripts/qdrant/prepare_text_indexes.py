"""Build the optional OCR FTS database outside of API startup."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from online.src.retrieval.modalities.ocr import build_ocr_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("AIC_DATA_ROOT", "/data"))
    )
    parser.add_argument(
        "--state-root", type=Path, default=Path(os.environ.get("AIC_STATE_ROOT", "/state"))
    )
    args = parser.parse_args()
    # ASR is prepared independently by prepare_asr_index.py.  This command
    # owns OCR only and never opens, rebuilds, or reports the ASR database.
    manifest = build_ocr_index(
        args.data_root / "ocr_transcripts",
        args.state_root / "ocr.sqlite3",
        data_root=args.data_root,
        manifest_path=args.state_root / "branch3_ocr_manifest.json",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
