"""Batch CLI Submission Runner for AIC 2026 Competition.

Takes a folder containing query text files (e.g., query-1-kis.txt, query-2-qa.txt, query-3-trake.txt),
automatically runs the full 2-stage multimodal pipeline on each query, generates the official CSVs,
and packages them into a valid `submission.zip` with the required `submission/` directory.

Usage:
    python -m online.src.submission.run_batch_submission --queries-dir /path/to/batch1 --output-zip team_AIC_round1.zip
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import tempfile
from pathlib import Path

from online.src.retrieval.pipeline import VideoRetrievalEngine
from online.src.submission.export_submission import export_query_csv, package_submission_zip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def infer_task_type(filename: str) -> str:
    """Infer task type from official AIC query file suffix."""
    name_lower = filename.lower()
    if "trake" in name_lower:
        return "TRAKE"
    elif "qa" in name_lower or "vqa" in name_lower or "q&a" in name_lower:
        return "VQA"
    else:
        return "KIS"


def run_batch_submission(
    queries_dir: str | Path,
    output_zip: str | Path = "submission.zip",
    top_k: int = 100,
    qdrant_db_path: str = "/Users/khoale/Downloads/AIC_HCM/qdrant_db",
) -> Path:
    """Process all query files in queries_dir, run retrieval, and create submission ZIP."""
    q_dir = Path(queries_dir)
    if not q_dir.exists() or not q_dir.is_dir():
        logger.error(f"❌ Queries directory not found: {q_dir}")
        sys.exit(1)

    query_files = sorted(
        [p for p in q_dir.glob("*.txt") if not p.name.startswith(".")],
        key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.stem)]
    )

    if not query_files:
        logger.warning(f"⚠️ No .txt query files found in {q_dir}")
        return Path(output_zip)

    logger.info("=" * 80)
    logger.info(f"🚀 STARTING BATCH SUBMISSION GENERATION ({len(query_files)} Queries)")
    logger.info("=" * 80)

    # Initialize Engine & Pre-warm
    engine = VideoRetrievalEngine(qdrant_db_path=qdrant_db_path)
    engine.models.warmup()

    exported_csvs = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for q_file in query_files:
            task_type = infer_task_type(q_file.name)
            query_text = q_file.read_text(encoding="utf-8").strip()

            logger.info(f"\n🔍 Processing: {q_file.name} [{task_type}]")
            logger.info(f"   Query: \"{query_text[:90]}{'...' if len(query_text) > 90 else ''}\"")

            # Run Full Multimodal Search
            resp = engine.search(query=query_text, task_type=task_type, top_k=top_k)

            # Export corresponding CSV: query-X-kis.txt -> query-X-kis.csv
            csv_filename = q_file.stem + ".csv"
            csv_out = tmp_path / csv_filename
            export_query_csv(csv_out, resp, max_rows=top_k)
            exported_csvs.append(csv_out)

        # Package into submission.zip with required submission/ root directory
        logger.info("\n" + "=" * 80)
        logger.info("📦 PACKAGING ALL CSVs INTO OFFICIAL SUBMISSION ARCHIVE...")
        final_zip = package_submission_zip(exported_csvs, output_zip)
        logger.info(f"🎉 SUCCESS! Official submission file ready: {final_zip.resolve()}")
        logger.info("=" * 80)

        return final_zip


def main():
    parser = argparse.ArgumentParser(description="Official AIC 2026 Batch Submission Packager")
    parser.add_argument("--queries-dir", type=str, required=True, help="Folder containing query-X-*.txt files")
    parser.add_argument("--output-zip", type=str, default="submission.zip", help="Output ZIP path (default: submission.zip)")
    parser.add_argument("--top-k", type=int, default=100, help="Max rows per CSV (default: 100)")
    parser.add_argument("--qdrant-db", type=str, default="/Users/khoale/Downloads/AIC_HCM/qdrant_db", help="Path to Qdrant DB")

    args = parser.parse_args()
    run_batch_submission(
        queries_dir=args.queries_dir,
        output_zip=args.output_zip,
        top_k=args.top_k,
        qdrant_db_path=args.qdrant_db,
    )


if __name__ == "__main__":
    main()
