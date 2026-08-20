"""Official AIC 2026 Batch Submission Exporter and Packager.

Validates and exports search results according to official competition rules:
1. Textual KIS: `<video_id>, <frame_idx>`
2. Q&A: `<video_id>, <frame_idx>, "<answer>"` (max 100 chars, escaped quotes)
3. TRAKE: `<video_id>, <frame_1>, <frame_2>, ..., <frame_N>` (exactly N frames, monotonic)
4. Packaging: Creates `submission/` folder and compresses to `submission.zip` with no CSV headers.
"""

from __future__ import annotations

import csv
import logging
import zipfile
from pathlib import Path
from typing import Optional, Union

from online.src.contracts.query import ParsedQuery, SearchResponse, SearchResult

logger = logging.getLogger(__name__)


def clean_answer_for_csv(answer: str) -> str:
    """Format and sanitize Q&A answer according to official AIC-2026 CSV specs."""
    if not answer:
        return ""
    # Strip excessive whitespace and cap at 100 chars
    cleaned = answer.strip()[:100]
    # Remove raw linebreaks
    cleaned = cleaned.replace("\r", " ").replace("\n", " ").strip()
    return cleaned


def format_kis_row(video_id: str, frame_idx: int | str) -> str:
    """Format single row for KIS task: <video_name>, <frame_id>"""
    vid = str(video_id).replace(".mp4", "").strip()
    f_idx = int(frame_idx)
    return f"{vid}, {f_idx}"


def format_qa_row(video_id: str, frame_idx: int | str, answer: str) -> str:
    """Format single row for Q&A task: <video_name>, <frame_id>, "<answer>\""""
    vid = str(video_id).replace(".mp4", "").strip()
    f_idx = int(frame_idx)
    ans = clean_answer_for_csv(answer)
    # Always escape internal double quotes
    escaped_ans = ans.replace('"', '""')
    return f'{vid}, {f_idx}, "{escaped_ans}"'


def format_trake_row(video_id: str, frame_ids: list[int | str]) -> str:
    """Format single row for TRAKE task: <video_name>, <frame_1>, <frame_2>, ..., <frame_N>"""
    vid = str(video_id).replace(".mp4", "").strip()
    frames_str = ", ".join(str(int(f)) for f in frame_ids)
    return f"{vid}, {frames_str}"


def export_query_csv(
    output_filepath: Union[str, Path],
    search_response: SearchResponse,
    max_rows: int = 100,
) -> Path:
    """Write search results to an official AIC-2026 CSV file with no header."""
    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    task_type = search_response.task_type
    rows_to_write = search_response.results[:max_rows]

    lines = []
    for item in rows_to_write:
        if task_type == "KIS":
            lines.append(format_kis_row(item.video_id, item.frame_idx))
        elif task_type in ("VQA", "Q&A", "QA"):
            ans = search_response.vqa_answer or item.vqa_answer or ""
            lines.append(format_qa_row(item.video_id, item.frame_idx, ans))
        elif task_type == "TRAKE":
            if item.trake_matched_frames:
                lines.append(format_trake_row(item.video_id, item.trake_matched_frames))
            else:
                lines.append(format_kis_row(item.video_id, item.frame_idx))

    # Write UTF-8 CSV with no header
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")

    logger.info(f"✅ Exported {len(lines)} rows to {out_path} ({task_type})")
    return out_path


def package_submission_zip(
    csv_files: list[Union[str, Path]],
    output_zip_path: Union[str, Path] = "submission.zip",
) -> Path:
    """Package CSV files into submission.zip containing the required 'submission/' root folder."""
    zip_path = Path(output_zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for csv_file in csv_files:
            p = Path(csv_file)
            if p.exists():
                # Enforce archive path: submission/<filename.csv>
                arcname = f"submission/{p.name}"
                zf.write(p, arcname=arcname)
                logger.info(f"  📦 Archived: {arcname}")
            else:
                logger.warning(f"  ⚠️ CSV file not found: {p}")

    logger.info(f"🎉 Created official submission archive: {zip_path}")
    return zip_path
