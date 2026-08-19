#!/usr/bin/env python3
"""Measure the required synthetic sparse-matcher workloads without model evidence."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aic2026.contracts import OcrDetection  # noqa: E402
from aic2026.ocr.tracking import (  # noqa: E402
    TrackingConfig,
    _minimum_cost_maximum_matching,
    sparse_matching_structure,
)


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _workload(
    name: str,
) -> tuple[
    list[str],
    list[OcrDetection],
    dict[tuple[str, str], tuple[float, float, float, int]],
]:
    if name == "chain-2000":
        rows, columns = 2_000, 2_000
        edges = [(index, index) for index in range(rows)] + [
            (index, index + 1) for index in range(rows - 1)
        ]
    elif name == "star-20000x1":
        rows, columns = 20_000, 1
        edges = [(index, 0) for index in range(rows)]
    elif name == "dense-500x500":
        rows, columns = 500, 500
        edges = [(row, column) for row in range(rows) for column in range(columns)]
    else:
        raise ValueError(f"unknown benchmark workload: {name}")
    trajectory_ids = [f"traj-{index:06d}" for index in range(rows)]
    detections = [
        OcrDetection.model_construct(detection_id=f"det-{index:06d}", source_order=index)
        for index in range(columns)
    ]
    candidates = {
        (trajectory_ids[row], detections[column].detection_id): (
            abs(row - column) / max(rows, columns),
            0.0,
            0.0,
            1,
        )
        for row, column in edges
    }
    return trajectory_ids, detections, candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=("chain-2000", "star-20000x1", "dense-500x500"))
    args = parser.parse_args(argv)
    total_started = time.perf_counter()
    trajectory_ids, detections, candidates = _workload(args.workload)
    setup_seconds = time.perf_counter() - total_started
    solve_started = time.perf_counter()
    matches = _minimum_cost_maximum_matching(
        trajectory_ids,
        detections,
        candidates,
        config=TrackingConfig(
            maximum_candidate_edges_per_frame=300_000,
            maximum_candidate_edges_per_component=300_000,
        ),
    )
    solve_seconds = time.perf_counter() - solve_started
    expected_matches = min(len(trajectory_ids), len(detections))
    if len(matches) != expected_matches:
        raise RuntimeError(f"expected {expected_matches} matches, got {len(matches)}")
    print(
        json.dumps(
            {
                "workload": args.workload,
                "rows": len(trajectory_ids),
                "columns": len(detections),
                "candidate_edges": len(candidates),
                "matches": len(matches),
                "setup_seconds": setup_seconds,
                "solve_seconds": solve_seconds,
                "total_seconds": time.perf_counter() - total_started,
                "peak_rss_bytes": _peak_rss_bytes(),
                "structure": sparse_matching_structure(
                    len(trajectory_ids), len(detections), len(candidates)
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
