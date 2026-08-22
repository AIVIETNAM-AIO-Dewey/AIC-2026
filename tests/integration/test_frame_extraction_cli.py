from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from aic2026.contracts import FrameSampleRecord, RunManifest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_extract_frame_samples_cli_with_synthetic_video(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe are required for frame extraction CLI integration")

    video = tmp_path / "aic-26-video" / "videos" / "L21_V001.mp4"
    video.parent.mkdir(parents=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x36:rate=5:duration=2",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    map_csv = tmp_path / "aic-test-dataset" / "map-keyframes" / "L21_V001.csv"
    map_csv.parent.mkdir(parents=True)
    map_csv.write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,5.0,0\n2,1.0,5.0,5\n",
        encoding="utf-8",
    )
    artifact_root = tmp_path / "artifacts"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "extract_frame_samples.py"),
            "--config",
            str(REPO_ROOT / "configs" / "offline" / "frame_extraction.yaml"),
            "--video-id",
            "L21_V001",
            "--search-root",
            str(tmp_path),
            "--output-root",
            str(artifact_root),
            "--limit",
            "2",
            "--resume",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["frames"] == 2
    output = Path(report["output"])
    records = [
        FrameSampleRecord.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 2
    for record in records:
        image_path = artifact_root / record.frame_relpath
        with Image.open(image_path) as image:
            assert image.size == (64, 36)
    manifest = RunManifest.model_validate_json(
        output.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.status == "completed"
    assert manifest.stage == "frame_extraction"
