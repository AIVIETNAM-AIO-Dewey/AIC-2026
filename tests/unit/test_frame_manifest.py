from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from aic2026.common.frame_manifest import build_frame_refs, read_frame_map

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_attached_mapping_shape_is_modeled_without_time_recalculation(tmp_path: Path) -> None:
    frames = tmp_path / "keyframes" / "L21_V011"
    frames.mkdir(parents=True)
    for keyframe_n in (1, 2, 3):
        Image.new("RGB", (16, 9)).save(frames / f"{keyframe_n:06d}.jpg")

    refs = build_frame_refs(
        video_id="L21_V011",
        map_csv=FIXTURES / "frame_map.csv",
        frames_dir=frames,
        data_root=tmp_path,
    )

    assert [(ref.keyframe_n, ref.frame_idx, ref.pts_time_s) for ref in refs] == [
        (1, 0, 0.0),
        (2, 90, 3.6),
        (3, 265, 10.6),
    ]
    assert refs[0].frame_uid == "L21_V011:0"


def test_duplicate_mapping_keys_fail(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("n,pts_time,fps,frame_idx\n1,0,25,0\n1,1,25,25\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate n"):
        read_frame_map(path)


def test_full_join_rejects_surplus_frame_number(tmp_path: Path) -> None:
    frames = tmp_path / "keyframes" / "L21_V011"
    frames.mkdir(parents=True)
    for keyframe_n in (1, 2, 3, 99):
        Image.new("RGB", (16, 9)).save(frames / f"{keyframe_n:06d}.jpg")

    with pytest.raises(ValueError, match="absent from mapping"):
        build_frame_refs(
            video_id="L21_V011",
            map_csv=FIXTURES / "frame_map.csv",
            frames_dir=frames,
            data_root=tmp_path,
        )


def test_corrupt_frame_fails_manifest_build(tmp_path: Path) -> None:
    frames = tmp_path / "keyframes" / "L21_V011"
    frames.mkdir(parents=True)
    (frames / "000001.jpg").write_bytes(b"not a jpeg")

    with pytest.raises(OSError):
        build_frame_refs(
            video_id="L21_V011",
            map_csv=FIXTURES / "frame_map.csv",
            frames_dir=frames,
            data_root=tmp_path,
            limit=1,
        )
