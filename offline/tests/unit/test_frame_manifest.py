from __future__ import annotations

from pathlib import Path

import pytest
from aic2026.common.frame_manifest import build_frame_refs, read_frame_map
from PIL import Image

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
    assert refs[0].source_image_sha256 is not None
    assert len(refs[0].source_image_sha256) == 64


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


def test_default_keeps_second_keyframe_per_duplicate_frame_idx(tmp_path: Path) -> None:
    """Organizer maps sometimes floor pts_time*fps, colliding row 2 into row 1."""
    csv_path = tmp_path / "L21_V006.csv"
    csv_path.write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,0.0333333,30.0,0\n3,3.03333,30.0,90\n",
        encoding="utf-8",
    )
    rows = read_frame_map(csv_path)
    assert [row.frame_idx for row in rows] == [0, 90]
    # Every surviving frame_idx is exactly what the organizer shipped.
    assert [row.keyframe_n for row in rows] == [2, 3]
    assert rows[0].discarded_keyframe_ns == (1,)


def test_drop_picks_the_middle_of_a_longer_run(tmp_path: Path) -> None:
    csv_path = tmp_path / "L21_V097.csv"
    csv_path.write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,0.01,30.0,0\n3,0.02,30.0,0\n4,3.0,30.0,90\n",
        encoding="utf-8",
    )
    rows = read_frame_map(csv_path, drop_duplicate_frame_idx=True)
    assert [row.keyframe_n for row in rows] == [2, 4]
    assert rows[0].discarded_keyframe_ns == (1, 3)


def test_drop_is_deterministic(tmp_path: Path) -> None:
    csv_path = tmp_path / "L21_V096.csv"
    csv_path.write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,0.0333333,30.0,0\n3,3.03333,30.0,90\n",
        encoding="utf-8",
    )
    first = read_frame_map(csv_path, drop_duplicate_frame_idx=True)
    assert first == read_frame_map(csv_path, drop_duplicate_frame_idx=True)


def test_drop_still_rejects_an_unexplained_regression(tmp_path: Path) -> None:
    """Dropping handles ties, not a genuine backwards jump."""
    csv_path = tmp_path / "L21_V098.csv"
    csv_path.write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,50\n2,1.0,30.0,30\n3,4.0,30.0,120\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frame_idx must be strictly increasing"):
        read_frame_map(csv_path, drop_duplicate_frame_idx=True)


def test_strict_duplicate_policy_is_available_for_audits(tmp_path: Path) -> None:
    csv_path = tmp_path / "L21_V006.csv"
    csv_path.write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,0.0333333,30.0,0\n3,3.03333,30.0,90\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate frame_idx"):
        read_frame_map(csv_path, drop_duplicate_frame_idx=False)


def test_discarded_keyframe_image_is_not_reported_as_surplus(tmp_path: Path) -> None:
    """The dropped keyframe still has a file on disk; that must not fail the build."""
    frames = tmp_path / "keyframes" / "L21_V006"
    frames.mkdir(parents=True)
    for n in (1, 2, 3):
        Image.new("RGB", (32, 18), color="white").save(frames / f"{n:03d}.jpg")
    csv_path = tmp_path / "L21_V006.csv"
    csv_path.write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,0.0333333,30.0,0\n3,3.03333,30.0,90\n",
        encoding="utf-8",
    )
    refs = build_frame_refs(
        video_id="L21_V006",
        map_csv=csv_path,
        frames_dir=frames,
        data_root=tmp_path,
        drop_duplicate_frame_idx=True,
    )
    assert [ref.frame_idx for ref in refs] == [0, 90]
    assert [ref.frame_relpath.rsplit("/", 1)[-1] for ref in refs] == ["002.jpg", "003.jpg"]
