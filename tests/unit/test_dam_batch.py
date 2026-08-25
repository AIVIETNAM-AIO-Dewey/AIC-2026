"""Unit tests for multi-worker DAM batch runner."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest
from scripts.run_dam_batch import load_master_video_list, PathResolver, is_video_completed


def test_master_video_list_sharding():
    list_path = REPO_ROOT / "configs/master_video_list.txt"
    assert list_path.exists(), "master_video_list.txt must exist"
    
    all_videos = load_master_video_list(list_path, objects_root=REPO_ROOT / "data")
    assert len(all_videos) == 873, f"Expected 873 videos, got {len(all_videos)}"
    assert len(set(all_videos)) == 873, "Master list contains duplicates"

    num_workers = 8
    worker_slices = []
    for worker_id in range(num_workers):
        assigned = all_videos[worker_id::num_workers]
        assert len(assigned) in (109, 110), f"Worker {worker_id} workload {len(assigned)} outside [109, 110]"
        worker_slices.append(set(assigned))

    # Verify 0 overlap between any two workers
    for i in range(num_workers):
        for j in range(i + 1, num_workers):
            overlap = worker_slices[i].intersection(worker_slices[j])
            assert not overlap, f"Workers {i} and {j} have overlapping assignments: {overlap}"

    # Verify complete union equals 873
    all_assigned = set().union(*worker_slices)
    assert len(all_assigned) == 873, "Assigned union does not cover all 873 videos"


def test_path_resolver_local_fixture(tmp_path: Path):
    kf_root = tmp_path / "Keyframes" / "Keyframes_L21" / "keyframes" / "L21_V001"
    kf_root.mkdir(parents=True)
    (kf_root / "001.jpg").write_text("dummy")

    obj_root = tmp_path / "objects" / "L21_V001"
    obj_root.mkdir(parents=True)
    (obj_root / "001.json").write_text("{}")

    map_root = tmp_path / "map-keyframes"
    map_root.mkdir(parents=True)
    (map_root / "L21_V001.csv").write_text("n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n")

    resolver = PathResolver(
        keyframes_root=tmp_path / "Keyframes",
        objects_root=tmp_path / "objects",
        map_keyframes_root=tmp_path / "map-keyframes",
    )

    f_dir = resolver.resolve_frames_dir("L21_V001")
    assert f_dir == kf_root

    o_dir = resolver.resolve_objects_dir("L21_V001")
    assert o_dir == obj_root

    m_csv = resolver.resolve_map_csv("L21_V001")
    assert m_csv == map_root / "L21_V001.csv"


def test_create_keyframe_zip(tmp_path: Path):
    from scripts.run_dam_batch import create_keyframe_zip
    import zipfile

    kf_dir = tmp_path / "keyframes" / "L21_V001"
    kf_dir.mkdir(parents=True)
    (kf_dir / "00000001.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    (kf_dir / "00000002.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    zip_out = tmp_path / "keyframes_zips" / "L21_V001.zip"
    created = create_keyframe_zip(kf_dir, zip_out)
    assert created is not None
    assert zip_out.is_file()

    with zipfile.ZipFile(zip_out, "r") as zf:
        namelist = zf.namelist()
        assert "L21_V001/00000001.jpg" in namelist
        assert "L21_V001/00000002.jpg" in namelist
