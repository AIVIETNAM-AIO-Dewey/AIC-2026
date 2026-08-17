from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import yaml
from aic2026.common.data_prep import discover_archives, prepare_subset, safe_member_path


def test_locked_l21_regression_counts_match_real_inventory() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "configs" / "data" / "aic25-b1.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["subsets"]["L21"] == {
        "videos": 29,
        "keyframes_raw": 7800,
        "keyframes_canonical": 7790,
        "objects": 7800,
        "duplicate_frame_idx_rows": 10,
    }


def _zip(path: Path, values: dict[str, str | bytes]) -> str:
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in values.items():
            archive.writestr(name, value)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(root: Path) -> dict:
    mapping = "n,pts_time,fps,frame_idx\n1,0,25,0\n2,0.01,25,0\n3,1,25,25\n"
    files = {
        "Videos_L21_a.zip": {"video/L21_V001.mp4": b"video"},
        "Keyframes_L21.zip": {
            "keyframes/L21_V001/001.jpg": b"one",
            "keyframes/L21_V001/002.jpg": b"two",
            "keyframes/L21_V001/003.jpg": b"three",
        },
        "map-keyframes-aic25-b1.zip": {"map-keyframes/L21_V001.csv": mapping},
        "objects-aic25-b1.zip": {
            f"objects/L21_V001/{index:03d}.json": "{}" for index in range(1, 4)
        },
        "media-info-aic25-b1.zip": {"media-info/L21_V001.json": "{}"},
        "clip-features-32-aic25-b1.zip": {"clip-features-32/L21_V001.npy": b"skip"},
    }
    config = {"archives": {}}
    for name, entries in files.items():
        config["archives"][name] = {"sha256": _zip(root / name, entries)}
    config["archives"]["clip-features-32-aic25-b1.zip"]["ignored_by_policy"] = True
    return config


def test_prepare_selects_complete_subset_and_records_duplicates(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    config = _sources(raw)
    inventory = prepare_subset(
        specs=discover_archives(raw, config),
        prepared_root=tmp_path / "prepared",
        subset="L21",
        resume=False,
    )
    target = tmp_path / "prepared" / "L21"
    assert inventory["counts"] == {
        "videos": 1,
        "keyframes_raw": 3,
        "keyframes_canonical": 2,
        "objects": 3,
        "duplicate_frame_idx_rows": 1,
    }
    assert inventory["duplicate_frame_idx"]["L21_V001"] == [
        {
            "frame_idx": 0,
            "discarded_keyframe_n": 1,
            "kept_keyframe_n": 2,
            "discarded_pts_time_s": 0.0,
            "kept_pts_time_s": 0.01,
        }
    ]
    assert (target / "videos" / "L21_V001.mp4").read_bytes() == b"video"
    assert not (target / "clip-features-32").exists()
    inventory = json.loads((target / "inventory.json").read_text(encoding="utf-8"))
    assert inventory["status"] == "completed"


def test_resume_rejects_changed_source(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    config = _sources(raw)
    specs = discover_archives(raw, config)
    prepare_subset(specs=specs, prepared_root=tmp_path / "prepared", subset="L21", resume=False)
    changed = dict(config)
    changed["archives"] = {name: dict(value) for name, value in config["archives"].items()}
    changed["archives"]["Videos_L21_a.zip"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_subset(
            specs=discover_archives(raw, changed),
            prepared_root=tmp_path / "prepared",
            subset="L21",
            resume=True,
        )


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape"])
def test_zip_traversal_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        safe_member_path(name)
