from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from aic2026.common import iter_jsonl, write_jsonl_atomic
from aic2026.common.frame_manifest import build_frame_refs
from aic2026.contracts import ObjectFrameRecord
from aic2026.object_description import prepare_masks, run_descriptions
from aic2026.object_description.rle import rectangle_mask
from aic2026.object_description.sam_backend import MaskPrediction

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class FakeSam:
    def generate(self, image: Image.Image, boxes_xyxy: list[tuple[int, int, int, int]]):
        width, height = image.size
        return [
            MaskPrediction(rectangle_mask(height, width, box), "sam", 0.95) for box in boxes_xyxy
        ]


class FakeDam:
    def describe(self, image: Image.Image, mask: Image.Image, *, max_new_tokens: int = 48):
        assert image.size == mask.size
        assert mask.getbbox() is not None
        return (
            "A person in a bright red jacket speaks beside a black microphone on an outdoor stage."
        )


class AlwaysFailDam:
    def describe(self, image: Image.Image, mask: Image.Image, *, max_new_tokens: int = 48):
        raise RuntimeError("systemic DAM API mismatch")


TorchOutOfMemoryError = type(
    "OutOfMemoryError",
    (RuntimeError,),
    {"__module__": "torch.cuda"},
)


class AlwaysOomDam:
    def describe(self, image: Image.Image, mask: Image.Image, *, max_new_tokens: int = 48):
        raise TorchOutOfMemoryError("CUDA out of memory")


def test_synthetic_end_to_end_and_completed_resume(tmp_path: Path) -> None:
    frames_dir = tmp_path / "keyframes" / "L21_V011"
    objects_dir = tmp_path / "objects" / "L21_V011"
    frames_dir.mkdir(parents=True)
    objects_dir.mkdir(parents=True)
    Image.new("RGB", (100, 50), color="white").save(frames_dir / "000001.jpg")
    shutil.copy(FIXTURES / "objects" / "000001.json", objects_dir / "000001.json")
    frame_refs = build_frame_refs(
        video_id="L21_V011",
        map_csv=FIXTURES / "frame_map.csv",
        frames_dir=frames_dir,
        data_root=tmp_path,
        limit=1,
    )
    frame_manifest = tmp_path / "frames.jsonl"
    write_jsonl_atomic(frame_manifest, frame_refs)
    masks = tmp_path / "masks.jsonl"
    descriptions = tmp_path / "descriptions.jsonl"

    mask_counts = prepare_masks(
        frame_manifest=frame_manifest,
        objects_dir=objects_dir,
        data_root=tmp_path,
        output=masks,
        run_id="test-run",
        mask_backend=FakeSam(),
    )
    caption_counts = run_descriptions(
        mask_artifact=masks,
        data_root=tmp_path,
        output=descriptions,
        caption_backend=FakeDam(),
    )

    assert mask_counts["regions"] == 2
    assert caption_counts["captions_ok"] == 2
    record = ObjectFrameRecord.model_validate(next(iter_jsonl(descriptions)))
    assert [region.region_id for region in record.regions] == [
        "L21_V011:0:d000",
        "L21_V011:0:d003",
    ]
    assert all(region.caption.status == "ok" for region in record.regions)
    assert all(region.caption.word_count <= 20 for region in record.regions)

    failed_output = tmp_path / "all-error-descriptions.jsonl"
    with pytest.raises(RuntimeError, match="failed captions"):
        run_descriptions(
            mask_artifact=masks,
            data_root=tmp_path,
            output=failed_output,
            caption_backend=AlwaysFailDam(),
        )
    assert not failed_output.exists()
    assert failed_output.with_suffix(".jsonl.partial").exists()

    mask_record = ObjectFrameRecord.model_validate(next(iter_jsonl(masks)))
    extra_region = mask_record.regions[0].model_copy(
        update={"region_id": "L21_V011:0:d999", "source_detection_index": 999}
    )
    oom_masks = tmp_path / "oom-masks.jsonl"
    write_jsonl_atomic(
        oom_masks,
        [mask_record.model_copy(update={"regions": [*mask_record.regions, extra_region]})],
    )
    with pytest.raises(RuntimeError, match="3 consecutive DAM OOM"):
        run_descriptions(
            mask_artifact=oom_masks,
            data_root=tmp_path,
            output=tmp_path / "oom-descriptions.jsonl",
            caption_backend=AlwaysOomDam(),
        )

    description_partial = descriptions.with_suffix(".jsonl.partial")
    descriptions.replace(description_partial)
    resumed_descriptions = run_descriptions(
        mask_artifact=masks,
        data_root=tmp_path,
        output=descriptions,
        caption_backend=FakeDam(),
        resume=True,
    )
    assert resumed_descriptions["captions_ok"] == 2
    assert resumed_descriptions["skipped"] == 1

    partial = masks.with_suffix(masks.suffix + ".partial")
    masks.replace(partial)
    resumed_partial = prepare_masks(
        frame_manifest=frame_manifest,
        objects_dir=objects_dir,
        data_root=tmp_path,
        output=masks,
        run_id="test-run",
        mask_backend=FakeSam(),
        resume=True,
    )
    assert resumed_partial["skipped"] == 1
    assert resumed_partial["regions"] == 2
    assert len(list(iter_jsonl(masks))) == 1

    resumed = prepare_masks(
        frame_manifest=frame_manifest,
        objects_dir=objects_dir,
        data_root=tmp_path,
        output=masks,
        run_id="test-run",
        mask_backend=FakeSam(),
        resume=True,
    )
    assert resumed["resumed_complete"] == 1


def test_resume_rejects_schema_invalid_complete_partial_record(tmp_path: Path) -> None:
    partial_output = tmp_path / "masks.jsonl"
    partial = partial_output.with_suffix(".jsonl.partial")
    partial.write_text(json.dumps({"frame_uid": "L21_V011:0"}) + "\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        prepare_masks(
            frame_manifest=tmp_path / "unused-frames.jsonl",
            objects_dir=tmp_path / "unused-objects",
            data_root=tmp_path,
            output=partial_output,
            run_id="test-run",
            mask_backend=FakeSam(),
            resume=True,
        )


def test_direct_pipeline_rejects_non_positive_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        run_descriptions(
            mask_artifact=tmp_path / "unused.jsonl",
            data_root=tmp_path,
            output=tmp_path / "descriptions.jsonl",
            caption_backend=FakeDam(),
            limit=0,
        )


def test_resume_adds_separator_after_complete_line_without_newline(tmp_path: Path) -> None:
    frames_dir = tmp_path / "keyframes" / "L21_V011"
    objects_dir = tmp_path / "objects" / "L21_V011"
    frames_dir.mkdir(parents=True)
    objects_dir.mkdir(parents=True)
    for keyframe_n in (1, 2):
        Image.new("RGB", (100, 50), color="white").save(frames_dir / f"{keyframe_n:06d}.jpg")
        shutil.copy(
            FIXTURES / "objects" / "000001.json",
            objects_dir / f"{keyframe_n:06d}.json",
        )
    frame_refs = build_frame_refs(
        video_id="L21_V011",
        map_csv=FIXTURES / "frame_map.csv",
        frames_dir=frames_dir,
        data_root=tmp_path,
        limit=2,
    )
    frame_manifest = tmp_path / "frames.jsonl"
    write_jsonl_atomic(frame_manifest, frame_refs)
    output = tmp_path / "masks.jsonl"
    prepare_masks(
        frame_manifest=frame_manifest,
        objects_dir=objects_dir,
        data_root=tmp_path,
        output=output,
        run_id="test-run",
        mask_backend=FakeSam(),
        limit=1,
    )
    partial = output.with_suffix(".jsonl.partial")
    output.replace(partial)
    partial.write_bytes(partial.read_bytes().rstrip(b"\n"))

    counts = prepare_masks(
        frame_manifest=frame_manifest,
        objects_dir=objects_dir,
        data_root=tmp_path,
        output=output,
        run_id="test-run",
        mask_backend=FakeSam(),
        resume=True,
    )

    assert counts["frames"] == 2
    assert len(list(iter_jsonl(output))) == 2
