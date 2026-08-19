from __future__ import annotations

import hashlib
import io
import math
import random
from fractions import Fraction

import aic2026.ocr.tracking as tracking_module
import pytest
from aic2026.contracts import (
    CropProvenance,
    OcrDetection,
    OcrDetectionFrameRecord,
    QuadGeometry,
    RawQuadGeometry,
)
from aic2026.ocr.geometry import (
    CropConfig,
    CropGeometryError,
    canonical_quad,
    encode_crop,
    reconstruct_crop,
)
from aic2026.ocr.tracking import (
    TrackingConfig,
    _association_cost,
    _minimum_cost_maximum_matching,
    build_trajectories,
    select_representatives,
    sparse_matching_structure,
    validate_and_sort_detection_frames,
)
from PIL import Image, ImageDraw

REVISION = "1" * 64
CONFIG_HASH = "2" * 64
CROP_HASH = "3" * 64
TREE_HASH = "5" * 64
RUNTIME_HASH = "6" * 64
CANONICAL_HASH = "7" * 64


def test_tilted_and_edge_polygon_crop_is_reconstructable_and_lossless() -> None:
    image = Image.new("RGB", (80, 50), "white")
    ImageDraw.Draw(image).polygon([(0, 7), (50, 2), (55, 28), (0, 32)], fill="black")
    crop = encode_crop(
        image,
        canonical_quad([(55, 28), (-4, 33), (-3, 7), (50, 2)]),
        config=CropConfig(),
    )

    assert crop.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert crop.provenance.output_height >= 96
    assert crop.provenance.edge_truncation_penalty > 0
    reconstructed = reconstruct_crop(image, crop.provenance)
    assert reconstructed == crop.png_bytes
    assert hashlib.sha256(reconstructed).hexdigest() == crop.provenance.png_sha256


def test_padding_expands_both_vertical_edges_without_clipping_diacritics() -> None:
    image = Image.new("RGB", (120, 100), "white")
    polygon = [(20, 30), (100, 25), (102, 70), (18, 75)]
    crop = encode_crop(image, polygon, config=CropConfig())
    padded = crop.provenance.padded_polygon_xy.points
    ordered = canonical_quad(polygon)

    assert min(y for _, y in padded) < min(y for _, y in ordered)
    assert max(y for _, y in padded) > max(y for _, y in ordered)
    assert crop.provenance.edge_truncation_penalty == 0


def test_degenerate_polygon_is_explicitly_rejected() -> None:
    image = Image.new("RGB", (30, 30), "white")
    with pytest.raises(CropGeometryError, match="degenerate"):
        encode_crop(image, [(1, 1), (2, 2), (3, 3), (4, 4)], config=CropConfig())


def test_small_crop_upscales_to_exact_minimum_height_with_aspect_ratio() -> None:
    image = Image.new("RGB", (80, 40), "white")
    crop = encode_crop(
        image,
        [(10, 10), (50, 10), (50, 20), (10, 20)],
        config=CropConfig(horizontal_padding_ratio=0, vertical_padding_ratio=0),
    )
    assert crop.provenance.output_height == 96
    assert crop.provenance.output_width == 384


def test_vertical_crop_uses_paddlex_ccw_rotation_before_bytes_and_metrics() -> None:
    quad = canonical_quad(((42, 10), (78, 10), (78, 130), (42, 130)))
    image = Image.new("RGB", (120, 150), "white")
    _paint_oriented_quad(image, quad)
    config = CropConfig(
        horizontal_padding_ratio=0,
        vertical_padding_ratio=0,
        minimum_height_px=1,
    )
    encoded = encode_crop(image, quad, config=config)
    crop = Image.open(io.BytesIO(encoded.png_bytes)).convert("RGB")
    assert encoded.provenance.algorithm == "aic26.pil_quad_crop.v3"
    assert encoded.provenance.rotation_quadrants_ccw == 1
    assert (crop.width, crop.height) == (120, 36)
    assert reconstruct_crop(image, encoded.provenance) == encoded.png_bytes
    samples = (
        crop.getpixel((crop.width // 4, crop.height // 4)),
        crop.getpixel((3 * crop.width // 4, crop.height // 4)),
        crop.getpixel((crop.width // 4, 3 * crop.height // 4)),
        crop.getpixel((3 * crop.width // 4, 3 * crop.height // 4)),
    )
    assert samples[0][1] > max(samples[0][0], samples[0][2])
    assert samples[1][0] > 150 and samples[1][1] > 150 and samples[1][2] < 100
    assert samples[2][0] > max(samples[2][1:])
    assert samples[3][2] > max(samples[3][:2])


def _rotated_quad(angle_degrees: float) -> tuple[tuple[float, float], ...]:
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)

    def rotate(x: float, y: float) -> tuple[float, float]:
        return (120 + x * cosine - y * sine, 120 + x * sine + y * cosine)

    return tuple(rotate(x, y) for x, y in ((-60, -18), (60, -18), (60, 18), (-60, 18)))


def _paint_oriented_quad(image: Image.Image, quad: tuple[tuple[float, float], ...]) -> None:
    tl, tr, br, bl = quad

    def blend(u: float, v: float) -> tuple[float, float]:
        top = (tl[0] * (1 - u) + tr[0] * u, tl[1] * (1 - u) + tr[1] * u)
        bottom = (bl[0] * (1 - u) + br[0] * u, bl[1] * (1 - u) + br[1] * u)
        return (top[0] * (1 - v) + bottom[0] * v, top[1] * (1 - v) + bottom[1] * v)

    draw = ImageDraw.Draw(image)
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    cells = ((0, 0), (1, 0), (0, 1), (1, 1))
    for color, (column, row) in zip(colors, cells, strict=True):
        u0, u1 = column / 2, (column + 1) / 2
        v0, v1 = row / 2, (row + 1) / 2
        draw.polygon(
            [blend(u0, v0), blend(u1, v0), blend(u1, v1), blend(u0, v1)],
            fill=color,
        )


def test_required_clipped_repro_crop_is_permutation_stable_and_not_mirrored() -> None:
    raw = ((0.0, 45.0), (35.0, 100.0), (21.0, 100.0), (0.0, 59.0))
    quad = canonical_quad(raw)
    image = Image.new("RGB", (36, 101), "white")
    _paint_oriented_quad(image, quad)
    config = CropConfig(
        horizontal_padding_ratio=0,
        vertical_padding_ratio=0,
        minimum_height_px=40,
    )
    expected = encode_crop(image, quad, config=config)
    assert expected.provenance.rotation_quadrants_ccw == 0
    crop = Image.open(io.BytesIO(expected.png_bytes)).convert("RGB")
    samples = (
        crop.getpixel((crop.width // 4, crop.height // 4)),
        crop.getpixel((3 * crop.width // 4, crop.height // 4)),
        crop.getpixel((crop.width // 4, 3 * crop.height // 4)),
        crop.getpixel((3 * crop.width // 4, 3 * crop.height // 4)),
    )
    assert samples[0][0] > max(samples[0][1:])
    assert samples[1][1] > max(samples[1][0], samples[1][2])
    assert samples[2][2] > max(samples[2][:2])
    assert samples[3][0] > 150 and samples[3][1] > 150 and samples[3][2] < 100
    variants = [raw[offset:] + raw[:offset] for offset in range(4)] + [
        tuple(reversed(raw[offset:] + raw[:offset])) for offset in range(4)
    ]
    assert all(
        encode_crop(image, canonical_quad(variant), config=config).png_bytes == expected.png_bytes
        for variant in variants
    )


@pytest.mark.parametrize(
    "quad",
    [
        *[_rotated_quad(angle) for angle in (60, -60, 75, -75, 80, -80, 89, -89)],
        ((35.0, 65.0), (205.0, 45.0), (190.0, 160.0), (55.0, 175.0)),
        ((90.0, 20.0), (150.0, 45.0), (205.0, 215.0), (35.0, 190.0)),
    ],
)
def test_steep_and_perspective_crop_preserves_horizontal_and_vertical_orientation(
    quad: tuple[tuple[float, float], ...],
) -> None:
    quad = canonical_quad(quad)
    image = Image.new("RGB", (240, 240), "white")
    _paint_oriented_quad(image, quad)
    config = CropConfig(
        horizontal_padding_ratio=0,
        vertical_padding_ratio=0,
        minimum_height_px=72,
    )
    first = encode_crop(image, quad, config=config)
    second = encode_crop(image, quad, config=config)
    crop = Image.open(io.BytesIO(first.png_bytes)).convert("RGB")

    assert first.png_bytes == second.png_bytes
    assert first.provenance.output_height >= config.minimum_height_px
    samples = [
        crop.getpixel((crop.width // 4, crop.height // 4)),
        crop.getpixel((3 * crop.width // 4, crop.height // 4)),
        crop.getpixel((crop.width // 4, 3 * crop.height // 4)),
        crop.getpixel((3 * crop.width // 4, 3 * crop.height // 4)),
    ]
    if first.provenance.rotation_quadrants_ccw == 1:
        samples = [samples[2], samples[0], samples[3], samples[1]]
    elif first.provenance.rotation_quadrants_ccw == 3:
        samples = [samples[1], samples[3], samples[0], samples[2]]
    assert samples[0][0] > max(samples[0][1:])
    assert samples[1][1] > max(samples[1][0], samples[1][2])
    assert samples[2][2] > max(samples[2][:2])
    assert samples[3][0] > 150 and samples[3][1] > 150 and samples[3][2] < 100

    variants = [quad[offset:] + quad[:offset] for offset in range(4)] + [
        tuple(reversed(quad[offset:] + quad[:offset])) for offset in range(4)
    ]
    for variant in variants:
        recovered = canonical_quad(variant)
        for actual, expected in zip(recovered, quad, strict=True):
            assert actual == pytest.approx(expected)
        assert encode_crop(image, recovered, config=config).png_bytes == first.png_bytes


@pytest.mark.parametrize("angle", (80, 89))
def test_opposite_near_vertical_tilts_share_final_resolution_policy(angle: int) -> None:
    positive_quad = _rotated_quad(angle)
    negative_quad = _rotated_quad(-angle)
    positive_image = Image.new("RGB", (240, 240), "white")
    negative_image = Image.new("RGB", (240, 240), "white")
    _paint_oriented_quad(positive_image, positive_quad)
    _paint_oriented_quad(negative_image, negative_quad)
    config = CropConfig(
        horizontal_padding_ratio=0,
        vertical_padding_ratio=0,
        minimum_height_px=96,
    )
    positive = encode_crop(positive_image, canonical_quad(positive_quad), config=config)
    negative = encode_crop(negative_image, canonical_quad(negative_quad), config=config)

    assert positive.provenance.rotation_quadrants_ccw == 0
    assert negative.provenance.rotation_quadrants_ccw == 3
    assert positive.provenance.output_height >= config.minimum_height_px
    assert negative.provenance.output_height >= config.minimum_height_px
    assert (
        positive.provenance.output_width,
        positive.provenance.output_height,
    ) == (
        negative.provenance.output_width,
        negative.provenance.output_height,
    )
    for encoded in (positive, negative):
        crop = Image.open(io.BytesIO(encoded.png_bytes)).convert("RGB")
        samples = (
            crop.getpixel((crop.width // 4, crop.height // 4)),
            crop.getpixel((3 * crop.width // 4, crop.height // 4)),
            crop.getpixel((crop.width // 4, 3 * crop.height // 4)),
            crop.getpixel((3 * crop.width // 4, 3 * crop.height // 4)),
        )
        assert samples[0][0] > max(samples[0][1:])
        assert samples[1][1] > max(samples[1][0], samples[1][2])
        assert samples[2][2] > max(samples[2][:2])
        assert samples[3][0] > 150 and samples[3][1] > 150 and samples[3][2] < 100
    assert reconstruct_crop(positive_image, positive.provenance) == positive.png_bytes
    assert reconstruct_crop(negative_image, negative.provenance) == negative.png_bytes


def _crop(visual_hash: str, *, sharpness: float = 0.5, width: int = 200) -> CropProvenance:
    return CropProvenance(
        crop_config_sha256=CROP_HASH,
        padded_polygon_xy=QuadGeometry(
            points=((10.0, 10.0), (50.0, 10.0), (50.0, 30.0), (10.0, 30.0))
        ),
        perspective_width=width,
        perspective_height=96,
        output_width=width,
        output_height=96,
        png_sha256=hashlib.sha256(visual_hash.encode()).hexdigest(),
        visual_hash=visual_hash,
        sharpness=sharpness,
        edge_truncation_penalty=0,
    )


def _frame(
    video_id: str,
    frame_idx: int,
    *,
    visual_hash: str = "0" * 16,
    sharpness: float = 0.5,
    detector_score: float = 0.9,
    width: int = 200,
) -> OcrDetectionFrameRecord:
    frame_uid = f"{video_id}:{frame_idx}"
    detection = OcrDetection(
        detection_id=f"{frame_uid}:det-0000",
        source_order=0,
        polygon_raw_xy=RawQuadGeometry(
            points=((10.0, 10.0), (50.0, 10.0), (50.0, 30.0), (10.0, 30.0))
        ),
        polygon_xy=QuadGeometry(points=((10.0, 10.0), (50.0, 10.0), (50.0, 30.0), (10.0, 30.0))),
        polygon_clamped=False,
        detector_score=detector_score,
        crop=_crop(visual_hash, sharpness=sharpness, width=width),
    )
    return OcrDetectionFrameRecord(
        video_id=video_id,
        frame_uid=frame_uid,
        keyframe_n=frame_idx + 1,
        frame_idx=frame_idx,
        pts_time_s=float(frame_idx),
        fps=25.0,
        frame_relpath=f"frames/{video_id}/{frame_idx}.png",
        source_image_sha256="4" * 64,
        width=100,
        height=60,
        run_id="phase1-test",
        detector_revision=REVISION,
        detector_tree_sha256=TREE_HASH,
        runtime_identity_sha256=RUNTIME_HASH,
        config_sha256=CONFIG_HASH,
        canonical_image_sha256=CANONICAL_HASH,
        detections=[detection],
    )


def test_natural_numeric_frame_order_and_no_cross_video_tracking() -> None:
    records = [_frame("video10", 0), _frame("video2", 10), _frame("video2", 0)]
    ordered = validate_and_sort_detection_frames(records)
    assert [(item.video_id, item.frame_idx) for item in ordered] == [
        ("video2", 0),
        ("video2", 10),
        ("video10", 0),
    ]
    trajectories = build_trajectories(records, config=TrackingConfig())
    assert len(trajectories) == 2
    assert [len(item.members) for item in trajectories] == [2, 1]
    assert all(len({member.video_id for member in item.members}) == 1 for item in trajectories)


def test_stationary_ticker_splits_when_visual_content_changes() -> None:
    records = [
        _frame("video1", 0, visual_hash="0" * 16),
        _frame("video1", 10, visual_hash="f" * 16),
    ]
    trajectories = build_trajectories(records, config=TrackingConfig())
    assert len(trajectories) == 2
    assert [len(item.members) for item in trajectories] == [1, 1]


@pytest.mark.parametrize(("length", "expected"), [(1, 1), (2, 2), (5, 3)])
def test_trajectory_lengths_keep_required_representatives(length: int, expected: int) -> None:
    records = [_frame("video1", index * 10) for index in range(length)]
    trajectories = build_trajectories(records, config=TrackingConfig())
    representatives = select_representatives(trajectories, config=TrackingConfig())
    assert len(trajectories) == 1
    assert len(representatives) == expected


def test_deterministic_top_three_and_fully_locked_tie_breaker() -> None:
    records = [_frame("video1", index * 10) for index in range(4)]
    trajectories = build_trajectories(records, config=TrackingConfig())
    first = select_representatives(trajectories, config=TrackingConfig())
    second = select_representatives(trajectories, config=TrackingConfig())

    assert first == second
    assert [item.frame_idx for item in first] == [0, 30, 10]
    assert [item.representative_rank for item in first] == [1, 2, 3]


def test_natural_sort_collisions_keep_exact_video_groups_and_unique_trajectories() -> None:
    records = [
        _frame(video_id, frame_idx)
        for frame_idx in (10, 0)
        for video_id in ("Video1", "video1", "video01")
    ]
    trajectories = build_trajectories(records, config=TrackingConfig())
    assert len(trajectories) == 3
    assert len({item.trajectory_id for item in trajectories}) == 3
    assert {item.video_id: len(item.members) for item in trajectories} == {
        "Video1": 2,
        "video1": 2,
        "video01": 2,
    }


def test_bipartite_matching_maximizes_cardinality_before_cost_and_is_replay_stable() -> None:
    x = _frame("video1", 10).detections[0]
    y = x.model_copy(update={"detection_id": "video1:10:det-0001", "source_order": 1})
    trajectories = ["video1:traj-000001", "video1:traj-000002"]
    metrics = (0.0, 0.0, 0.0, 1)
    candidates = {
        (trajectories[0], x.detection_id): metrics,
        (trajectories[0], y.detection_id): metrics,
        (trajectories[1], x.detection_id): metrics,
    }
    first = _minimum_cost_maximum_matching(
        trajectories, [x, y], candidates, config=TrackingConfig()
    )
    second = _minimum_cost_maximum_matching(
        list(reversed(trajectories)),
        [y, x],
        dict(reversed(list(candidates.items()))),
        config=TrackingConfig(),
    )
    assert [(trajectory, item.detection_id) for trajectory, item in first] == [
        (trajectories[0], y.detection_id),
        (trajectories[1], x.detection_id),
    ]
    assert first == second

    equal_cost = {
        (trajectory, detection.detection_id): metrics
        for trajectory in trajectories
        for detection in (x, y)
    }
    assert _minimum_cost_maximum_matching(
        trajectories, [x, y], equal_cost, config=TrackingConfig()
    ) == _minimum_cost_maximum_matching(
        list(reversed(trajectories)), [y, x], equal_cost, config=TrackingConfig()
    )


def test_multi_detection_candidate_graph_builds_two_length_two_trajectories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _frame("video1", 0)
    first.detections.append(
        first.detections[0].model_copy(
            update={"detection_id": "video1:0:det-0001", "source_order": 1}
        )
    )
    second = _frame("video1", 10)
    second.detections.append(
        second.detections[0].model_copy(
            update={"detection_id": "video1:10:det-0001", "source_order": 1}
        )
    )

    def candidate_graph(previous, current, *, frame_idx, config):
        del frame_idx, config
        previous_order = previous.detection_id.rsplit("-", 1)[1]
        current_order = current.detection_id.rsplit("-", 1)[1]
        if (previous_order, current_order) == ("0001", "0001"):
            return None
        return (0.0, 0.0, 0.0, 1)

    monkeypatch.setattr(tracking_module, "_association_metrics", candidate_graph)
    expected = build_trajectories([first, second], config=TrackingConfig())
    replay = build_trajectories([second, first], config=TrackingConfig())

    assert [len(item.members) for item in expected] == [2, 2]
    assert expected == replay


def _brute_force_matching_objective(
    trajectories: list[str],
    detection_ids: list[str],
    candidates: dict[tuple[str, str], tuple[float, float, float, int]],
    config: TrackingConfig,
) -> tuple[int, float]:
    best = (0, math.inf)

    def visit(row: int, used: set[str], matched: int, cost: float) -> None:
        nonlocal best
        if row == len(trajectories):
            if matched > best[0] or (matched == best[0] and cost < best[1]):
                best = (matched, cost)
            return
        visit(row + 1, used, matched, cost)
        trajectory_id = trajectories[row]
        for detection_id in detection_ids:
            metrics = candidates.get((trajectory_id, detection_id))
            if detection_id in used or metrics is None:
                continue
            visit(
                row + 1,
                used | {detection_id},
                matched + 1,
                cost + _association_cost(metrics, config),
            )

    visit(0, set(), 0, 0.0)
    return best


def test_sparse_matching_matches_brute_force_on_seeded_small_graphs() -> None:
    generator = random.Random(2026)
    config = TrackingConfig()
    prototype = _frame("video1", 10).detections[0]
    for case in range(100):
        size = generator.randint(1, 5)
        trajectories = [f"traj-{index}" for index in range(size)]
        detections = [
            prototype.model_copy(update={"detection_id": f"det-{index}", "source_order": index})
            for index in range(size)
        ]
        candidates = {}
        for trajectory_id in trajectories:
            for detection in detections:
                if generator.random() < 0.4:
                    candidates[(trajectory_id, detection.detection_id)] = (
                        generator.randint(0, 10) / 100,
                        generator.randint(0, 10) / 100,
                        generator.randint(0, 10) / 100,
                        generator.randint(1, 10),
                    )
        actual = _minimum_cost_maximum_matching(trajectories, detections, candidates, config=config)
        actual_cost = sum(
            _association_cost(candidates[(trajectory_id, detection.detection_id)], config)
            for trajectory_id, detection in actual
        )
        brute_count, brute_cost = _brute_force_matching_objective(
            trajectories,
            [item.detection_id for item in detections],
            candidates,
            config,
        )
        assert len(actual) == brute_count, case
        assert actual_cost == pytest.approx(brute_cost, abs=1e-12), case
        permuted = _minimum_cost_maximum_matching(
            list(reversed(trajectories)),
            list(reversed(detections)),
            dict(reversed(list(candidates.items()))),
            config=config,
        )
        assert actual == permuted, case


def test_sparse_matching_2000_chain_has_linear_residual_structure() -> None:
    prototype = _frame("video1", 10).detections[0]
    trajectories = [f"traj-{index:04d}" for index in range(2_000)]
    detections = [
        prototype.model_copy(update={"detection_id": f"det-{index:04d}", "source_order": index})
        for index in range(2_000)
    ]
    candidates = {
        (trajectories[index], detections[index].detection_id): (0.0, 0.0, 0.0, 1)
        for index in range(2_000)
    }
    candidates.update(
        {
            (trajectories[index], detections[index + 1].detection_id): (0.01, 0.0, 0.0, 1)
            for index in range(1_999)
        }
    )
    structure = sparse_matching_structure(2_000, 2_000, 3_999)
    assert structure == {
        "nodes": 4_002,
        "forward_edges": 7_999,
        "residual_arcs": 15_998,
        "dense_cells": 0,
    }
    matches = _minimum_cost_maximum_matching(
        trajectories, detections, candidates, config=TrackingConfig()
    )
    assert len(matches) == 2_000


def test_sparse_component_and_active_trajectory_caps_fail_closed() -> None:
    prototype = _frame("video1", 10).detections[0]
    detections = [
        prototype.model_copy(update={"detection_id": f"det-{index}", "source_order": index})
        for index in range(2)
    ]
    candidates = {
        ("traj-0", "det-0"): (0.0, 0.0, 0.0, 1),
        ("traj-0", "det-1"): (0.0, 0.0, 0.0, 1),
        ("traj-1", "det-1"): (0.0, 0.0, 0.0, 1),
    }
    with pytest.raises(ValueError, match="maximum_candidate_edges_per_component"):
        _minimum_cost_maximum_matching(
            ["traj-0", "traj-1"],
            detections,
            candidates,
            config=TrackingConfig(maximum_candidate_edges_per_component=2),
        )

    frame = _frame("video1", 0)
    frame.detections.append(
        frame.detections[0].model_copy(
            update={"detection_id": "video1:0:det-0001", "source_order": 1}
        )
    )
    with pytest.raises(ValueError, match="maximum_active_trajectories"):
        build_trajectories(
            [frame],
            config=TrackingConfig(
                maximum_detections_per_frame=2,
                maximum_active_trajectories=1,
            ),
        )


def test_candidate_evaluation_cap_is_checked_before_any_pair_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _frame("video1", 0)
    second = _frame("video1", 10)
    for frame in (first, second):
        frame.detections.append(
            frame.detections[0].model_copy(
                update={
                    "detection_id": f"{frame.frame_uid}:det-0001",
                    "source_order": 1,
                }
            )
        )
    calls = 0

    def association(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return (0.0, 0.0, 0.0, 1)

    monkeypatch.setattr(tracking_module, "_association_metrics", association)
    with pytest.raises(ValueError, match="maximum_candidate_evaluations_per_frame"):
        build_trajectories(
            [first, second],
            config=TrackingConfig(maximum_candidate_evaluations_per_frame=3),
        )
    assert calls == 0


def test_total_candidate_edge_cap_is_per_frame_not_per_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _frame("video1", 0)
    second = _frame("video1", 10)
    for frame in (first, second):
        frame.detections.append(
            frame.detections[0].model_copy(
                update={
                    "detection_id": f"{frame.frame_uid}:det-0001",
                    "source_order": 1,
                }
            )
        )
    monkeypatch.setattr(
        tracking_module,
        "_association_metrics",
        lambda *_args, **_kwargs: (0.0, 0.0, 0.0, 1),
    )
    with pytest.raises(ValueError, match="maximum_candidate_edges_per_frame"):
        build_trajectories(
            [first, second],
            config=TrackingConfig(
                maximum_candidate_edges_per_frame=3,
                maximum_candidate_edges_per_component=10,
            ),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_frame_gap": 1.5},
        {"max_frame_gap": True},
        {"minimum_polygon_iou": float("nan")},
        {"maximum_centroid_distance_ratio": float("inf")},
        {"maximum_log_aspect_delta": float("-inf")},
        {"version": "aic26.ocr_tracking.future.v2"},
        {"sharpness_weight": 0.4000000005},
    ],
)
def test_tracking_config_rejects_non_finite_wrong_type_version_and_weight_sum(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        TrackingConfig(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"minimum_polygon_iou": Fraction(1, 2)},
        {"minimum_polygon_iou": __import__("numpy").float64(0.5)},
        {"maximum_representatives": __import__("numpy").int64(3)},
    ],
)
def test_tracking_config_only_accepts_json_native_numbers(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrackingConfig(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"minimum_height_px": True},
        {"minimum_height_px": 96.0},
        {"png_compress_level": False},
        {"png_compress_level": 9.0},
        {"horizontal_padding_ratio": Fraction(1, 10)},
        {"vertical_padding_ratio": float("nan")},
        {"vertical_padding_ratio": float("inf")},
    ],
)
def test_crop_config_rejects_non_json_non_finite_and_non_exact_integers(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        CropConfig(**overrides)


def test_natural_key_handles_extreme_digit_runs_without_integer_conversion() -> None:
    values = ["video" + "9" * 100_000, "video1", "video" + "0" * 100_000 + "1"]
    ordered = sorted(values, key=tracking_module.natural_key)
    assert ordered[:2] == ["video1", "video" + "0" * 100_000 + "1"]
