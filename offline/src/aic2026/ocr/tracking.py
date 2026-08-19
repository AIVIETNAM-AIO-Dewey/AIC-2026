"""Deterministic appearance-aware OCR polygon tracking and crop selection."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from aic2026.contracts import (
    OcrDetection,
    OcrDetectionFrameRecord,
    OcrTrajectoryRecord,
    RepresentativeCropBinding,
    TrajectoryMember,
)

from .geometry import Point, Quad, visual_hash_distance


def natural_key(value: str) -> tuple[tuple[tuple[Any, ...], ...], str]:
    """Natural numeric total ordering with the exact raw value as final tie-breaker."""

    return (
        tuple(
            (
                (0, len(part.lstrip("0")), part.lstrip("0") or "0", len(part))
                if part.isdigit()
                else (1, part.casefold())
            )
            for part in re.split(r"(\d+)", value)
            if part
        ),
        value,
    )


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    # Pilot-tunable until evaluated on a labeled temporal OCR corpus.
    max_frame_gap: int = 120
    minimum_polygon_iou: float = 0.15
    maximum_centroid_distance_ratio: float = 0.75
    minimum_area_ratio: float = 0.40
    maximum_log_aspect_delta: float = 0.45
    maximum_visual_hash_distance: float = 0.22
    sharpness_weight: float = 0.40
    detector_score_weight: float = 0.30
    resolution_weight: float = 0.15
    edge_weight: float = 0.15
    temporal_diversity_weight: float = 0.20
    maximum_representatives: int = 3
    maximum_frames_per_shard: int = 25_000
    maximum_detections_per_shard: int = 250_000
    maximum_detections_per_frame: int = 2_000
    maximum_active_trajectories: int = 20_000
    maximum_candidate_evaluations_per_frame: int = 2_000_000
    maximum_candidate_edges_per_frame: int = 250_000
    maximum_candidate_edges_per_component: int = 250_000
    version: str = "aic26.ocr_tracking.pilot.v3"

    def __post_init__(self) -> None:
        integer_fields = (
            self.max_frame_gap,
            self.maximum_representatives,
            self.maximum_frames_per_shard,
            self.maximum_detections_per_shard,
            self.maximum_detections_per_frame,
            self.maximum_active_trajectories,
            self.maximum_candidate_evaluations_per_frame,
            self.maximum_candidate_edges_per_frame,
            self.maximum_candidate_edges_per_component,
        )
        if any(type(value) is not int for value in integer_fields):
            raise ValueError("tracking counts and frame gap must be true integers")
        if (
            self.max_frame_gap < 1
            or self.maximum_representatives != 3
            or self.maximum_frames_per_shard < 1
            or self.maximum_detections_per_shard < 1
            or self.maximum_detections_per_frame < 1
            or self.maximum_active_trajectories < 1
            or self.maximum_candidate_evaluations_per_frame < 1
            or self.maximum_candidate_edges_per_frame < 1
            or self.maximum_candidate_edges_per_component < 1
        ):
            raise ValueError(
                "tracking counts/gap must be positive and representative cap must be 3"
            )
        if self.version != "aic26.ocr_tracking.pilot.v3":
            raise ValueError("unsupported tracking algorithm version")
        bounded = (
            self.minimum_polygon_iou,
            self.maximum_centroid_distance_ratio,
            self.minimum_area_ratio,
            self.maximum_visual_hash_distance,
            self.sharpness_weight,
            self.detector_score_weight,
            self.resolution_weight,
            self.edge_weight,
            self.temporal_diversity_weight,
        )
        numeric = (*bounded, self.maximum_log_aspect_delta)
        if any(
            type(value) not in (int, float) or not math.isfinite(float(value)) for value in numeric
        ):
            raise ValueError("tracking thresholds and weights must be finite numbers")
        if any(not 0 <= value <= 1 for value in bounded):
            raise ValueError("tracking ratios and weights must be inside [0, 1]")
        if self.maximum_log_aspect_delta < 0:
            raise ValueError("aspect delta must be non-negative")
        quality_sum = (
            self.sharpness_weight
            + self.detector_score_weight
            + self.resolution_weight
            + self.edge_weight
        )
        if not math.isclose(quality_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("representative quality weights must sum to one")

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def resource_limits_sha256(self) -> str:
        limits = {
            "maximum_active_trajectories": self.maximum_active_trajectories,
            "maximum_candidate_edges_per_frame": self.maximum_candidate_edges_per_frame,
            "maximum_candidate_edges_per_component": self.maximum_candidate_edges_per_component,
            "maximum_candidate_evaluations_per_frame": (
                self.maximum_candidate_evaluations_per_frame
            ),
            "maximum_detections_per_frame": self.maximum_detections_per_frame,
            "maximum_detections_per_shard": self.maximum_detections_per_shard,
            "maximum_frames_per_shard": self.maximum_frames_per_shard,
        }
        payload = json.dumps(limits, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _signed_area(polygon: list[Point] | Quad) -> float:
    return (
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
        / 2.0
    )


def _intersection(first: Point, second: Point, clip_first: Point, clip_second: Point) -> Point:
    x1, y1 = first
    x2, y2 = second
    x3, y3 = clip_first
    x4, y4 = clip_second
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) <= 1e-12:
        return second
    determinant_first = x1 * y2 - y1 * x2
    determinant_second = x3 * y4 - y3 * x4
    return (
        (determinant_first * (x3 - x4) - (x1 - x2) * determinant_second) / denominator,
        (determinant_first * (y3 - y4) - (y1 - y2) * determinant_second) / denominator,
    )


def _convex_intersection(subject: Quad, clip: Quad) -> list[Point]:
    output = list(subject)
    orientation = 1.0 if _signed_area(clip) >= 0 else -1.0
    for index, clip_first in enumerate(clip):
        clip_second = clip[(index + 1) % 4]
        source = output
        output = []
        if not source:
            break

        def inside(
            point: Point, edge_start: Point = clip_first, edge_end: Point = clip_second
        ) -> bool:
            cross = (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - (
                edge_end[1] - edge_start[1]
            ) * (point[0] - edge_start[0])
            return orientation * cross >= -1e-9

        previous = source[-1]
        previous_inside = inside(previous)
        for current in source:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(_intersection(previous, current, clip_first, clip_second))
                output.append(current)
            elif previous_inside:
                output.append(_intersection(previous, current, clip_first, clip_second))
            previous = current
            previous_inside = current_inside
    return output


def polygon_iou(first: Quad, second: Quad) -> float:
    first_area = abs(_signed_area(first))
    second_area = abs(_signed_area(second))
    intersection = _convex_intersection(first, second)
    intersection_area = abs(_signed_area(intersection)) if len(intersection) >= 3 else 0.0
    union = first_area + second_area - intersection_area
    return intersection_area / union if union > 0 else 0.0


def _shape(quad: Quad) -> tuple[float, float, float, Point]:
    area = abs(_signed_area(quad))
    horizontal = (math.dist(quad[0], quad[1]) + math.dist(quad[3], quad[2])) / 2.0
    vertical = (math.dist(quad[0], quad[3]) + math.dist(quad[1], quad[2])) / 2.0
    aspect = horizontal / max(vertical, 1e-9)
    diagonal = math.hypot(horizontal, vertical)
    centroid = (
        sum(point[0] for point in quad) / 4.0,
        sum(point[1] for point in quad) / 4.0,
    )
    return area, aspect, diagonal, centroid


def _association_metrics(
    previous: TrajectoryMember, current: OcrDetection, *, frame_idx: int, config: TrackingConfig
) -> tuple[float, float, float, int] | None:
    gap = frame_idx - previous.frame_idx
    if gap <= 0 or gap > config.max_frame_gap:
        return None
    previous_quad = previous.polygon_xy.points
    current_quad = current.polygon_xy.points
    previous_area, previous_aspect, previous_diagonal, previous_centroid = _shape(previous_quad)
    current_area, current_aspect, current_diagonal, current_centroid = _shape(current_quad)
    area_ratio = min(previous_area, current_area) / max(previous_area, current_area)
    aspect_delta = abs(math.log(max(previous_aspect, 1e-9) / max(current_aspect, 1e-9)))
    if area_ratio < config.minimum_area_ratio or aspect_delta > config.maximum_log_aspect_delta:
        return None
    iou = polygon_iou(previous_quad, current_quad)
    centroid_ratio = math.dist(previous_centroid, current_centroid) / max(
        (previous_diagonal + current_diagonal) / 2.0, 1e-9
    )
    if iou < config.minimum_polygon_iou and centroid_ratio > config.maximum_centroid_distance_ratio:
        return None
    appearance = visual_hash_distance(previous.crop.visual_hash, current.crop.visual_hash)
    if appearance > config.maximum_visual_hash_distance:
        return None
    return appearance, 1.0 - iou, centroid_ratio, gap


def _member(frame: OcrDetectionFrameRecord, detection: OcrDetection) -> TrajectoryMember:
    assert frame.source_image_sha256 is not None
    return TrajectoryMember(
        video_id=frame.video_id,
        frame_uid=frame.frame_uid,
        frame_idx=frame.frame_idx,
        pts_time_s=frame.pts_time_s,
        frame_relpath=frame.frame_relpath,
        source_image_sha256=frame.source_image_sha256,
        canonical_image_sha256=frame.canonical_image_sha256,
        source_width=frame.width,
        source_height=frame.height,
        detection_id=detection.detection_id,
        detector_score=detection.detector_score,
        polygon_xy=detection.polygon_xy,
        crop=detection.crop,
    )


def validate_and_sort_detection_frames(
    records: list[OcrDetectionFrameRecord],
) -> list[OcrDetectionFrameRecord]:
    identities = [record.frame_uid for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate frame_uid in detection artifact")
    frame_keys = [(record.video_id, record.frame_idx) for record in records]
    if len(frame_keys) != len(set(frame_keys)):
        raise ValueError("duplicate video_id/frame_idx in detection artifact")
    groups: dict[str, list[OcrDetectionFrameRecord]] = {}
    for record in records:
        groups.setdefault(record.video_id, []).append(record)
    ordered: list[OcrDetectionFrameRecord] = []
    for video_id in sorted(groups, key=natural_key):
        ordered.extend(
            sorted(
                groups[video_id],
                key=lambda item: (item.frame_idx, natural_key(item.frame_uid)),
            )
        )
    return ordered


def _association_cost(metrics: tuple[float, float, float, int], config: TrackingConfig) -> float:
    appearance, inverse_iou, centroid_ratio, gap = metrics
    appearance_component = (
        appearance / config.maximum_visual_hash_distance
        if config.maximum_visual_hash_distance > 0
        else 0.0
    )
    centroid_component = (
        min(centroid_ratio / config.maximum_centroid_distance_ratio, 1.0)
        if config.maximum_centroid_distance_ratio > 0
        else 0.0
    )
    return (
        min(appearance_component, 1.0)
        + inverse_iou
        + centroid_component
        + gap / config.max_frame_gap
    ) / 4.0


def _minimum_cost_maximum_matching(
    trajectory_ids: list[str],
    detections: list[OcrDetection],
    candidates: dict[tuple[str, str], tuple[float, float, float, int]],
    *,
    config: TrackingConfig,
    metrics: dict[str, int] | None = None,
) -> list[tuple[str, OcrDetection]]:
    """Solve sparse connected components with max-cardinality/min-cost objective."""

    if not trajectory_ids or not detections or not candidates:
        return []
    if len(candidates) > config.maximum_candidate_edges_per_frame:
        raise ValueError("candidate edges exceed maximum_candidate_edges_per_frame")
    trajectory_set = set(trajectory_ids)
    detections_by_id = {item.detection_id: item for item in detections}
    edges = {
        edge: metrics
        for edge, metrics in candidates.items()
        if edge[0] in trajectory_set and edge[1] in detections_by_id
    }
    if not edges:
        return []
    trajectories_to_detections: dict[str, set[str]] = {}
    detections_to_trajectories: dict[str, set[str]] = {}
    row_edges: dict[str, list[tuple[str, tuple[float, float, float, int]]]] = {}
    for (trajectory_id, detection_id), association in edges.items():
        trajectories_to_detections.setdefault(trajectory_id, set()).add(detection_id)
        detections_to_trajectories.setdefault(detection_id, set()).add(trajectory_id)
        row_edges.setdefault(trajectory_id, []).append((detection_id, association))
    for trajectory_id in row_edges:
        row_edges[trajectory_id].sort(key=lambda item: natural_key(item[0]))

    remaining = set(trajectories_to_detections)
    components: list[tuple[list[str], list[OcrDetection]]] = []
    while remaining:
        seed = min(remaining, key=natural_key)
        row_component: set[str] = set()
        column_component: set[str] = set()
        row_frontier = [seed]
        while row_frontier:
            trajectory_id = row_frontier.pop()
            if trajectory_id in row_component:
                continue
            row_component.add(trajectory_id)
            remaining.discard(trajectory_id)
            for detection_id in sorted(trajectories_to_detections[trajectory_id], key=natural_key):
                if detection_id in column_component:
                    continue
                column_component.add(detection_id)
                row_frontier.extend(
                    sorted(
                        detections_to_trajectories[detection_id] - row_component,
                        key=natural_key,
                        reverse=True,
                    )
                )
        component_edges = sum(
            len(trajectories_to_detections[trajectory_id]) for trajectory_id in row_component
        )
        if component_edges > config.maximum_candidate_edges_per_component:
            raise ValueError("candidate component exceeds maximum_candidate_edges_per_component")
        if metrics is not None:
            metrics["maximum_candidate_component_size"] = max(
                metrics.get("maximum_candidate_component_size", 0), component_edges
            )
        components.append(
            (
                sorted(row_component, key=natural_key),
                [detections_by_id[key] for key in sorted(column_component, key=natural_key)],
            )
        )

    output: list[tuple[str, OcrDetection]] = []
    for rows, columns in components:
        output.extend(_solve_sparse_component(rows, columns, row_edges, config=config))
    return sorted(output, key=lambda item: natural_key(item[0]))


def sparse_matching_structure(
    row_count: int, column_count: int, candidate_edge_count: int
) -> dict[str, int]:
    """Describe the linear sparse residual allocation used by the solver."""

    return {
        "nodes": row_count + column_count + 2,
        "forward_edges": row_count + candidate_edge_count + column_count,
        "residual_arcs": 2 * (row_count + candidate_edge_count + column_count),
        "dense_cells": 0,
    }


def _solve_sparse_component(
    rows: list[str],
    columns: list[OcrDetection],
    row_edges: dict[str, list[tuple[str, tuple[float, float, float, int]]]],
    *,
    config: TrackingConfig,
) -> list[tuple[str, OcrDetection]]:
    """Sparse max-cardinality matching with minimum 1e-12-quantized cost."""

    row_count = len(rows)
    column_count = len(columns)
    source = row_count + column_count
    sink = source + 1
    node_count = sink + 1
    graph: list[list[list[int]]] = [[] for _ in range(node_count)]

    def add_edge(start: int, end: int, capacity: int, cost: int) -> None:
        forward = [end, len(graph[end]), capacity, cost]
        reverse = [start, len(graph[start]), 0, -cost]
        graph[start].append(forward)
        graph[end].append(reverse)

    for row_index in range(row_count):
        add_edge(source, row_index, 1, 0)
    for column_index in range(column_count):
        add_edge(row_count + column_index, sink, 1, 0)

    sparse_edges: list[tuple[int, int, int]] = []
    detection_index = {item.detection_id: index for index, item in enumerate(columns)}
    for row_index, trajectory_id in enumerate(rows):
        for detection_id, association in row_edges[trajectory_id]:
            if detection_id not in detection_index:
                continue
            quantized = round(_association_cost(association, config) * 10**12)
            sparse_edges.append((row_index, detection_index[detection_id], quantized))
    tie_scale = min(row_count, column_count) * max(1, len(sparse_edges)) + 1
    for edge_rank, (row_index, column_index, quantized) in enumerate(sparse_edges):
        add_edge(
            row_index,
            row_count + column_index,
            1,
            quantized * tie_scale + edge_rank,
        )

    potentials = [0] * node_count
    infinity = 10**40
    while True:
        distances = [infinity] * node_count
        previous: list[tuple[int, int] | None] = [None] * node_count
        distances[source] = 0
        queue: list[tuple[int, int]] = [(0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                target, _, capacity, cost = edge
                if not capacity:
                    continue
                candidate_distance = distance + cost + potentials[node] - potentials[target]
                candidate_previous = (node, edge_index)
                if candidate_distance < distances[target]:
                    distances[target] = candidate_distance
                    previous[target] = candidate_previous
                    heapq.heappush(queue, (candidate_distance, target))
        if previous[sink] is None:
            break
        for node, distance in enumerate(distances):
            if distance < infinity:
                potentials[node] += distance
        node = sink
        while node != source:
            prior, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[prior][edge_index]
            edge[2] -= 1
            graph[node][edge[1]][2] += 1
            node = prior

    output: list[tuple[str, OcrDetection]] = []
    for row_index, trajectory_id in enumerate(rows):
        for edge in graph[row_index]:
            target, _, capacity, _ = edge
            if row_count <= target < row_count + column_count and capacity == 0:
                output.append((trajectory_id, columns[target - row_count]))
                break
    return output


def build_trajectories(
    records: list[OcrDetectionFrameRecord],
    *,
    config: TrackingConfig,
    metrics: dict[str, int] | None = None,
) -> list[OcrTrajectoryRecord]:
    ordered = validate_and_sort_detection_frames(records)
    if not ordered:
        raise ValueError("detection artifact is empty")
    run_ids = {record.run_id for record in ordered}
    config_hashes = {record.config_sha256 for record in ordered}
    revisions = {record.detector_revision for record in ordered}
    tree_hashes = {record.detector_tree_sha256 for record in ordered}
    runtime_hashes = {record.runtime_identity_sha256 for record in ordered}
    if (
        len(run_ids) != 1
        or len(config_hashes) != 1
        or len(revisions) != 1
        or len(tree_hashes) != 1
        or len(runtime_hashes) != 1
    ):
        raise ValueError("detection artifact run/config/model identity is inconsistent")
    detection_count = sum(len(record.detections) for record in ordered)
    if len(ordered) > config.maximum_frames_per_shard:
        raise ValueError("detection shard exceeds maximum_frames_per_shard")
    if detection_count > config.maximum_detections_per_shard:
        raise ValueError("detection shard exceeds maximum_detections_per_shard")
    if any(len(record.detections) > config.maximum_detections_per_frame for record in ordered):
        raise ValueError("detection frame exceeds maximum_detections_per_frame")
    run_id = next(iter(run_ids))
    config_hash = next(iter(config_hashes))
    revision = next(iter(revisions))
    tree_hash = next(iter(tree_hashes))
    runtime_hash = next(iter(runtime_hashes))
    finished: list[OcrTrajectoryRecord] = []
    current_video: str | None = None
    active: dict[str, list[TrajectoryMember]] = {}
    next_index = 1
    if metrics is not None:
        metrics.setdefault("maximum_active_trajectories", 0)
        metrics.setdefault("maximum_candidate_component_size", 0)

    def flush() -> None:
        for trajectory_id in sorted(active, key=natural_key):
            members = active[trajectory_id]
            finished.append(
                OcrTrajectoryRecord(
                    run_id=run_id,
                    video_id=members[0].video_id,
                    trajectory_id=trajectory_id,
                    config_sha256=config_hash,
                    tracking_config_sha256=config.sha256,
                    detector_revision=revision,
                    detector_tree_sha256=tree_hash,
                    runtime_identity_sha256=runtime_hash,
                    members=members,
                )
            )
        active.clear()

    for frame in ordered:
        if frame.video_id != current_video:
            if current_video is not None:
                flush()
            current_video = frame.video_id
            next_index = 1
        stale = [
            trajectory_id
            for trajectory_id, members in active.items()
            if frame.frame_idx - members[-1].frame_idx > config.max_frame_gap
        ]
        for trajectory_id in sorted(stale, key=natural_key):
            members = active.pop(trajectory_id)
            finished.append(
                OcrTrajectoryRecord(
                    run_id=run_id,
                    video_id=frame.video_id,
                    trajectory_id=trajectory_id,
                    config_sha256=config_hash,
                    tracking_config_sha256=config.sha256,
                    detector_revision=revision,
                    detector_tree_sha256=tree_hash,
                    runtime_identity_sha256=runtime_hash,
                    members=members,
                )
            )

        if len(active) > config.maximum_active_trajectories:
            raise ValueError("active trajectories exceed maximum_active_trajectories")

        candidate_evaluations = len(active) * len(frame.detections)
        if candidate_evaluations > config.maximum_candidate_evaluations_per_frame:
            raise ValueError("candidate evaluations exceed maximum_candidate_evaluations_per_frame")

        candidate_metrics: dict[tuple[str, str], tuple[float, float, float, int]] = {}
        for detection in sorted(frame.detections, key=lambda item: item.source_order):
            for trajectory_id, members in active.items():
                association = _association_metrics(
                    members[-1], detection, frame_idx=frame.frame_idx, config=config
                )
                if association is not None:
                    if len(candidate_metrics) >= config.maximum_candidate_edges_per_frame:
                        raise ValueError("candidate edges exceed maximum_candidate_edges_per_frame")
                    candidate_metrics[(trajectory_id, detection.detection_id)] = association
        assigned_detections: set[str] = set()
        matches = _minimum_cost_maximum_matching(
            list(active), frame.detections, candidate_metrics, config=config, metrics=metrics
        )
        for trajectory_id, detection in matches:
            active[trajectory_id].append(_member(frame, detection))
            assigned_detections.add(detection.detection_id)
        for detection in sorted(frame.detections, key=lambda item: item.source_order):
            if detection.detection_id in assigned_detections:
                continue
            trajectory_id = f"{frame.video_id}:traj-{next_index:06d}"
            next_index += 1
            active[trajectory_id] = [_member(frame, detection)]
        if len(active) > config.maximum_active_trajectories:
            raise ValueError("active trajectories exceed maximum_active_trajectories")
        if metrics is not None:
            metrics["maximum_active_trajectories"] = max(
                metrics["maximum_active_trajectories"], len(active)
            )
    flush()
    return sorted(
        finished,
        key=lambda item: (natural_key(item.video_id), natural_key(item.trajectory_id)),
    )


def _quality_scores(members: list[TrajectoryMember], config: TrackingConfig) -> dict[str, float]:
    maximum_sharpness = max((member.crop.sharpness for member in members), default=0.0)
    resolutions = [member.crop.output_width * member.crop.output_height for member in members]
    maximum_resolution = max(resolutions, default=1)
    scores: dict[str, float] = {}
    for member, resolution in zip(members, resolutions, strict=True):
        sharpness = member.crop.sharpness / maximum_sharpness if maximum_sharpness > 0 else 0.0
        score = (
            config.sharpness_weight * sharpness
            + config.detector_score_weight * member.detector_score
            + config.resolution_weight * (resolution / maximum_resolution)
            + config.edge_weight * (1.0 - member.crop.edge_truncation_penalty)
        )
        scores[member.detection_id] = min(max(score, 0.0), 1.0)
    return scores


def select_representatives(
    trajectories: list[OcrTrajectoryRecord], *, config: TrackingConfig
) -> list[RepresentativeCropBinding]:
    output: list[RepresentativeCropBinding] = []
    seen: set[str] = set()
    for trajectory in sorted(
        trajectories,
        key=lambda item: (natural_key(item.video_id), natural_key(item.trajectory_id)),
    ):
        if trajectory.tracking_config_sha256 != config.sha256:
            raise ValueError("trajectory tracking config identity drift")
        if trajectory.trajectory_id in seen:
            raise ValueError("duplicate trajectory_id")
        seen.add(trajectory.trajectory_id)
        members = trajectory.members
        quality = _quality_scores(members, config)
        selected: list[TrajectoryMember] = []
        target = min(len(members), config.maximum_representatives)
        span = max(1, members[-1].frame_idx - members[0].frame_idx)
        diversity_by_id: dict[str, float] = {}
        while len(selected) < target:
            ranked: list[tuple[tuple[Any, ...], TrajectoryMember, float]] = []
            for member in members:
                if member in selected:
                    continue
                diversity = (
                    0.0
                    if not selected
                    else min(abs(member.frame_idx - item.frame_idx) for item in selected) / span
                )
                adjusted = (1.0 - config.temporal_diversity_weight) * quality[
                    member.detection_id
                ] + config.temporal_diversity_weight * diversity
                resolution = member.crop.output_width * member.crop.output_height
                # Every component and the final natural identity are explicit tie-breakers.
                key = (
                    -adjusted,
                    -quality[member.detection_id],
                    -member.detector_score,
                    -member.crop.sharpness,
                    -resolution,
                    member.crop.edge_truncation_penalty,
                    member.frame_idx,
                    natural_key(member.detection_id),
                )
                ranked.append((key, member, diversity))
            _, chosen, diversity = min(ranked, key=lambda item: item[0])
            selected.append(chosen)
            diversity_by_id[chosen.detection_id] = diversity
        for rank, member in enumerate(selected, start=1):
            output.append(
                RepresentativeCropBinding(
                    run_id=trajectory.run_id,
                    video_id=trajectory.video_id,
                    trajectory_id=trajectory.trajectory_id,
                    representative_rank=rank,
                    config_sha256=trajectory.config_sha256,
                    tracking_config_sha256=trajectory.tracking_config_sha256,
                    detector_revision=trajectory.detector_revision,
                    detector_tree_sha256=trajectory.detector_tree_sha256,
                    runtime_identity_sha256=trajectory.runtime_identity_sha256,
                    frame_uid=member.frame_uid,
                    frame_idx=member.frame_idx,
                    frame_relpath=member.frame_relpath,
                    source_image_sha256=member.source_image_sha256,
                    canonical_image_sha256=member.canonical_image_sha256,
                    source_width=member.source_width,
                    source_height=member.source_height,
                    detection_id=member.detection_id,
                    detector_score=member.detector_score,
                    polygon_xy=member.polygon_xy,
                    crop=member.crop,
                    quality_score=quality[member.detection_id],
                    temporal_diversity_score=diversity_by_id[member.detection_id],
                )
            )
    return output
