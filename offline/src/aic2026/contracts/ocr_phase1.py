"""Versioned contracts for deterministic detector-only OCR Phase 1 artifacts."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .models import FrameRef, StrictModel
from .paths import require_safe_relative_path

SHA256_PATTERN = r"^[0-9a-f]{64}$"
VISUAL_HASH_PATTERN = r"^[0-9a-f]{16}$"


def _validate_quad_point_set(
    points: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ],
) -> None:
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError("quadrilateral coordinates must be finite")
    if len(set(points)) != 4:
        raise ValueError("quadrilateral points must be unique")

    def cross(origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    ordered = sorted(points)
    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-6:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-6:
            upper.pop()
        upper.append(point)
    if len(lower[:-1] + upper[:-1]) != 4:
        raise ValueError("quadrilateral must be finite, convex and non-degenerate")


def canonical_quad_points(
    points: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return a deterministic cyclic order using only hull adjacency and winding.

    The monotone hull is positive-winding in image coordinates.  Starting at
    the top-most (then left-most) hull vertex makes cyclic/reversed detector
    serializations identical without inventing non-adjacent left/right pairs.
    A cyclic 90-degree choice for negatively sloped text is intentional; the
    crop algorithm applies PaddleX's vertical-crop rotation convention.
    """

    _validate_quad_point_set(points)

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    ordered = sorted(points)
    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-6:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-6:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    start = min(range(4), key=lambda index: (hull[index][1], hull[index][0]))
    rotated = hull[start:] + hull[:start]
    return tuple(rotated)  # type: ignore[return-value]


class RawQuadGeometry(StrictModel):
    """Four raw PaddleX vertices; cyclic start and winding are intentionally preserved."""

    points: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ]

    @field_validator("points", mode="before")
    @classmethod
    def accept_json_points(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(tuple(point) if isinstance(point, list) else point for point in value)
        return value

    @model_validator(mode="after")
    def finite_convex(self) -> RawQuadGeometry:
        _validate_quad_point_set(self.points)
        return self


class QuadGeometry(StrictModel):
    """Canonical PaddleX crop order TL, TR, BR, BL in source-frame pixels."""

    point_order: Literal["tl_tr_br_bl"] = "tl_tr_br_bl"
    points: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ]

    @field_validator("points", mode="before")
    @classmethod
    def accept_json_points(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(tuple(point) if isinstance(point, list) else point for point in value)
        return value

    @model_validator(mode="after")
    def finite_non_degenerate(self) -> QuadGeometry:
        _validate_quad_point_set(self.points)
        if self.points != canonical_quad_points(self.points):
            raise ValueError("quadrilateral must use canonical hull-adjacent order")
        signed_area = (
            sum(
                self.points[index][0] * self.points[(index + 1) % 4][1]
                - self.points[(index + 1) % 4][0] * self.points[index][1]
                for index in range(4)
            )
            / 2
        )
        if signed_area <= 1e-6:
            raise ValueError("quadrilateral must be non-degenerate")
        crosses = []
        for index in range(4):
            previous = self.points[index - 1]
            current = self.points[index]
            following = self.points[(index + 1) % 4]
            crosses.append(
                (current[0] - previous[0]) * (following[1] - current[1])
                - (current[1] - previous[1]) * (following[0] - current[0])
            )
        if not all(value > 1e-6 for value in crosses):
            raise ValueError("quadrilateral must use clockwise image-space winding")
        return self


class CropProvenance(StrictModel):
    crop_config_sha256: str = Field(pattern=SHA256_PATTERN)
    algorithm: Literal["aic26.pil_quad_crop.v3"] = "aic26.pil_quad_crop.v3"
    perspective_resampling: Literal["bicubic"] = "bicubic"
    vertical_normalization_resampling: Literal["bicubic"] = "bicubic"
    png_compress_level: int = Field(default=9, ge=0, le=9)
    visual_hash_algorithm: Literal["dhash64-bilinear-v1"] = "dhash64-bilinear-v1"
    padded_polygon_xy: QuadGeometry
    perspective_width: int = Field(ge=1)
    perspective_height: int = Field(ge=1)
    output_width: int = Field(ge=1)
    output_height: int = Field(ge=1)
    rotation_quadrants_ccw: Literal[0, 1, 3] = 0
    png_sha256: str = Field(pattern=SHA256_PATTERN)
    visual_hash: str = Field(pattern=VISUAL_HASH_PATTERN)
    sharpness: float = Field(ge=0)
    edge_truncation_penalty: float = Field(ge=0, le=1)


class OcrDetection(StrictModel):
    detection_id: str = Field(min_length=1)
    source_order: int = Field(ge=0)
    polygon_raw_xy: RawQuadGeometry
    polygon_xy: QuadGeometry
    polygon_clamped: bool
    detector_score: float = Field(ge=0, le=1)
    crop: CropProvenance


class OcrDetectionFrameRecord(FrameRef):
    schema_version: Literal["aic26.ocr_detection_frame.v1"] = "aic26.ocr_detection_frame.v1"
    run_id: str = Field(min_length=1)
    detector_id: Literal["PP-OCRv6_small_det"] = "PP-OCRv6_small_det"
    detector_revision: str = Field(pattern=SHA256_PATTERN)
    detector_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    runtime_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    canonical_image_sha256: str = Field(pattern=SHA256_PATTERN)
    detections: list[OcrDetection]

    @model_validator(mode="after")
    def validate_detections(self) -> OcrDetectionFrameRecord:
        if self.source_image_sha256 is None:
            raise ValueError("detection records require source image SHA-256")
        require_safe_relative_path(self.frame_relpath, field_name="frame_relpath")
        ids = [item.detection_id for item in self.detections]
        orders = [item.source_order for item in self.detections]
        if len(ids) != len(set(ids)) or len(orders) != len(set(orders)):
            raise ValueError("detection identity/order must be unique within a frame")
        for item in self.detections:
            expected = f"{self.frame_uid}:det-{item.source_order:04d}"
            if item.detection_id != expected:
                raise ValueError(f"detection_id must equal {expected!r}")
            for x, y in item.polygon_xy.points:
                if not (0 <= x <= self.width - 1 and 0 <= y <= self.height - 1):
                    raise ValueError("detection polygon exceeds source frame bounds")
        return self


class TrajectoryMember(StrictModel):
    video_id: str = Field(min_length=1)
    frame_uid: str = Field(min_length=3)
    frame_idx: int = Field(ge=0)
    pts_time_s: float = Field(ge=0)
    frame_relpath: str = Field(min_length=1)
    source_image_sha256: str = Field(pattern=SHA256_PATTERN)
    canonical_image_sha256: str = Field(pattern=SHA256_PATTERN)
    source_width: int = Field(ge=1)
    source_height: int = Field(ge=1)
    detection_id: str = Field(min_length=1)
    detector_score: float = Field(ge=0, le=1)
    polygon_xy: QuadGeometry
    crop: CropProvenance

    @model_validator(mode="after")
    def validate_frame_identity(self) -> TrajectoryMember:
        if self.frame_uid != f"{self.video_id}:{self.frame_idx}":
            raise ValueError("trajectory member frame_uid is inconsistent")
        require_safe_relative_path(self.frame_relpath, field_name="frame_relpath")
        return self


class OcrTrajectoryRecord(StrictModel):
    schema_version: Literal["aic26.ocr_trajectory.v1"] = "aic26.ocr_trajectory.v1"
    run_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    tracking_config_sha256: str = Field(pattern=SHA256_PATTERN)
    detector_id: Literal["PP-OCRv6_small_det"] = "PP-OCRv6_small_det"
    detector_revision: str = Field(pattern=SHA256_PATTERN)
    detector_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    runtime_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    members: list[TrajectoryMember] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_members(self) -> OcrTrajectoryRecord:
        if any(member.video_id != self.video_id for member in self.members):
            raise ValueError("trajectory cannot span video_id values")
        identities = [member.detection_id for member in self.members]
        if len(identities) != len(set(identities)):
            raise ValueError("trajectory detection identities must be unique")
        order = [(member.frame_idx, member.detection_id) for member in self.members]
        if order != sorted(order):
            raise ValueError("trajectory members must be in deterministic frame order")
        return self


class RepresentativeCropBinding(StrictModel):
    schema_version: Literal["aic26.ocr_representative_crop.v1"] = "aic26.ocr_representative_crop.v1"
    run_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    representative_rank: int = Field(ge=1, le=3)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    tracking_config_sha256: str = Field(pattern=SHA256_PATTERN)
    detector_id: Literal["PP-OCRv6_small_det"] = "PP-OCRv6_small_det"
    detector_revision: str = Field(pattern=SHA256_PATTERN)
    detector_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    runtime_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    frame_uid: str = Field(min_length=3)
    frame_idx: int = Field(ge=0)
    frame_relpath: str = Field(min_length=1)
    source_image_sha256: str = Field(pattern=SHA256_PATTERN)
    canonical_image_sha256: str = Field(pattern=SHA256_PATTERN)
    source_width: int = Field(ge=1)
    source_height: int = Field(ge=1)
    detection_id: str = Field(min_length=1)
    detector_score: float = Field(ge=0, le=1)
    polygon_xy: QuadGeometry
    crop: CropProvenance
    quality_score: float = Field(ge=0, le=1)
    temporal_diversity_score: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_frame_identity(self) -> RepresentativeCropBinding:
        if self.frame_uid != f"{self.video_id}:{self.frame_idx}":
            raise ValueError("representative frame_uid is inconsistent")
        require_safe_relative_path(self.frame_relpath, field_name="frame_relpath")
        return self


class OcrPhase1Receipt(StrictModel):
    schema_version: Literal["aic26.ocr_phase1_receipt.v1"] = "aic26.ocr_phase1_receipt.v1"
    run_id: str = Field(min_length=1)
    stage: Literal["detect_crop", "track", "select_representatives"]
    status: Literal["running", "completed"]
    input_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    detector_id: Literal["PP-OCRv6_small_det"] = "PP-OCRv6_small_det"
    detector_revision: str = Field(pattern=SHA256_PATTERN)
    detector_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    runtime_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    resource_limits_sha256: str = Field(pattern=SHA256_PATTERN)
    shard_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    shard_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    trust_boundary: Literal["integrity_metadata_not_a_signature"] = (
        "integrity_metadata_not_a_signature"
    )
    record_counts: dict[str, int]
    output_sha256: str = Field(pattern=SHA256_PATTERN)
    committed_bytes: int = Field(ge=0)
    committed_records: int = Field(ge=0)
    committed_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def completion_is_commit_marker(self) -> OcrPhase1Receipt:
        if self.output_sha256 != self.committed_sha256:
            raise ValueError("receipt output and committed hashes must agree")
        if any(value < 0 for value in self.record_counts.values()):
            raise ValueError("receipt record counts cannot be negative")
        required_counts = {
            "detect_crop": {"frames", "detections"},
            "track": {"trajectories", "members"},
            "select_representatives": {"representatives", "trajectories"},
        }[self.stage]
        if set(self.record_counts) != required_counts:
            raise ValueError(f"{self.stage} receipt has invalid record count fields")
        if self.status == "running" and self.stage != "detect_crop":
            raise ValueError("only detect/crop supports receipt-authenticated partial output")
        primary_key = {
            "detect_crop": "frames",
            "track": "trajectories",
            "select_representatives": "representatives",
        }[self.stage]
        primary_count = self.record_counts[primary_key]
        if self.committed_records != primary_count:
            raise ValueError("receipt committed record count is inconsistent")
        if self.stage == "detect_crop" and (
            self.shard_manifest_sha256 != self.input_artifact_sha256
        ):
            raise ValueError("detect receipt must bind its exact shard manifest input")
        return self


class OcrFrameShard(StrictModel):
    shard_id: str = Field(min_length=1, pattern=r"^shard-[0-9]{6}$")
    manifest_relpath: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    video_ids: list[str] = Field(min_length=1)
    frame_uids: list[str] = Field(min_length=1)
    frame_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_membership(self) -> OcrFrameShard:
        require_safe_relative_path(self.manifest_relpath, field_name="manifest_relpath")
        if self.video_ids != sorted(set(self.video_ids)):
            raise ValueError("shard video_ids must be sorted and unique")
        if (
            len(self.frame_uids) != self.frame_count
            or len(set(self.frame_uids)) != self.frame_count
        ):
            raise ValueError("shard frame membership/count is inconsistent")
        return self


class OcrGlobalShardManifest(StrictModel):
    schema_version: Literal["aic26.ocr_global_shards.v1"] = "aic26.ocr_global_shards.v1"
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    maximum_frames_per_shard: int = Field(ge=1)
    shards: list[OcrFrameShard] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_shards(self) -> OcrGlobalShardManifest:
        shard_ids = [item.shard_id for item in self.shards]
        if len(shard_ids) != len(set(shard_ids)):
            raise ValueError("global shard IDs must be unique")
        return self


class OcrGlobalShardReceipt(StrictModel):
    schema_version: Literal["aic26.ocr_global_shards_receipt.v1"] = (
        "aic26.ocr_global_shards_receipt.v1"
    )
    status: Literal["completed"] = "completed"
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    global_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    shard_count: int = Field(ge=1)
    frame_count: int = Field(ge=1)
    trust_boundary: Literal["integrity_metadata_not_a_signature"] = (
        "integrity_metadata_not_a_signature"
    )


class OcrCheckpointFile(StrictModel):
    """One immutable byte-bearing file stored inside a checkpoint bundle."""

    role: Literal[
        "source_manifest",
        "global_manifest",
        "global_receipt",
        "shard_manifest",
    ]
    bundle_relpath: str = Field(min_length=1)
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def safe_bundle_path(self) -> OcrCheckpointFile:
        require_safe_relative_path(self.bundle_relpath, field_name="bundle_relpath")
        return self


class OcrCheckpointArtifact(StrictModel):
    """A committed artifact prefix and the receipt authenticating that prefix."""

    stage: Literal["detect", "track", "select"]
    artifact_relpath: str = Field(min_length=1)
    payload_relpath: str = Field(min_length=1)
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=SHA256_PATTERN)
    committed_records: int = Field(ge=0)
    receipt_relpath: str = Field(min_length=1)
    receipt_sha256: str = Field(pattern=SHA256_PATTERN)
    receipt_status: Literal["running", "completed"]

    @model_validator(mode="after")
    def safe_paths_and_stage_status(self) -> OcrCheckpointArtifact:
        require_safe_relative_path(self.artifact_relpath, field_name="artifact_relpath")
        require_safe_relative_path(self.payload_relpath, field_name="payload_relpath")
        require_safe_relative_path(self.receipt_relpath, field_name="receipt_relpath")
        if self.receipt_status == "running" and self.stage != "detect":
            raise ValueError("only detection checkpoint artifacts may be running")
        return self


class OcrCheckpointCounts(StrictModel):
    frames: int = Field(ge=0)
    detections: int = Field(ge=0)
    trajectories: int = Field(ge=0)
    representatives: int = Field(ge=0)


class OcrPhase1Checkpoint(StrictModel):
    """Commit marker for a portable, fail-closed OCR Phase 1 checkpoint."""

    schema_version: Literal["aic26.ocr_phase1_checkpoint.v1"] = "aic26.ocr_phase1_checkpoint.v1"
    run_id: str = Field(min_length=1)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    detector_id: Literal["PP-OCRv6_small_det"] = "PP-OCRv6_small_det"
    detector_revision: str = Field(pattern=SHA256_PATTERN)
    detector_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    runtime_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    resource_limits_sha256: str = Field(pattern=SHA256_PATTERN)
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    global_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    shard_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    shard_id: str = Field(min_length=1, pattern=r"^shard-[0-9]{6}$")
    video_ids: tuple[str, ...] = Field(min_length=1)
    stage: Literal["detect", "track", "select", "completed"]
    files: tuple[OcrCheckpointFile, ...]
    artifacts: tuple[OcrCheckpointArtifact, ...]
    counts: OcrCheckpointCounts
    next_frame_uid: str | None = None
    next_stage: Literal["detect", "track", "select", "completed"]
    created_at: datetime
    checkpoint_sequence: int = Field(ge=1)
    previous_checkpoint_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    trust_boundary: Literal["integrity_metadata_not_a_signature"] = (
        "integrity_metadata_not_a_signature"
    )

    @field_validator("video_ids", mode="before")
    @classmethod
    def accept_json_video_ids(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_checkpoint_state(self) -> OcrPhase1Checkpoint:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("checkpoint created_at must be timezone-aware")
        if self.video_ids != tuple(sorted(set(self.video_ids))):
            raise ValueError("checkpoint video_ids must be sorted and unique")
        roles = [item.role for item in self.files]
        if (
            set(roles)
            != {
                "source_manifest",
                "global_manifest",
                "global_receipt",
                "shard_manifest",
            }
            or len(roles) != 4
        ):
            raise ValueError("checkpoint requires exactly four manifest files")
        stages = [item.stage for item in self.artifacts]
        required = {
            "detect": ["detect"],
            "track": ["detect"],
            "select": ["detect", "track"],
            "completed": ["detect", "track", "select"],
        }[self.stage]
        if stages != required:
            raise ValueError("checkpoint artifact stages are incomplete or out of order")
        if self.next_stage != self.stage:
            raise ValueError("checkpoint stage and next_stage must agree")
        if self.stage == "detect":
            if self.artifacts[0].receipt_status != "running":
                raise ValueError("detect checkpoint requires a running detection receipt")
        elif any(item.receipt_status != "completed" for item in self.artifacts):
            raise ValueError("post-detection checkpoints require completed receipts")
        if self.stage != "detect" and self.next_frame_uid is not None:
            raise ValueError("only a detection checkpoint may identify a next frame")
        bundle_paths = [item.bundle_relpath for item in self.files]
        bundle_paths += [
            path for item in self.artifacts for path in (item.payload_relpath, item.receipt_relpath)
        ]
        if len(bundle_paths) != len(set(bundle_paths)):
            raise ValueError("checkpoint bundle paths must be unique")
        artifact_paths = [item.artifact_relpath for item in self.artifacts]
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError("checkpoint artifact restore paths must be unique")
        return self
