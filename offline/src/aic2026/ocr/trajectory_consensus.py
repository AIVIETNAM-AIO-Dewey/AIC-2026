"""Deterministic trajectory consensus and backend-facing OCR adapter."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from aic2026.common import atomic_write_json, iter_jsonl, sha256_file, write_jsonl_atomic
from aic2026.contracts import (
    OcrPhase1Receipt,
    OcrTrajectoryRecord,
    RepresentativeCropBinding,
    TrajectoryMember,
)
from aic2026.contracts.models import StrictModel
from aic2026.ocr.phase1 import receipt_path_for

from .representative_recognition import (
    RepresentativeRecognitionReceipt,
    RepresentativeRecognitionResult,
    _parse_prefix,
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"
CONSENSUS_SELECTION_POLICY = "support_count_then_phase1_quality_then_rank_then_text.v1"


class TrajectoryConsensusRecord(StrictModel):
    schema_version: Literal["aic26.ocr_trajectory_consensus.v1"] = (
        "aic26.ocr_trajectory_consensus.v1"
    )
    source_phase1_run_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    frame_uids: list[str] = Field(min_length=1)
    representative_ranks: list[int] = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=SHA256_PATTERN)
    status: Literal["accepted", "empty"]
    method: Literal["single", "exact_agreement", "ranked_vote", "empty"]
    transcript_raw: str
    transcript_nfc: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    supporting_ranks: list[int]
    disagreeing_ranks: list[int]

    @model_validator(mode="after")
    def validate_consensus(self) -> TrajectoryConsensusRecord:
        if self.representative_ranks != sorted(set(self.representative_ranks)):
            raise ValueError("representative ranks must be sorted and unique")
        known = set(self.representative_ranks)
        if not set(self.supporting_ranks).issubset(known) or not set(
            self.disagreeing_ranks
        ).issubset(known):
            raise ValueError("consensus rank evidence must bind input representatives")
        if set(self.supporting_ranks) & set(self.disagreeing_ranks):
            raise ValueError("supporting and disagreeing ranks cannot overlap")
        if set(self.supporting_ranks) | set(self.disagreeing_ranks) != known:
            raise ValueError("consensus evidence must account for every representative rank")
        if self.status == "empty":
            if (
                self.method != "empty"
                or self.transcript_raw
                or self.transcript_nfc
                or self.confidence is not None
                or self.supporting_ranks
            ):
                raise ValueError("empty consensus must have canonical empty payload")
        elif not self.transcript_nfc or not self.supporting_ranks or self.method == "empty":
            raise ValueError("accepted consensus requires text and supporting evidence")
        return self


class TrajectoryConsensusReceipt(StrictModel):
    schema_version: Literal[
        "aic26.ocr_trajectory_consensus_receipt.v1",
        "aic26.ocr_trajectory_consensus_receipt.v2",
    ] = "aic26.ocr_trajectory_consensus_receipt.v2"
    status: Literal["completed"] = "completed"
    run_id: str = Field(min_length=1)
    source_phase1_run_id: str = Field(min_length=1)
    trajectories_sha256: str = Field(pattern=SHA256_PATTERN)
    representatives_sha256: str = Field(pattern=SHA256_PATTERN)
    recognition_output_sha256: str = Field(pattern=SHA256_PATTERN)
    recognition_receipt_sha256: str = Field(pattern=SHA256_PATTERN)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=SHA256_PATTERN)
    selection_policy: Literal["support_count_then_phase1_quality_then_rank_then_text.v1"] | None = (
        None
    )
    trajectories: int = Field(ge=0)
    accepted: int = Field(ge=0)
    empty: int = Field(ge=0)
    output_sha256: str = Field(pattern=SHA256_PATTERN)
    trust_boundary: Literal["integrity_metadata_not_a_signature"] = (
        "integrity_metadata_not_a_signature"
    )

    @model_validator(mode="after")
    def validate_counts(self) -> TrajectoryConsensusReceipt:
        if self.accepted + self.empty != self.trajectories:
            raise ValueError("consensus receipt counters are inconsistent")
        if self.schema_version.endswith(".v2") and self.selection_policy is None:
            raise ValueError("v2 consensus receipts must bind the selection policy")
        return self


class FinalOcrLine(StrictModel):
    line_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    raw_text: str
    normalized_text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    accepted: bool
    polygon_xy: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    reading_order: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_text(self) -> FinalOcrLine:
        if self.accepted != bool(self.normalized_text):
            raise ValueError("final OCR line acceptance must match non-empty text")
        return self


class FinalOcrFrameRecord(StrictModel):
    schema_version: Literal["aic26.ocr_frame.v2"] = "aic26.ocr_frame.v2"
    video_id: str = Field(min_length=1)
    frame_uid: str = Field(min_length=3)
    frame_idx: int = Field(ge=0)
    keyframe_n: int | None = Field(default=None, ge=1)
    pts_time_s: float = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    source_image_sha256: str = Field(pattern=SHA256_PATTERN)
    terminal_status: Literal["success"] = "success"
    full_text: str
    texts: list[FinalOcrLine]

    @model_validator(mode="after")
    def validate_frame(self) -> FinalOcrFrameRecord:
        if self.frame_uid != f"{self.video_id}:{self.frame_idx}":
            raise ValueError("final OCR frame identity is inconsistent")
        ids = [line.line_id for line in self.texts]
        if len(ids) != len(set(ids)):
            raise ValueError("final OCR frame contains duplicate line_id values")
        expected = " ".join(line.normalized_text for line in self.texts if line.accepted)
        if self.full_text != expected:
            raise ValueError("final OCR frame full_text differs from accepted lines")
        return self


def _phase1_receipt(path: Path, stage: str) -> OcrPhase1Receipt:
    receipt = OcrPhase1Receipt.model_validate_json(
        receipt_path_for(path).read_text(encoding="utf-8")
    )
    if receipt.status != "completed" or receipt.stage != stage:
        raise ValueError(f"{stage} Phase 1 artifact is not completed")
    if receipt.output_sha256 != sha256_file(path):
        raise ValueError(f"{stage} Phase 1 artifact checksum drift")
    return receipt


def _verified_consensus_inputs(
    *,
    trajectories: Path,
    representatives: Path,
    recognition_output: Path,
) -> tuple[
    list[OcrTrajectoryRecord],
    list[RepresentativeCropBinding],
    list[RepresentativeRecognitionResult],
    RepresentativeRecognitionReceipt,
]:
    trajectory_hash = sha256_file(trajectories)
    representative_hash = sha256_file(representatives)
    recognition_hash = sha256_file(recognition_output)
    trajectory_receipt = _phase1_receipt(trajectories, "track")
    representative_receipt = _phase1_receipt(representatives, "select_representatives")
    if representative_receipt.input_artifact_sha256 != trajectory_hash:
        raise ValueError("representative receipt does not bind the trajectory artifact")
    recognition_receipt_path = recognition_output.with_suffix(
        recognition_output.suffix + ".receipt.json"
    )
    recognition_receipt_hash = sha256_file(recognition_receipt_path)
    recognition_receipt = RepresentativeRecognitionReceipt.model_validate_json(
        recognition_receipt_path.read_text(encoding="utf-8")
    )
    if recognition_receipt.status != "completed":
        raise ValueError("representative recognition is not completed")
    if (
        recognition_receipt.trajectories_sha256 != trajectory_hash
        or recognition_receipt.representatives_sha256 != representative_hash
        or recognition_receipt.output_sha256 != recognition_hash
    ):
        raise ValueError("representative recognition input/output identity drift")
    if recognition_output.stat().st_size != recognition_receipt.committed_bytes:
        raise ValueError("representative recognition byte count drift")
    if (
        trajectory_receipt.run_id != recognition_receipt.run_id
        or representative_receipt.run_id != recognition_receipt.run_id
        or trajectory_receipt.config_sha256 != recognition_receipt.phase1_config_sha256
        or representative_receipt.config_sha256 != recognition_receipt.phase1_config_sha256
    ):
        raise ValueError("Phase 1 run/config identity differs from recognition receipt")
    trajectory_records = [
        OcrTrajectoryRecord.model_validate(row) for row in iter_jsonl(trajectories)
    ]
    representative_records = [
        RepresentativeCropBinding.model_validate(row) for row in iter_jsonl(representatives)
    ]
    result_records = _parse_prefix(
        recognition_output.read_bytes(),
        representative_records,
        model_id=recognition_receipt.model_id,
        model_revision=recognition_receipt.model_revision,
    )
    if (
        len(result_records) != recognition_receipt.total_representatives
        or len(result_records) != recognition_receipt.committed_records
    ):
        raise ValueError("representative recognition record count drift")
    if (
        sha256_file(trajectories) != trajectory_hash
        or sha256_file(representatives) != representative_hash
        or sha256_file(recognition_output) != recognition_hash
        or sha256_file(recognition_receipt_path) != recognition_receipt_hash
    ):
        raise ValueError("consensus input changed during verification")
    return trajectory_records, representative_records, result_records, recognition_receipt


def _choose_consensus(
    trajectory: OcrTrajectoryRecord,
    results: list[RepresentativeRecognitionResult],
    *,
    source_run_id: str,
    model_id: str,
    model_revision: str,
) -> TrajectoryConsensusRecord:
    ranks = [result.binding.representative_rank for result in results]
    usable = [
        result
        for result in results
        if result.status == "ok"
        and result.transcript_nfc is not None
        and bool(result.transcript_nfc.strip())
    ]
    if not usable:
        return TrajectoryConsensusRecord(
            source_phase1_run_id=source_run_id,
            video_id=trajectory.video_id,
            trajectory_id=trajectory.trajectory_id,
            frame_uids=[member.frame_uid for member in trajectory.members],
            representative_ranks=ranks,
            model_id=model_id,
            model_revision=model_revision,
            status="empty",
            method="empty",
            transcript_raw="",
            transcript_nfc="",
            confidence=None,
            supporting_ranks=[],
            disagreeing_ranks=ranks,
        )
    groups: dict[str, list[RepresentativeRecognitionResult]] = defaultdict(list)
    for result in usable:
        assert result.transcript_nfc is not None
        groups[result.transcript_nfc].append(result)

    def group_key(item: tuple[str, list[RepresentativeRecognitionResult]]) -> tuple:
        text, members = item
        phase1_quality = sum(member.binding.quality_score for member in members)
        return (
            -len(members),
            -phase1_quality,
            min(member.binding.representative_rank for member in members),
            text,
        )

    winning_text, winners = min(groups.items(), key=group_key)
    representative = min(
        winners,
        key=lambda item: (
            -item.binding.quality_score,
            item.binding.representative_rank,
        ),
    )
    confidences = [item.confidence for item in winners if item.confidence is not None]
    supporting = sorted(item.binding.representative_rank for item in winners)
    method: Literal["single", "exact_agreement", "ranked_vote"]
    if len(usable) == 1:
        method = "single"
    elif len(groups) == 1:
        method = "exact_agreement"
    else:
        method = "ranked_vote"
    return TrajectoryConsensusRecord(
        source_phase1_run_id=source_run_id,
        video_id=trajectory.video_id,
        trajectory_id=trajectory.trajectory_id,
        frame_uids=[member.frame_uid for member in trajectory.members],
        representative_ranks=ranks,
        model_id=model_id,
        model_revision=model_revision,
        status="accepted",
        method=method,
        transcript_raw=representative.transcript_raw or winning_text,
        transcript_nfc=winning_text,
        confidence=sum(confidences) / len(confidences) if confidences else None,
        supporting_ranks=supporting,
        disagreeing_ranks=sorted(set(ranks) - set(supporting)),
    )


def run_trajectory_consensus(
    *,
    trajectories: Path,
    representatives: Path,
    recognition_output: Path,
    output: Path,
    run_id: str,
) -> dict[str, int | str]:
    """Publish one deterministic consensus record per Phase 1 trajectory."""

    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("fresh trajectory consensus output and receipt are required")
    trajectory_hash = sha256_file(trajectories)
    representative_hash = sha256_file(representatives)
    recognition_hash = sha256_file(recognition_output)
    recognition_receipt_path = recognition_output.with_suffix(
        recognition_output.suffix + ".receipt.json"
    )
    recognition_receipt_hash = sha256_file(recognition_receipt_path)
    trajectory_records, representative_records, results, recognition_receipt = (
        _verified_consensus_inputs(
            trajectories=trajectories,
            representatives=representatives,
            recognition_output=recognition_output,
        )
    )
    results_by_trajectory: dict[str, list[RepresentativeRecognitionResult]] = defaultdict(list)
    for result in results:
        results_by_trajectory[result.binding.trajectory_id].append(result)
    representative_ids = {item.trajectory_id for item in representative_records}
    if representative_ids != {trajectory.trajectory_id for trajectory in trajectory_records}:
        raise ValueError("trajectory/representative membership drift")
    consensus: list[TrajectoryConsensusRecord] = []
    for trajectory in trajectory_records:
        trajectory_results = results_by_trajectory.get(trajectory.trajectory_id, [])
        expected_ranks = list(range(1, min(len(trajectory.members), 3) + 1))
        actual_ranks = [item.binding.representative_rank for item in trajectory_results]
        if actual_ranks != expected_ranks:
            raise ValueError("trajectory representative rank/order drift")
        consensus.append(
            _choose_consensus(
                trajectory,
                trajectory_results,
                source_run_id=recognition_receipt.run_id,
                model_id=recognition_receipt.model_id,
                model_revision=recognition_receipt.model_revision,
            )
        )
    if (
        sha256_file(trajectories) != trajectory_hash
        or sha256_file(representatives) != representative_hash
        or sha256_file(recognition_output) != recognition_hash
        or sha256_file(recognition_receipt_path) != recognition_receipt_hash
    ):
        raise ValueError("consensus input changed before publication")
    write_jsonl_atomic(output, consensus)
    receipt = TrajectoryConsensusReceipt(
        run_id=run_id,
        source_phase1_run_id=recognition_receipt.run_id,
        trajectories_sha256=trajectory_hash,
        representatives_sha256=representative_hash,
        recognition_output_sha256=recognition_hash,
        recognition_receipt_sha256=recognition_receipt_hash,
        model_id=recognition_receipt.model_id,
        model_revision=recognition_receipt.model_revision,
        selection_policy=CONSENSUS_SELECTION_POLICY,
        trajectories=len(consensus),
        accepted=sum(item.status == "accepted" for item in consensus),
        empty=sum(item.status == "empty" for item in consensus),
        output_sha256=sha256_file(output),
    )
    atomic_write_json(receipt_path, receipt.model_dump(mode="json"))
    return {
        "trajectories": len(consensus),
        "accepted": receipt.accepted,
        "empty": receipt.empty,
        "output_sha256": receipt.output_sha256,
    }


def build_final_ocr_artifact(
    *, trajectories: Path, consensus: Path, output: Path, run_id: str
) -> dict[str, int | str]:
    """Expand trajectory text over bound frames and publish an ingestible OCR JSONL."""

    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("fresh final OCR output and manifest are required")
    trajectory_hash = sha256_file(trajectories)
    consensus_hash = sha256_file(consensus)
    trajectory_receipt = _phase1_receipt(trajectories, "track")
    consensus_receipt_path = consensus.with_suffix(consensus.suffix + ".receipt.json")
    consensus_receipt_hash = sha256_file(consensus_receipt_path)
    consensus_receipt = TrajectoryConsensusReceipt.model_validate_json(
        consensus_receipt_path.read_text(encoding="utf-8")
    )
    if (
        consensus_receipt.trajectories_sha256 != trajectory_hash
        or consensus_receipt.output_sha256 != consensus_hash
    ):
        raise ValueError("consensus receipt input/output identity drift")
    trajectory_records = [
        OcrTrajectoryRecord.model_validate(row) for row in iter_jsonl(trajectories)
    ]
    consensus_records = [
        TrajectoryConsensusRecord.model_validate(row) for row in iter_jsonl(consensus)
    ]
    if len(trajectory_records) != len(consensus_records):
        raise ValueError("consensus must contain one record per trajectory")
    if len(consensus_records) != consensus_receipt.trajectories:
        raise ValueError("consensus receipt record count drift")

    frame_lines: dict[tuple[str, int], list[tuple[TrajectoryMember, TrajectoryConsensusRecord]]] = (
        defaultdict(list)
    )
    frame_identity: dict[tuple[str, int], tuple] = {}
    for index, (trajectory, result) in enumerate(
        zip(trajectory_records, consensus_records, strict=True)
    ):
        if (
            result.trajectory_id != trajectory.trajectory_id
            or result.video_id != trajectory.video_id
            or result.frame_uids != [member.frame_uid for member in trajectory.members]
            or result.source_phase1_run_id != trajectory_receipt.run_id
            or result.model_id != consensus_receipt.model_id
            or result.model_revision != consensus_receipt.model_revision
        ):
            raise ValueError(f"consensus trajectory identity/order drift at record {index}")
        for member in trajectory.members:
            key = (member.video_id, member.frame_idx)
            identity = (
                member.frame_uid,
                member.pts_time_s,
                member.source_width,
                member.source_height,
                member.source_image_sha256,
            )
            if key in frame_identity and frame_identity[key] != identity:
                raise ValueError("trajectory members disagree on source frame identity")
            frame_identity[key] = identity
            frame_lines[key].append((member, result))

    frames: list[FinalOcrFrameRecord] = []
    for key in sorted(frame_lines):
        entries = sorted(
            frame_lines[key],
            key=lambda pair: (
                min(point[1] for point in pair[0].polygon_xy.points),
                min(point[0] for point in pair[0].polygon_xy.points),
                pair[1].trajectory_id,
            ),
        )
        lines = [
            FinalOcrLine(
                line_id=result.trajectory_id,
                trajectory_id=result.trajectory_id,
                raw_text=result.transcript_raw,
                normalized_text=result.transcript_nfc,
                confidence=result.confidence,
                accepted=result.status == "accepted",
                polygon_xy=member.polygon_xy.points,
                reading_order=reading_order,
            )
            for reading_order, (member, result) in enumerate(entries)
        ]
        first_member = entries[0][0]
        frames.append(
            FinalOcrFrameRecord(
                video_id=first_member.video_id,
                frame_uid=first_member.frame_uid,
                frame_idx=first_member.frame_idx,
                pts_time_s=first_member.pts_time_s,
                width=first_member.source_width,
                height=first_member.source_height,
                source_image_sha256=first_member.source_image_sha256,
                full_text=" ".join(line.normalized_text for line in lines if line.accepted),
                texts=lines,
            )
        )
    if (
        sha256_file(trajectories) != trajectory_hash
        or sha256_file(consensus) != consensus_hash
        or sha256_file(consensus_receipt_path) != consensus_receipt_hash
    ):
        raise ValueError("final OCR input changed before publication")
    write_jsonl_atomic(output, frames)
    output_hash = sha256_file(output)
    if (
        sha256_file(trajectories) != trajectory_hash
        or sha256_file(consensus) != consensus_hash
        or sha256_file(consensus_receipt_path) != consensus_receipt_hash
    ):
        raise ValueError("final OCR input changed during publication")
    manifest = {
        "schema_version": "aic26.ocr_final_manifest.v1",
        "status": "completed",
        "run_id": run_id,
        "counters": {
            "frames": len(frames),
            "lines": sum(len(frame.texts) for frame in frames),
            "accepted_lines": sum(line.accepted for frame in frames for line in frame.texts),
        },
        "models": [
            {
                "model_id": consensus_receipt.model_id,
                "revision": consensus_receipt.model_revision,
            }
        ],
        "inputs": [
            {"source_id": str(trajectories), "sha256": trajectory_hash},
            {"source_id": str(consensus), "sha256": consensus_hash},
            {
                "source_id": str(consensus_receipt_path),
                "sha256": consensus_receipt_hash,
            },
        ],
        "outputs": [{"source_id": str(output), "sha256": output_hash}],
    }
    atomic_write_json(manifest_path, manifest)
    return {
        "frames": len(frames),
        "lines": manifest["counters"]["lines"],
        "accepted_lines": manifest["counters"]["accepted_lines"],
        "output_sha256": output_hash,
    }
