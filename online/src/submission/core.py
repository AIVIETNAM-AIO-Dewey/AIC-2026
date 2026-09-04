"""Pure submission validation, completion, and official CSV policies.

The module intentionally knows nothing about FastAPI or the on-disk dataset.
Callers inject an authoritative source-frame lookup, which keeps request
validation, preview, and final export on one deterministic policy path.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

SubmissionTaskType = Literal["KIS", "VQA", "QA", "TRAKE"]
VerifiedFrameLookup = Callable[[str, int], Mapping[str, Any] | None]
VideoFramesLookup = Callable[[str], Sequence[Mapping[str, Any]]]

_PROVENANCE_FIELDS = (
    "source",
    "modality",
    "score",
    "score_type",
    "rank",
    "reservoir_rank",
    "scope",
    "query_id",
)
_DISPLAY_FIELDS = (
    "frame_uid",
    "fps",
    "image_relpath",
    "image_available",
    "indexed_keyframe",
    "validation",
    "frame_index_base",
    "max_frame_idx",
    "duration_s",
    "timing_method",
    "preview_frame_idx",
    "preview_keyframe_n",
    "preview_pts_time_s",
    "preview_image_relpath",
    "related_seed_frame_idx",
)
_SEQUENCE_PROVENANCE_FIELDS = (
    "source",
    "modality",
    "rank",
    "reservoir_rank",
    "sequence_score",
    "score_type",
    "query_id",
)
_MAX_SUBMISSION_ROWS = 100
_MAX_VQA_ANSWER_CHARACTERS = 100
_MAX_TRAKE_EXPANSION_ATTEMPTS = 20_000
_MAX_TRAKE_NEIGHBOR_RADIUS = 12
_MAX_TRAKE_NEIGHBOR_TIME_DELTA_SECONDS = 30.0


class SubmissionValidationError(ValueError):
    """Raised when a manual or canonical submission reference is invalid."""


def _strict_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise SubmissionValidationError(f"{field_name} must be a non-negative integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.startswith("+"):
            text = text[1:]
        if not text.isdigit():
            raise SubmissionValidationError(f"{field_name} must be a non-negative integer")
        parsed = int(text)
    else:
        raise SubmissionValidationError(f"{field_name} must be a non-negative integer")
    if parsed < 0:
        raise SubmissionValidationError(f"{field_name} must be a non-negative integer")
    return parsed


def _strict_non_negative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise SubmissionValidationError(f"{field_name} must be a finite non-negative number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SubmissionValidationError(
            f"{field_name} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise SubmissionValidationError(f"{field_name} must be a finite non-negative number")
    return parsed


def _verified_frame(
    value: Mapping[str, Any],
    *,
    expected_video_id: str | None = None,
    expected_frame_idx: int | None = None,
) -> dict[str, Any]:
    video_id = str(value.get("video_id", "")).strip()
    if not video_id:
        raise SubmissionValidationError("video_id is required")
    frame_idx = _strict_non_negative_int(value.get("frame_idx"), "frame_idx")
    keyframe_value = value.get("keyframe_n")
    keyframe_n = (
        None
        if keyframe_value is None
        else _strict_non_negative_int(keyframe_value, "keyframe_n")
    )
    pts_time_s = _strict_non_negative_float(value.get("pts_time_s"), "pts_time_s")

    if expected_video_id is not None and video_id != expected_video_id:
        raise SubmissionValidationError(
            "Frame lookup returned a different video_id "
            f"({video_id!r} instead of {expected_video_id!r})"
        )
    if expected_frame_idx is not None and frame_idx != expected_frame_idx:
        raise SubmissionValidationError(
            "Frame lookup returned a different frame_idx "
            f"({frame_idx} instead of {expected_frame_idx})"
        )

    frame: dict[str, Any] = {
        "video_id": video_id,
        "frame_idx": frame_idx,
        "keyframe_n": keyframe_n,
        "pts_time_s": pts_time_s,
        "submission_string": f"{video_id}, {frame_idx}",
    }
    for field_name in _DISPLAY_FIELDS:
        if field_name in value:
            frame[field_name] = value[field_name]
    return frame


def validate_frame_reference(
    value: Mapping[str, Any],
    frame_lookup: VerifiedFrameLookup,
) -> dict[str, Any]:
    """Resolve a user/result reference to a verified zero-based source frame."""
    if not isinstance(value, Mapping):
        raise SubmissionValidationError("A frame reference must be an object")
    video_id = str(value.get("video_id", "")).strip()
    if not video_id:
        raise SubmissionValidationError("video_id is required")
    frame_idx = _strict_non_negative_int(value.get("frame_idx"), "frame_idx")
    resolved = frame_lookup(video_id, frame_idx)
    if resolved is None:
        raise SubmissionValidationError(
            f"Frame {video_id}:{frame_idx} is outside the verified source timeline"
        )

    frame = _verified_frame(
        resolved,
        expected_video_id=video_id,
        expected_frame_idx=frame_idx,
    )
    for field_name in _PROVENANCE_FIELDS:
        if field_name in value:
            frame[field_name] = value[field_name]
    return frame


def _frame_identity(frame: Mapping[str, Any]) -> tuple[str, int]:
    return str(frame["video_id"]), int(frame["frame_idx"])


def _nearest_timeline_position(
    frames: Sequence[Mapping[str, Any]],
    seed: Mapping[str, Any],
) -> int | None:
    """Locate an exact indexed frame or the closest earlier-tie anchor."""

    if not frames:
        return None
    target = int(seed["frame_idx"])
    return min(
        range(len(frames)),
        key=lambda position: (
            abs(int(frames[position]["frame_idx"]) - target),
            int(frames[position]["frame_idx"]),
        ),
    )


def _frame_row(
    frame: Mapping[str, Any],
    *,
    origin: str,
    manual: bool,
    seed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = dict(frame)
    row.update(
        {
            "manual": manual,
            "auto_filled": not manual and origin == "canonical_neighbor",
            "selection_origin": origin,
        }
    )
    if seed is not None:
        row["neighbor_seed"] = {
            "video_id": str(seed["video_id"]),
            "frame_idx": int(seed["frame_idx"]),
        }
    return row


def _load_video_frames(
    video_id: str,
    *,
    video_frames_lookup: VideoFramesLookup,
    frame_lookup: VerifiedFrameLookup,
) -> tuple[list[dict[str, Any]], list[str]]:
    frames: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[tuple[str, int]] = set()
    for offset, value in enumerate(video_frames_lookup(video_id)):
        try:
            frame = validate_frame_reference(value, frame_lookup)
        except SubmissionValidationError as exc:
            warnings.append(f"Skipped invalid canonical neighbor {video_id}[{offset}]: {exc}")
            continue
        identity = _frame_identity(frame)
        if identity in seen:
            continue
        seen.add(identity)
        frames.append(frame)
    frames.sort(
        key=lambda frame: (
            int(frame["keyframe_n"])
            if frame.get("keyframe_n") is not None
            else math.inf,
            float(frame["pts_time_s"]),
            int(frame["frame_idx"]),
        )
    )
    return frames, warnings


def _complete_frame_rows(
    *,
    manual_items: Sequence[Mapping[str, Any]],
    candidate_items: Sequence[Mapping[str, Any]],
    frame_lookup: VerifiedFrameLookup,
    video_frames_lookup: VideoFramesLookup | None,
    target_rows: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[tuple[str, int]] = set()

    for offset, value in enumerate(manual_items):
        try:
            frame = validate_frame_reference(value, frame_lookup)
        except SubmissionValidationError as exc:
            raise SubmissionValidationError(
                f"Invalid manual frame at position {offset + 1}: {exc}"
            ) from exc
        identity = _frame_identity(frame)
        if identity in seen:
            continue
        seen.add(identity)
        rows.append(_frame_row(frame, origin="manual", manual=True))
    if len(rows) > target_rows:
        raise SubmissionValidationError(
            f"The submission has {len(rows)} unique manual frames but target_rows is {target_rows}"
        )

    for offset, value in enumerate(candidate_items):
        if len(rows) >= target_rows:
            break
        try:
            frame = validate_frame_reference(value, frame_lookup)
        except SubmissionValidationError as exc:
            warnings.append(f"Skipped invalid candidate at position {offset + 1}: {exc}")
            continue
        identity = _frame_identity(frame)
        if identity in seen:
            continue
        seen.add(identity)
        rows.append(_frame_row(frame, origin="active_query_reservoir", manual=False))

    if len(rows) >= target_rows or video_frames_lookup is None:
        return rows, warnings

    seeds = list(rows)
    cached_video_frames: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        if len(rows) >= target_rows:
            break
        video_id = str(seed["video_id"])
        if video_id not in cached_video_frames:
            video_frames, load_warnings = _load_video_frames(
                video_id,
                video_frames_lookup=video_frames_lookup,
                frame_lookup=frame_lookup,
            )
            cached_video_frames[video_id] = video_frames
            warnings.extend(load_warnings)
        video_frames = cached_video_frames[video_id]
        seed_identity = _frame_identity(seed)
        exact_seed_offset = next(
            (
                offset
                for offset, frame in enumerate(video_frames)
                if _frame_identity(frame) == seed_identity
            ),
            None,
        )
        seed_offset = (
            exact_seed_offset
            if exact_seed_offset is not None
            else _nearest_timeline_position(video_frames, seed)
        )
        if seed_offset is None:
            warnings.append(
                f"Could not locate seed {video_id}:{seed['frame_idx']} in its indexed timeline"
            )
            continue

        # A verified non-keyframe has no identical sparse timeline row. Admit
        # its nearest indexed anchor before walking outward; the submitted
        # source-frame identity remains untouched and first in the draft.
        if exact_seed_offset is None:
            nearest_anchor = video_frames[seed_offset]
            identity = _frame_identity(nearest_anchor)
            if identity not in seen:
                seen.add(identity)
                rows.append(
                    _frame_row(
                        nearest_anchor,
                        origin="canonical_neighbor",
                        manual=False,
                        seed=seed,
                    )
                )
                if len(rows) >= target_rows:
                    return rows, warnings

        for distance in range(1, len(video_frames)):
            for neighbor_offset in (seed_offset - distance, seed_offset + distance):
                if not 0 <= neighbor_offset < len(video_frames):
                    continue
                neighbor = video_frames[neighbor_offset]
                identity = _frame_identity(neighbor)
                if identity in seen:
                    continue
                seen.add(identity)
                rows.append(
                    _frame_row(
                        neighbor,
                        origin="canonical_neighbor",
                        manual=False,
                        seed=seed,
                    )
                )
                if len(rows) >= target_rows:
                    return rows, warnings

    return rows, warnings


def _sequence_values(value: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        for field_name in ("matched_events", "events", "frames"):
            candidate = value.get(field_name)
            if isinstance(candidate, Sequence) and not isinstance(candidate, str | bytes):
                return candidate
        raise SubmissionValidationError(
            "A TRAKE sequence object must contain matched_events, events, or frames"
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    raise SubmissionValidationError("A TRAKE sequence must be a list of frame references")


def validate_trake_sequence(
    value: Any,
    *,
    frame_lookup: VerifiedFrameLookup,
    event_count: int,
) -> dict[str, Any]:
    """Validate one complete same-video, strictly increasing TRAKE sequence."""
    if event_count < 2:
        raise SubmissionValidationError("TRAKE event_count must be at least 2")
    values = list(_sequence_values(value))
    if len(values) != event_count:
        raise SubmissionValidationError(
            f"TRAKE sequence requires {event_count} events, received {len(values)}"
        )

    supplied_orders = [
        frame_value.get("event_order") if isinstance(frame_value, Mapping) else None
        for frame_value in values
    ]
    if any(order is not None for order in supplied_orders):
        if any(order is None for order in supplied_orders):
            raise SubmissionValidationError(
                "TRAKE event_order must be supplied for every event or omitted from every event"
            )
        parsed_orders = [
            _strict_non_negative_int(order, "event_order") for order in supplied_orders
        ]
        expected_orders = list(range(1, event_count + 1))
        if sorted(parsed_orders) != expected_orders:
            raise SubmissionValidationError(
                f"TRAKE event_order values must be unique and exactly 1..{event_count}"
            )
        values = [
            frame_value
            for _, frame_value in sorted(
                zip(parsed_orders, values, strict=True),
                key=lambda item: item[0],
            )
        ]

    frames: list[dict[str, Any]] = []
    for event_offset, frame_value in enumerate(values):
        if not isinstance(frame_value, Mapping):
            raise SubmissionValidationError(
                f"TRAKE event {event_offset + 1} must be a frame reference object"
            )
        try:
            frame = validate_frame_reference(frame_value, frame_lookup)
        except SubmissionValidationError as exc:
            raise SubmissionValidationError(
                f"Invalid TRAKE event {event_offset + 1}: {exc}"
            ) from exc
        frame["event_order"] = event_offset + 1
        frames.append(frame)

    video_ids = {str(frame["video_id"]) for frame in frames}
    if len(video_ids) != 1:
        raise SubmissionValidationError("Every event in a TRAKE sequence must use the same video")

    for previous, current in zip(frames, frames[1:], strict=False):
        if float(current["pts_time_s"]) <= float(previous["pts_time_s"]):
            raise SubmissionValidationError("TRAKE event timestamps must be strictly increasing")
        if (
            current.get("keyframe_n") is not None
            and previous.get("keyframe_n") is not None
            and int(current["keyframe_n"]) <= int(previous["keyframe_n"])
        ):
            raise SubmissionValidationError(
                "TRAKE indexed-keyframe numbers must be strictly increasing"
            )
        if int(current["frame_idx"]) <= int(previous["frame_idx"]):
            raise SubmissionValidationError("TRAKE frame indexes must be strictly increasing")

    video_id = str(frames[0]["video_id"])
    sequence = {
        "video_id": video_id,
        "event_count": event_count,
        "matched_frames": [int(frame["frame_idx"]) for frame in frames],
        "timestamps": [float(frame["pts_time_s"]) for frame in frames],
        "events": frames,
    }
    if isinstance(value, Mapping):
        for field_name in _SEQUENCE_PROVENANCE_FIELDS:
            if field_name in value:
                sequence[field_name] = value[field_name]
    return sequence


def _complete_trake_rows(
    *,
    manual_items: Sequence[Any],
    candidate_items: Sequence[Any],
    frame_lookup: VerifiedFrameLookup,
    video_frames_lookup: VideoFramesLookup | None,
    event_count: int,
    target_rows: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()

    for manual, items in ((True, manual_items), (False, candidate_items)):
        for offset, value in enumerate(items):
            if not manual and len(rows) >= target_rows:
                break
            try:
                sequence = validate_trake_sequence(
                    value,
                    frame_lookup=frame_lookup,
                    event_count=event_count,
                )
            except SubmissionValidationError as exc:
                if manual:
                    raise SubmissionValidationError(
                        f"Invalid manual TRAKE sequence at position {offset + 1}: {exc}"
                    ) from exc
                warnings.append(f"Skipped invalid TRAKE candidate at position {offset + 1}: {exc}")
                continue

            identity = (
                str(sequence["video_id"]),
                tuple(int(frame_idx) for frame_idx in sequence["matched_frames"]),
            )
            if identity in seen:
                continue
            seen.add(identity)
            sequence.update(
                {
                    "manual": manual,
                    "auto_filled": False,
                    "selection_origin": "manual" if manual else "active_query_reservoir",
                }
            )
            rows.append(sequence)

        if manual and len(rows) > target_rows:
            raise SubmissionValidationError(
                f"The submission has more manual TRAKE sequences than target_rows ({target_rows})"
            )

    if len(rows) >= target_rows:
        return rows[:target_rows], warnings

    if not rows:
        warnings.append(
            "TRAKE completion needs at least one valid manual or ordered-search sequence seed"
        )
        return rows, warnings
    if video_frames_lookup is None:
        warnings.append(
            "TRAKE canonical temporal expansion is unavailable because no video timeline lookup was supplied"
        )
        return rows, warnings

    # Expand only already-validated sequence seeds. Each alternative is resolved
    # from the canonical video timeline and is validated again before being
    # admitted. Generators are consumed round-robin so one video/seed cannot
    # monopolize all remaining official rows.
    seed_rows = list(rows)
    cached_video_frames: dict[str, list[dict[str, Any]]] = {}
    attempts = 0
    expansion_added = 0

    def offset_patterns(event_total: int, radius: int):
        """Yield a small deterministic, diverse neighborhood shell."""
        patterns: list[tuple[int, ...]] = []
        seen_patterns: set[tuple[int, ...]] = set()

        def add(values: Sequence[int]) -> None:
            pattern = tuple(values)
            if any(pattern) and pattern not in seen_patterns:
                seen_patterns.add(pattern)
                patterns.append(pattern)

        # Shift the whole event chain, individual events, and progressively
        # larger prefixes/suffixes. Spread/compress pairs cover asymmetric
        # alternatives without enumerating an exponential Cartesian product.
        for direction in (-1, 1):
            add([direction * radius] * event_total)
        for event_index in range(event_total):
            for direction in (-1, 1):
                values = [0] * event_total
                values[event_index] = direction * radius
                add(values)
        for width in range(1, event_total):
            for direction in (-1, 1):
                prefix = [
                    direction * radius if index < width else 0 for index in range(event_total)
                ]
                suffix = [
                    direction * radius if index >= event_total - width else 0
                    for index in range(event_total)
                ]
                add(prefix)
                add(suffix)
        for left in range(event_total):
            for right in range(left + 1, event_total):
                spread = [0] * event_total
                spread[left] = -radius
                spread[right] = radius
                add(spread)
                compress = [0] * event_total
                compress[left] = radius
                compress[right] = -radius
                add(compress)
                if radius > 1:
                    inner_radius = max(1, radius // 2)
                    for direction in (-1, 1):
                        leading = [0] * event_total
                        leading[left] = direction * radius
                        leading[right] = direction * inner_radius
                        add(leading)
                        trailing = [0] * event_total
                        trailing[left] = direction * inner_radius
                        trailing[right] = direction * radius
                        add(trailing)
        yield from patterns

    def alternatives(seed: Mapping[str, Any]):
        video_id = str(seed["video_id"])
        if video_id not in cached_video_frames:
            frames, load_warnings = _load_video_frames(
                video_id,
                video_frames_lookup=video_frames_lookup,
                frame_lookup=frame_lookup,
            )
            cached_video_frames[video_id] = frames
            warnings.extend(load_warnings)
        timeline = cached_video_frames[video_id]
        seed_positions: list[int] = []
        for event in seed["events"]:
            position = _nearest_timeline_position(timeline, event)
            if position is None:
                warnings.append(
                    "Could not locate TRAKE expansion seed event "
                    f"{video_id}:{event['frame_idx']} in its indexed timeline"
                )
                return
            seed_positions.append(position)

        # Keep lower-ranked alternatives local to the retrieved event rather
        # than drifting across an entire long video. Both keyframe distance and
        # timestamp distance are explicitly bounded; the global attempt cap is
        # a second hard runtime bound.
        max_radius = min(_MAX_TRAKE_NEIGHBOR_RADIUS, max(len(timeline) - 1, 0))
        for radius in range(1, max_radius + 1):
            for offsets in offset_patterns(event_count, radius):
                positions = [
                    position + delta
                    for position, delta in zip(seed_positions, offsets, strict=True)
                ]
                if any(position < 0 or position >= len(timeline) for position in positions):
                    continue
                if any(
                    current <= previous
                    for previous, current in zip(positions, positions[1:], strict=False)
                ):
                    continue
                if any(
                    abs(
                        float(timeline[position]["pts_time_s"])
                        - float(seed["events"][event_index]["pts_time_s"])
                    )
                    > _MAX_TRAKE_NEIGHBOR_TIME_DELTA_SECONDS
                    for event_index, position in enumerate(positions)
                ):
                    continue
                yield [timeline[position] for position in positions], offsets, radius

    generators = [alternatives(seed) for seed in seed_rows]
    active_generators = list(enumerate(generators))
    while (
        active_generators and len(rows) < target_rows and attempts < _MAX_TRAKE_EXPANSION_ATTEMPTS
    ):
        next_active: list[tuple[int, Any]] = []
        for seed_index, generator in active_generators:
            if len(rows) >= target_rows or attempts >= _MAX_TRAKE_EXPANSION_ATTEMPTS:
                break
            try:
                frame_values, offsets, radius = next(generator)
            except StopIteration:
                continue
            next_active.append((seed_index, generator))
            attempts += 1
            try:
                sequence = validate_trake_sequence(
                    frame_values,
                    frame_lookup=frame_lookup,
                    event_count=event_count,
                )
            except SubmissionValidationError:
                # Neighbor candidates are internal and expected to be rejected
                # at boundaries; avoid flooding the user with one warning per
                # attempted tuple. The aggregate expansion summary is retained.
                continue
            identity = (
                str(sequence["video_id"]),
                tuple(int(frame_idx) for frame_idx in sequence["matched_frames"]),
            )
            if identity in seen:
                continue
            seen.add(identity)
            seed = seed_rows[seed_index]
            sequence.update(
                {
                    "manual": False,
                    "auto_filled": True,
                    "selection_origin": "canonical_temporal_neighbor",
                    "neighbor_seed": {
                        "video_id": str(seed["video_id"]),
                        "matched_frames": [int(frame_idx) for frame_idx in seed["matched_frames"]],
                    },
                    "expansion_offsets": list(offsets),
                    "expansion_radius": radius,
                    "expansion_max_keyframe_radius": _MAX_TRAKE_NEIGHBOR_RADIUS,
                    "expansion_max_time_delta_seconds": (_MAX_TRAKE_NEIGHBOR_TIME_DELTA_SECONDS),
                }
            )
            rows.append(sequence)
            expansion_added += 1
        active_generators = next_active

    if expansion_added:
        warnings.append(
            f"Added {expansion_added} TRAKE rows from verified canonical neighboring keyframes"
        )
    if len(rows) < target_rows:
        reason = (
            f"the {_MAX_TRAKE_EXPANSION_ATTEMPTS}-attempt safety limit was reached"
            if attempts >= _MAX_TRAKE_EXPANSION_ATTEMPTS
            else "all bounded canonical neighbor alternatives were exhausted"
        )
        warnings.append(
            f"TRAKE submission has {len(rows)}/{target_rows} unique complete sequences; {reason}. "
            "No duplicate, fabricated, cross-video, or non-monotonic row was added."
        )

    return rows, warnings


def _official_csv_line(values: Sequence[Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter=",", lineterminator="", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(values)
    return output.getvalue()


def _attach_official_csv(
    task_type: str,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Attach the organizer's headerless UTF-8 CSV representation."""
    if task_type == "TRAKE":
        columns = ["video_id", "frame_id_1", "...", "frame_id_n"]
        for row in rows:
            values = [row["video_id"], *row["matched_frames"]]
            row["csv_values"] = values
            row["csv_line"] = _official_csv_line(values)
    elif task_type == "VQA":
        columns = ["video_id", "frame_idx", "answer"]
        for row in rows:
            values = [row["video_id"], row["frame_idx"], row["answer"]]
            row["csv_values"] = values
            row["csv_line"] = _official_csv_line(values)
    else:
        columns = ["video_id", "frame_idx"]
        for row in rows:
            values = [row["video_id"], row["frame_idx"]]
            row["csv_values"] = values
            row["csv_line"] = _official_csv_line(values)

    valid = 1 <= len(rows) <= _MAX_SUBMISSION_ROWS
    return {
        "has_header": False,
        "encoding": "UTF-8",
        "delimiter": ",",
        "line_ending": "CRLF",
        "columns": columns,
        "row_count": len(rows),
        "max_rows": _MAX_SUBMISSION_ROWS,
        "valid": valid,
        "content": "\r\n".join(str(row["csv_line"]) for row in rows),
    }


def _provenance_policy() -> dict[str, Any]:
    return {
        "priority": [
            "manual",
            "active_query_reservoir",
            "canonical_neighbor",
            "canonical_temporal_neighbor",
        ],
        "manual_rows_first": True,
        "duplicates_allowed": False,
        "fabricated_frames_allowed": False,
        "verified_source_timeline_frames_allowed": True,
        "source_frame_index_base": 0,
        "trake_neighbor_fill_allowed": True,
        "trake_neighbor_fill_requires_canonical_timeline": True,
        "trake_neighbor_max_keyframe_radius": _MAX_TRAKE_NEIGHBOR_RADIUS,
        "trake_neighbor_max_time_delta_seconds": _MAX_TRAKE_NEIGHBOR_TIME_DELTA_SECONDS,
        "active_candidate_order_preserved": True,
        "official_csv_has_header": False,
    }


def _empty_official_csv() -> dict[str, Any]:
    return {
        "has_header": False,
        "encoding": "UTF-8",
        "delimiter": ",",
        "line_ending": "CRLF",
        "columns": [],
        "row_count": 0,
        "max_rows": _MAX_SUBMISSION_ROWS,
        "valid": False,
        "content": "",
    }


def build_submission(
    task_type: SubmissionTaskType | str,
    *,
    manual_items: Sequence[Any],
    candidate_items: Sequence[Any],
    frame_lookup: VerifiedFrameLookup,
    video_frames_lookup: VideoFramesLookup | None = None,
    target_rows: int = 100,
    vqa_answer: str | None = None,
    event_count: int | None = None,
) -> dict[str, Any]:
    """Build a validated task-aware submission draft.

    KIS and VQA consume frame references, then use the active query reservoir
    before deterministic canonical timeline neighbors. TRAKE consumes complete
    event sequences and may fill remaining rows only with revalidated canonical
    neighboring keyframes from the same timeline.
    """
    normalized_task = str(task_type).strip().upper()
    if normalized_task == "QA":
        normalized_task = "VQA"
    if normalized_task not in {"KIS", "VQA", "TRAKE"}:
        raise SubmissionValidationError("task_type must be KIS, VQA/QA, or TRAKE")
    if (
        isinstance(target_rows, bool)
        or not isinstance(target_rows, int)
        or not 1 <= target_rows <= _MAX_SUBMISSION_ROWS
    ):
        raise SubmissionValidationError(
            f"target_rows must be an integer from 1 to {_MAX_SUBMISSION_ROWS}"
        )

    normalized_answer: str | None = None
    if normalized_task == "VQA":
        normalized_answer = str(vqa_answer or "").strip()
        if not normalized_answer:
            raise SubmissionValidationError("A human-provided VQA answer is required")
        if len(normalized_answer) > _MAX_VQA_ANSWER_CHARACTERS:
            raise SubmissionValidationError(
                f"VQA answer must contain at most {_MAX_VQA_ANSWER_CHARACTERS} characters"
            )

    if normalized_task == "TRAKE":
        if (
            event_count is None
            or isinstance(event_count, bool)
            or not isinstance(event_count, int)
            or event_count < 2
        ):
            raise SubmissionValidationError("TRAKE event_count must be an integer of at least 2")
        rows, warnings = _complete_trake_rows(
            manual_items=manual_items,
            candidate_items=candidate_items,
            frame_lookup=frame_lookup,
            video_frames_lookup=video_frames_lookup,
            event_count=event_count,
            target_rows=target_rows,
        )
    else:
        rows, warnings = _complete_frame_rows(
            manual_items=manual_items,
            candidate_items=candidate_items,
            frame_lookup=frame_lookup,
            video_frames_lookup=video_frames_lookup,
            target_rows=target_rows,
        )

    for row_number, row in enumerate(rows, 1):
        row["row_number"] = row_number
        if normalized_task == "VQA":
            # The answer is entered once by the human, but official-style CSV
            # rows carry it beside every candidate frame.
            row["answer"] = normalized_answer

    official_csv = _attach_official_csv(normalized_task, rows)

    manual_row_count = sum(bool(row.get("manual")) for row in rows)
    auto_filled_row_count = sum(bool(row.get("auto_filled")) for row in rows)
    return {
        "ok": True,
        "task_type": normalized_task,
        "target_rows": target_rows,
        "row_count": len(rows),
        "complete": len(rows) == target_rows,
        "valid_for_download": bool(official_csv["valid"]),
        "missing_rows": max(0, target_rows - len(rows)),
        "manual_row_count": manual_row_count,
        "reservoir_row_count": len(rows) - manual_row_count - auto_filled_row_count,
        "auto_filled_row_count": auto_filled_row_count,
        "vqa_answer": normalized_answer,
        "rows": rows,
        "official_csv": official_csv,
        "warnings": warnings,
        "errors": [],
        "provenance_policy": _provenance_policy(),
    }


def _payload_sequence(payload: Mapping[str, Any], field_name: str) -> Sequence[Any]:
    value = payload.get(field_name, ())
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise SubmissionValidationError(f"{field_name} must be a list")
    return value


def prepare_submission(
    payload: Mapping[str, Any],
    *,
    frame_lookup: VerifiedFrameLookup,
    video_frames_lookup: VideoFramesLookup | None = None,
) -> dict[str, Any]:
    """Prepare a normalized API response from a compact request-shaped payload.

    Blocking request/manual-selection errors are represented in ``errors`` so a
    FastAPI endpoint can return a complete preview contract without duplicating
    policy. Unexpected lookup/storage failures are intentionally not swallowed.
    """
    if not isinstance(payload, Mapping):
        return {
            "ok": False,
            "task_type": "",
            "mode": "",
            "query_id": "",
            "target_rows": 100,
            "row_count": 0,
            "complete": False,
            "valid_for_download": False,
            "missing_rows": 100,
            "manual_row_count": 0,
            "reservoir_row_count": 0,
            "auto_filled_row_count": 0,
            "vqa_answer": None,
            "rows": [],
            "official_csv": _empty_official_csv(),
            "warnings": [],
            "errors": ["Submission payload must be an object"],
            "provenance_policy": _provenance_policy(),
        }

    raw_task_type = str(payload.get("task_type", payload.get("mode", ""))).strip().upper()
    mode = str(payload.get("mode") or raw_task_type).strip()
    query_id = str(payload.get("query_id", "")).strip()
    raw_target_rows = payload.get("target_rows", 100)
    fallback_target = (
        raw_target_rows
        if isinstance(raw_target_rows, int)
        and not isinstance(raw_target_rows, bool)
        and 1 <= raw_target_rows <= _MAX_SUBMISSION_ROWS
        else 100
    )

    try:
        if raw_task_type == "TRAKE":
            manual_items = _payload_sequence(payload, "manual_sequences")
            candidate_items = _payload_sequence(payload, "candidate_sequences")
            if not manual_items and "manual_items" in payload:
                manual_items = _payload_sequence(payload, "manual_items")
            if not candidate_items and "candidate_items" in payload:
                candidate_items = _payload_sequence(payload, "candidate_items")
        else:
            manual_items = _payload_sequence(payload, "manual_selections")
            candidate_items = _payload_sequence(payload, "candidate_reservoir")
            if not manual_items and "manual_items" in payload:
                manual_items = _payload_sequence(payload, "manual_items")
            if not candidate_items and "candidate_items" in payload:
                candidate_items = _payload_sequence(payload, "candidate_items")

        result = build_submission(
            raw_task_type,
            manual_items=manual_items,
            candidate_items=candidate_items,
            frame_lookup=frame_lookup,
            video_frames_lookup=video_frames_lookup,
            target_rows=raw_target_rows,
            vqa_answer=payload.get("vqa_answer"),
            event_count=payload.get("event_count"),
        )
    except SubmissionValidationError as exc:
        return {
            "ok": False,
            "task_type": raw_task_type,
            "mode": mode,
            "query_id": query_id,
            "target_rows": fallback_target,
            "row_count": 0,
            "complete": False,
            "valid_for_download": False,
            "missing_rows": fallback_target,
            "manual_row_count": 0,
            "reservoir_row_count": 0,
            "auto_filled_row_count": 0,
            "vqa_answer": None,
            "rows": [],
            "official_csv": _empty_official_csv(),
            "warnings": [],
            "errors": [str(exc)],
            "provenance_policy": _provenance_policy(),
        }

    result["mode"] = mode
    result["query_id"] = query_id
    for row in result["rows"]:
        row["task_type"] = result["task_type"]
        row["mode"] = mode
        row["query_id"] = query_id
    return result
