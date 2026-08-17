"""One-to-one region assignment for independent object clauses."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .models import FrameCandidate


def assign_objects(candidates: Sequence[FrameCandidate], object_count: int) -> dict[str, float]:
    """Return mean maximum matching score per frame, never reusing a region.

    Object count is small in decomposed AIC queries, so a compact DP is clearer and
    deterministic than depending on a heavyweight Hungarian solver.
    """
    grouped: dict[str, list[FrameCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.object_slot is not None and candidate.region_id:
            grouped[candidate.frame_uid].append(candidate)
    output: dict[str, float] = {}
    for uid, rows in grouped.items():
        table: dict[int, list[FrameCandidate]] = defaultdict(list)
        for row in rows:
            table[row.object_slot or 0].append(row)
        states: dict[frozenset[str], float] = {frozenset(): 0.0}
        for slot in range(object_count):
            next_states: dict[frozenset[str], float] = {}
            for used, current in states.items():
                for row in table.get(slot, []):
                    if row.region_id in used:
                        continue
                    key = used | {row.region_id}
                    next_states[key] = max(next_states.get(key, float("-inf")), current + row.score)
            states = next_states
            if not states:
                break
        if states:
            output[uid] = max(states.values()) / object_count
    return output
