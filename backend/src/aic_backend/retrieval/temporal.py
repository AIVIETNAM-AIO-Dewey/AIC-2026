"""Beam-search ordered alignment for TRAKE event candidates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .models import EventFrame, SearchHit, TrakeSequence


def ordered_event_sequences(
    event_hits: Sequence[Sequence[SearchHit]], *, limit: int = 100, beam_width: int = 100
) -> list[TrakeSequence]:
    if not event_hits or any(not hits for hits in event_hits):
        return []
    by_video: dict[str, list[list[SearchHit]]] = defaultdict(lambda: [[] for _ in event_hits])
    for index, hits in enumerate(event_hits):
        for hit in hits:
            by_video[hit.video_id][index].append(hit)
    sequences: list[TrakeSequence] = []
    for video_id, per_event in by_video.items():
        if any(not values for values in per_event):
            continue
        paths: list[tuple[float, tuple[SearchHit, ...]]] = [(0.0, ())]
        for candidates in per_event:
            next_paths: list[tuple[float, tuple[SearchHit, ...]]] = []
            for score, path in paths:
                last_idx = path[-1].frame_idx if path else -1
                for candidate in candidates:
                    if candidate.frame_idx > last_idx:
                        next_paths.append((score + candidate.score, (*path, candidate)))
            paths = sorted(next_paths, key=lambda item: -item[0])[:beam_width]
            if not paths:
                break
        for score, path in paths:
            sequences.append(
                TrakeSequence(
                    video_id=video_id,
                    score=score / len(event_hits),
                    events=tuple(EventFrame(index, hit) for index, hit in enumerate(path)),
                )
            )
    return sorted(sequences, key=lambda item: (-item.score, item.video_id))[:limit]
