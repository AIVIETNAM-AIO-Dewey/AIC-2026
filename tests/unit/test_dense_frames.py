from __future__ import annotations

import pytest

from aic2026.scene_embedding.dense_frames import advance_sampling_clock


def test_timestamp_sampling_uses_pts_not_rounded_frame_math() -> None:
    target = 0.0
    selected = []
    for pts in (0.0, 0.041, 0.199, 0.205, 0.399, 0.407):
        keep, target = advance_sampling_clock(pts, target, 5.0)
        if keep:
            selected.append(pts)
    assert selected == [0.0, 0.205, 0.407]


def test_timestamp_sampling_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError, match="sampling_fps"):
        advance_sampling_clock(0.0, 0.0, 0.0)
