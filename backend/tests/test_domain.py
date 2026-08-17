from aic_backend.retrieval.assignment import assign_objects
from aic_backend.retrieval.fusion import normalized_weights, temporal_nms, weighted_rrf
from aic_backend.retrieval.models import FrameCandidate
from aic_backend.retrieval.temporal import ordered_event_sequences


def row(
    video: str,
    frame: int,
    score: float,
    *,
    modality: str = "scene",
    region: str | None = None,
    slot: int | None = None,
):
    return FrameCandidate(
        video_id=video,
        frame_idx=frame,
        pts_time_s=float(frame),
        score=score,
        modality=modality,
        region_id=region,
        object_slot=slot,
    )


def test_weighted_rrf_and_temporal_nms_keep_canonical_frame_ids():
    hits = weighted_rrf(
        {
            "scene": [row("L21_V011", 24925, 0.9)],
            "ocr": [row("L21_V011", 24925, 0.8, modality="ocr")],
        },
        weights=normalized_weights(["scene", "ocr"]),
    )
    assert hits[0].video_id == "L21_V011"
    assert hits[0].frame_idx == 24925
    assert len(temporal_nms([*hits, hits[0]], seconds=1.0)) == 1


def test_object_assignment_never_reuses_one_region_for_two_objects():
    values = assign_objects(
        [
            row("V", 10, 0.9, modality="object", region="a", slot=0),
            row("V", 10, 0.8, modality="object", region="a", slot=1),
            row("V", 10, 0.7, modality="object", region="b", slot=1),
        ],
        2,
    )
    assert values["V:10"] == 0.8


def test_ordered_dp_requires_strictly_increasing_frames():
    first = weighted_rrf({"scene": [row("V", 10, 0.9)]}, weights={"scene": 1})
    second = weighted_rrf({"scene": [row("V", 9, 1.0), row("V", 11, 0.8)]}, weights={"scene": 1})
    sequences = ordered_event_sequences([first, second])
    assert [event.frame.frame_idx for event in sequences[0].events] == [10, 11]
