import pytest

from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    FrameRef,
    SegmentKind,
    TimelineMapping,
    TimelineSegment,
    TimeRange,
)


@pytest.mark.parametrize("start_ms,end_ms", [(0, 0), (10, 5), (-1, 5), (False, 5)])
def test_time_range_rejects_invalid_bounds(start_ms: int, end_ms: int) -> None:
    with pytest.raises(ValueError):
        TimeRange(start_ms=start_ms, end_ms=end_ms)


def test_time_range_uses_half_open_boundaries() -> None:
    time_range = TimeRange(1_000, 2_000)

    assert time_range.contains_timestamp(1_000)
    assert time_range.contains_timestamp(1_999)
    assert not time_range.contains_timestamp(2_000)
    assert not time_range.overlaps(TimeRange(2_000, 3_000))


def test_time_range_intersection() -> None:
    first = TimeRange(1_000, 3_000)
    second = TimeRange(2_000, 4_000)

    assert first.intersection(second) == TimeRange(2_000, 3_000)


def test_timeline_segment_is_logical_and_parented() -> None:
    segment = TimelineSegment(
        segment_id="chunk-1",
        video_id="video-1",
        time_range=TimeRange(1_000, 3_000),
        kind=SegmentKind.CHUNK,
        parent_segment_id="shot-1",
    )

    assert segment.parent_segment_id == "shot-1"


def test_timeline_mapping_converts_clip_timestamp_to_source() -> None:
    mapping = TimelineMapping(
        source_video_id="video-1",
        source_range=TimeRange(10_000, 20_000),
        derived_video_id="clip-1",
        derived_range=TimeRange(0, 10_000),
    )

    assert mapping.to_source_timestamp(2_000) == 12_000
    assert mapping.to_derived_timestamp(15_000) == 5_000


def test_timeline_mapping_rejects_duration_change() -> None:
    with pytest.raises(ValueError, match="preserve duration"):
        TimelineMapping(
            source_video_id="video-1",
            source_range=TimeRange(10_000, 20_000),
            derived_video_id="clip-1",
            derived_range=TimeRange(0, 9_000),
        )


def test_frame_keeps_requested_and_actual_timestamps() -> None:
    image = ArtifactRef(
        artifact_id="image-1",
        kind=ArtifactKind.FRAME_IMAGE,
        uri="frames/1.jpg",
    )
    frame = FrameRef(
        frame_id="frame-1",
        video_id="video-1",
        timestamp_ms=10_042,
        requested_timestamp_ms=10_000,
        image=image,
        segment_ids=("shot-1",),
    )

    assert frame.timestamp_ms == 10_042
    assert frame.requested_timestamp_ms == 10_000
