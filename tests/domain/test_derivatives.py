import pytest

from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    AudioArtifact,
    TimelineMapping,
    TimeRange,
    VideoClipArtifact,
)


def test_audio_artifact_maps_back_to_source_timeline() -> None:
    source_range = TimeRange(10_000, 20_000)
    mapping = TimelineMapping("video-1", source_range, "audio-1", TimeRange(0, 10_000))
    audio_ref = ArtifactRef("audio-file", ArtifactKind.AUDIO, "audio/sample.wav")

    audio = AudioArtifact(
        audio_id="audio-1",
        video_id="video-1",
        artifact=audio_ref,
        source_range=source_range,
        timeline_mapping=mapping,
        stream_index=1,
        sample_rate_hz=16_000,
        channels=1,
    )

    assert audio.timeline_mapping.to_source_timestamp(2_000) == 12_000


def test_clip_actual_range_must_match_timeline_mapping() -> None:
    clip_ref = ArtifactRef("clip-file", ArtifactKind.VIDEO_CLIP, "clips/sample.mp4")
    mapping = TimelineMapping(
        "video-1",
        TimeRange(10_000, 20_000),
        "clip-1",
        TimeRange(0, 10_000),
    )

    with pytest.raises(ValueError, match="actual_range"):
        VideoClipArtifact(
            clip_id="clip-1",
            video_id="video-1",
            artifact=clip_ref,
            requested_range=TimeRange(9_000, 19_000),
            actual_range=TimeRange(9_000, 19_000),
            timeline_mapping=mapping,
            includes_audio=True,
        )
