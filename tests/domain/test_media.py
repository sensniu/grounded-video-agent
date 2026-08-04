from datetime import UTC, datetime

import pytest

from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    AudioStreamInfo,
    ContainerInfo,
    FrameRate,
    MediaProbe,
    TimeBase,
    VideoAsset,
    VideoStreamInfo,
)


def test_frame_rate_preserves_fractional_rate() -> None:
    frame_rate = FrameRate(numerator=30_000, denominator=1_001)

    assert frame_rate.frames_per_second == pytest.approx(29.97002997)


def test_video_asset_requires_a_video_artifact() -> None:
    image = ArtifactRef(
        artifact_id="image-1",
        kind=ArtifactKind.FRAME_IMAGE,
        uri="frames/1.jpg",
    )

    with pytest.raises(ValueError, match="source or normalized video"):
        VideoAsset(video_id="video-1", source=image)


def test_video_asset_requires_timezone_aware_registration_time() -> None:
    source = ArtifactRef(
        artifact_id="source-1",
        kind=ArtifactKind.SOURCE_VIDEO,
        uri="videos/source.mp4",
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        VideoAsset(
            video_id="video-1",
            source=source,
            registered_at=datetime(2026, 8, 3),
        )


def test_media_probe_selects_default_streams() -> None:
    first_video = VideoStreamInfo(
        stream_index=0,
        codec_name="h264",
        width=1920,
        height=1080,
        frame_rate=FrameRate(25, 1),
        time_base=TimeBase(1, 90_000),
    )
    default_video = VideoStreamInfo(
        stream_index=2,
        codec_name="h264",
        width=1280,
        height=720,
        is_default=True,
    )
    audio = AudioStreamInfo(
        stream_index=1,
        codec_name="aac",
        sample_rate_hz=48_000,
        channels=2,
        is_default=True,
    )
    probe = MediaProbe(
        video_id="video-1",
        container=ContainerInfo(format_names=("mov", "mp4"), duration_ms=30_000),
        video_streams=(first_video, default_video),
        audio_streams=(audio,),
    )

    assert probe.primary_video_stream == default_video
    assert probe.primary_audio_stream == audio


def test_media_probe_rejects_duplicate_global_stream_indexes() -> None:
    video = VideoStreamInfo(stream_index=0, codec_name="h264", width=1920, height=1080)
    audio = AudioStreamInfo(stream_index=0, codec_name="aac")

    with pytest.raises(ValueError, match="stream indexes"):
        MediaProbe(
            video_id="video-1",
            container=ContainerInfo(format_names=("mp4",)),
            video_streams=(video,),
            audio_streams=(audio,),
        )


def test_video_asset_is_immutable() -> None:
    source = ArtifactRef(
        artifact_id="source-1",
        kind=ArtifactKind.SOURCE_VIDEO,
        uri="videos/source.mp4",
    )
    asset = VideoAsset(
        video_id="video-1",
        source=source,
        registered_at=datetime(2026, 8, 3, tzinfo=UTC),
    )

    with pytest.raises(AttributeError):
        asset.video_id = "other"  # type: ignore[misc]
