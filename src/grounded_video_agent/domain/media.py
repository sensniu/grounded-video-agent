"""Source media identity and objective probe information."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from grounded_video_agent.domain.artifacts import ArtifactKind, ArtifactRef


def _require_text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_optional_non_negative_int(value: int | None, field_name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{field_name} must be a non-negative integer when provided")


def _require_optional_int(value: int | None, field_name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"{field_name} must be an integer when provided")


@dataclass(frozen=True, slots=True)
class FrameRate:
    """An exact rational frame rate, such as 30000/1001."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        _require_positive_int(self.numerator, "numerator")
        _require_positive_int(self.denominator, "denominator")

    @property
    def frames_per_second(self) -> float:
        return self.numerator / self.denominator


@dataclass(frozen=True, slots=True)
class TimeBase:
    """Duration in seconds represented by one stream timestamp tick."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        _require_positive_int(self.numerator, "numerator")
        _require_positive_int(self.denominator, "denominator")

    @property
    def seconds_per_tick(self) -> float:
        return self.numerator / self.denominator


@dataclass(frozen=True, slots=True)
class VideoAsset:
    """Stable identity and source reference for a registered video."""

    video_id: str
    source: ArtifactRef
    display_name: str | None = None
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        _require_text(self.video_id, "video_id")
        if self.source.kind not in {ArtifactKind.SOURCE_VIDEO, ArtifactKind.NORMALIZED_VIDEO}:
            raise ValueError("source must reference a source or normalized video artifact")
        if self.display_name is not None:
            _require_text(self.display_name, "display_name")
        if self.registered_at.tzinfo is None or self.registered_at.utcoffset() is None:
            raise ValueError("registered_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class ContainerInfo:
    """Objective container-level facts returned by media probing."""

    format_names: tuple[str, ...]
    format_long_name: str | None = None
    start_time_ms: int | None = None
    duration_ms: int | None = None
    size_bytes: int | None = None
    bit_rate: int | None = None
    probe_score: int | None = None
    tags: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if any(not name.strip() for name in self.format_names):
            raise ValueError("format_names must not contain empty values")
        if self.format_long_name is not None:
            _require_text(self.format_long_name, "format_long_name")
        _require_optional_int(self.start_time_ms, "start_time_ms")
        _require_optional_non_negative_int(self.duration_ms, "duration_ms")
        _require_optional_non_negative_int(self.size_bytes, "size_bytes")
        _require_optional_non_negative_int(self.bit_rate, "bit_rate")
        _require_optional_non_negative_int(self.probe_score, "probe_score")
        if any(not key.strip() for key, _ in self.tags):
            raise ValueError("tag keys must not be empty")


@dataclass(frozen=True, slots=True)
class VideoStreamInfo:
    """Objective facts about one encoded video stream."""

    stream_index: int
    codec_name: str | None
    width: int
    height: int
    codec_long_name: str | None = None
    codec_profile: str | None = None
    frame_rate: FrameRate | None = None
    average_frame_rate: FrameRate | None = None
    time_base: TimeBase | None = None
    start_time_ms: int | None = None
    duration_ms: int | None = None
    frame_count: int | None = None
    bit_rate: int | None = None
    pixel_format: str | None = None
    sample_aspect_ratio: str | None = None
    display_aspect_ratio: str | None = None
    color_range: str | None = None
    color_space: str | None = None
    rotation_degrees: int = 0
    is_default: bool = False
    is_attached_picture: bool = False
    tags: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_optional_non_negative_int(self.stream_index, "stream_index")
        _require_positive_int(self.width, "width")
        _require_positive_int(self.height, "height")
        _require_optional_int(self.start_time_ms, "start_time_ms")
        _require_optional_non_negative_int(self.duration_ms, "duration_ms")
        _require_optional_non_negative_int(self.frame_count, "frame_count")
        _require_optional_non_negative_int(self.bit_rate, "bit_rate")
        for field_name in (
            "codec_name",
            "codec_long_name",
            "codec_profile",
            "pixel_format",
            "sample_aspect_ratio",
            "display_aspect_ratio",
            "color_range",
            "color_space",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_text(value, field_name)
        if any(not key.strip() for key, _ in self.tags):
            raise ValueError("tag keys must not be empty")

    @property
    def is_variable_frame_rate(self) -> bool:
        if self.frame_rate is None or self.average_frame_rate is None:
            return False
        return (
            abs(self.frame_rate.frames_per_second - self.average_frame_rate.frames_per_second)
            > 0.001
        )


@dataclass(frozen=True, slots=True)
class AudioStreamInfo:
    """Objective facts about one encoded audio stream."""

    stream_index: int
    codec_name: str | None
    codec_long_name: str | None = None
    codec_profile: str | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    channel_layout: str | None = None
    sample_format: str | None = None
    time_base: TimeBase | None = None
    start_time_ms: int | None = None
    duration_ms: int | None = None
    bit_rate: int | None = None
    language: str | None = None
    is_default: bool = False
    tags: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_optional_non_negative_int(self.stream_index, "stream_index")
        if self.sample_rate_hz is not None:
            _require_positive_int(self.sample_rate_hz, "sample_rate_hz")
        if self.channels is not None:
            _require_positive_int(self.channels, "channels")
        _require_optional_int(self.start_time_ms, "start_time_ms")
        _require_optional_non_negative_int(self.duration_ms, "duration_ms")
        _require_optional_non_negative_int(self.bit_rate, "bit_rate")
        for field_name in (
            "codec_name",
            "codec_long_name",
            "codec_profile",
            "channel_layout",
            "sample_format",
            "language",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_text(value, field_name)
        if any(not key.strip() for key, _ in self.tags):
            raise ValueError("tag keys must not be empty")


@dataclass(frozen=True, slots=True)
class SubtitleStreamInfo:
    """Objective facts about one embedded subtitle stream."""

    stream_index: int
    codec_name: str | None
    language: str | None = None
    title: str | None = None
    time_base: TimeBase | None = None
    start_time_ms: int | None = None
    duration_ms: int | None = None
    is_default: bool = False
    is_forced: bool = False
    tags: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_optional_non_negative_int(self.stream_index, "stream_index")
        _require_optional_int(self.start_time_ms, "start_time_ms")
        _require_optional_non_negative_int(self.duration_ms, "duration_ms")
        for field_name in ("codec_name", "language", "title"):
            value = getattr(self, field_name)
            if value is not None:
                _require_text(value, field_name)
        if any(not key.strip() for key, _ in self.tags):
            raise ValueError("tag keys must not be empty")


@dataclass(frozen=True, slots=True)
class MediaProbe:
    """Probe facts kept separate from any support or validity decision."""

    video_id: str
    container: ContainerInfo
    video_streams: tuple[VideoStreamInfo, ...] = ()
    audio_streams: tuple[AudioStreamInfo, ...] = ()
    subtitle_streams: tuple[SubtitleStreamInfo, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.video_id, "video_id")
        indexes = [
            *(stream.stream_index for stream in self.video_streams),
            *(stream.stream_index for stream in self.audio_streams),
            *(stream.stream_index for stream in self.subtitle_streams),
        ]
        if len(indexes) != len(set(indexes)):
            raise ValueError("stream indexes must be unique within a media probe")

    @property
    def primary_video_stream(self) -> VideoStreamInfo | None:
        playable_streams = tuple(
            stream for stream in self.video_streams if not stream.is_attached_picture
        )
        return next(
            (stream for stream in playable_streams if stream.is_default),
            playable_streams[0] if playable_streams else None,
        )

    @property
    def primary_audio_stream(self) -> AudioStreamInfo | None:
        return next(
            (stream for stream in self.audio_streams if stream.is_default),
            self.audio_streams[0] if self.audio_streams else None,
        )

    @property
    def primary_subtitle_stream(self) -> SubtitleStreamInfo | None:
        return next(
            (stream for stream in self.subtitle_streams if stream.is_default),
            self.subtitle_streams[0] if self.subtitle_streams else None,
        )
