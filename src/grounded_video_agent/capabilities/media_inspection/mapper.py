"""Map FFprobe-specific JSON to stable media domain objects."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from grounded_video_agent.domain import (
    AudioStreamInfo,
    ContainerInfo,
    FrameRate,
    MediaProbe,
    SubtitleStreamInfo,
    TimeBase,
    VideoStreamInfo,
)


class ProbeMappingError(ValueError):
    pass


def map_ffprobe_payload(video_id: str, payload: dict[str, Any]) -> MediaProbe:
    format_payload = payload.get("format", {})
    streams_payload = payload.get("streams", [])
    if not isinstance(format_payload, dict):
        raise ProbeMappingError("FFprobe 'format' field must be an object.")
    if not isinstance(streams_payload, list):
        raise ProbeMappingError("FFprobe 'streams' field must be an array.")

    container = ContainerInfo(
        format_names=_format_names(format_payload.get("format_name")),
        format_long_name=_optional_text(format_payload.get("format_long_name")),
        start_time_ms=_milliseconds(format_payload.get("start_time")),
        duration_ms=_milliseconds(format_payload.get("duration")),
        size_bytes=_integer(format_payload.get("size")),
        bit_rate=_integer(format_payload.get("bit_rate")),
        probe_score=_integer(format_payload.get("probe_score")),
        tags=_tags(format_payload.get("tags")),
    )

    video_streams: list[VideoStreamInfo] = []
    audio_streams: list[AudioStreamInfo] = []
    subtitle_streams: list[SubtitleStreamInfo] = []
    for raw_stream in streams_payload:
        if not isinstance(raw_stream, dict):
            raise ProbeMappingError("Each FFprobe stream must be an object.")
        codec_type = raw_stream.get("codec_type")
        if codec_type == "video":
            video_streams.append(_video_stream(raw_stream))
        elif codec_type == "audio":
            audio_streams.append(_audio_stream(raw_stream))
        elif codec_type == "subtitle":
            subtitle_streams.append(_subtitle_stream(raw_stream))

    try:
        return MediaProbe(
            video_id=video_id,
            container=container,
            video_streams=tuple(video_streams),
            audio_streams=tuple(audio_streams),
            subtitle_streams=tuple(subtitle_streams),
        )
    except ValueError as error:
        raise ProbeMappingError(str(error)) from error


def _video_stream(raw: dict[str, Any]) -> VideoStreamInfo:
    disposition = _mapping(raw.get("disposition"))
    try:
        return VideoStreamInfo(
            stream_index=_required_integer(raw.get("index"), "video stream index"),
            codec_name=_optional_text(raw.get("codec_name")),
            codec_long_name=_optional_text(raw.get("codec_long_name")),
            codec_profile=_optional_text(raw.get("profile")),
            width=_required_integer(raw.get("width"), "video width"),
            height=_required_integer(raw.get("height"), "video height"),
            frame_rate=_rational(raw.get("r_frame_rate"), FrameRate),
            average_frame_rate=_rational(raw.get("avg_frame_rate"), FrameRate),
            time_base=_rational(raw.get("time_base"), TimeBase),
            start_time_ms=_milliseconds(raw.get("start_time")),
            duration_ms=_milliseconds(raw.get("duration")),
            frame_count=_integer(raw.get("nb_frames")),
            bit_rate=_integer(raw.get("bit_rate")),
            pixel_format=_optional_text(raw.get("pix_fmt")),
            sample_aspect_ratio=_optional_ratio_text(raw.get("sample_aspect_ratio")),
            display_aspect_ratio=_optional_ratio_text(raw.get("display_aspect_ratio")),
            color_range=_optional_text(raw.get("color_range")),
            color_space=_optional_text(raw.get("color_space")),
            rotation_degrees=_rotation(raw),
            is_default=_flag(disposition.get("default")),
            is_attached_picture=_flag(disposition.get("attached_pic")),
            tags=_tags(raw.get("tags")),
        )
    except ValueError as error:
        raise ProbeMappingError(str(error)) from error


def _audio_stream(raw: dict[str, Any]) -> AudioStreamInfo:
    disposition = _mapping(raw.get("disposition"))
    tags = _mapping(raw.get("tags"))
    try:
        return AudioStreamInfo(
            stream_index=_required_integer(raw.get("index"), "audio stream index"),
            codec_name=_optional_text(raw.get("codec_name")),
            codec_long_name=_optional_text(raw.get("codec_long_name")),
            codec_profile=_optional_text(raw.get("profile")),
            sample_rate_hz=_integer(raw.get("sample_rate")),
            channels=_integer(raw.get("channels")),
            channel_layout=_optional_text(raw.get("channel_layout")),
            sample_format=_optional_text(raw.get("sample_fmt")),
            time_base=_rational(raw.get("time_base"), TimeBase),
            start_time_ms=_milliseconds(raw.get("start_time")),
            duration_ms=_milliseconds(raw.get("duration")),
            bit_rate=_integer(raw.get("bit_rate")),
            language=_optional_text(tags.get("language")),
            is_default=_flag(disposition.get("default")),
            tags=_tags(raw.get("tags")),
        )
    except ValueError as error:
        raise ProbeMappingError(str(error)) from error


def _subtitle_stream(raw: dict[str, Any]) -> SubtitleStreamInfo:
    disposition = _mapping(raw.get("disposition"))
    tags = _mapping(raw.get("tags"))
    try:
        return SubtitleStreamInfo(
            stream_index=_required_integer(raw.get("index"), "subtitle stream index"),
            codec_name=_optional_text(raw.get("codec_name")),
            language=_optional_text(tags.get("language")),
            title=_optional_text(tags.get("title")),
            time_base=_rational(raw.get("time_base"), TimeBase),
            start_time_ms=_milliseconds(raw.get("start_time")),
            duration_ms=_milliseconds(raw.get("duration")),
            is_default=_flag(disposition.get("default")),
            is_forced=_flag(disposition.get("forced")),
            tags=_tags(raw.get("tags")),
        )
    except ValueError as error:
        raise ProbeMappingError(str(error)) from error


def _required_integer(value: Any, field_name: str) -> int:
    parsed = _integer(value)
    if parsed is None:
        raise ProbeMappingError(f"Missing or invalid {field_name}.")
    return parsed


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _milliseconds(value: Any) -> int | None:
    if value is None:
        return None
    try:
        seconds = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not seconds.is_finite():
        return None
    return int((seconds * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _rational(value: Any, target_type: type[FrameRate] | type[TimeBase]) -> Any:
    if not isinstance(value, str) or "/" not in value:
        return None
    numerator_text, denominator_text = value.split("/", 1)
    try:
        numerator = int(numerator_text)
        denominator = int(denominator_text)
    except ValueError:
        return None
    if numerator <= 0 or denominator <= 0:
        return None
    return target_type(numerator=numerator, denominator=denominator)


def _format_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    return tuple(name.strip() for name in value.split(",") if name.strip())


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_ratio_text(value: Any) -> str | None:
    text = _optional_text(value)
    return None if text in {None, "N/A", "0:1"} else text


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _tags(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict):
        return ()
    return tuple(sorted((str(key), str(tag_value)) for key, tag_value in value.items()))


def _flag(value: Any) -> bool:
    return value in {1, "1", "true"}


def _rotation(raw: dict[str, Any]) -> int:
    tags = _mapping(raw.get("tags"))
    tagged_rotation = _integer(tags.get("rotate"))
    if tagged_rotation is not None:
        return tagged_rotation % 360
    side_data = raw.get("side_data_list", [])
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, dict):
                rotation = _integer(item.get("rotation"))
                if rotation is not None:
                    return rotation % 360
    return 0
