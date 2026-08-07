"""Audio and clip artifacts derived from a source-video time range."""

from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain._invariants import (
    require_non_negative_int,
    require_positive_int,
    require_text,
)
from grounded_video_agent.domain.artifacts import ArtifactKind, ArtifactRef
from grounded_video_agent.domain.timeline import TimelineMapping, TimeRange


@dataclass(frozen=True, slots=True)
class AudioArtifact:
    audio_id: str
    video_id: str
    artifact: ArtifactRef
    source_range: TimeRange
    timeline_mapping: TimelineMapping
    stream_index: int
    sample_rate_hz: int
    channels: int

    def __post_init__(self) -> None:
        require_text(self.audio_id, "audio_id")
        require_text(self.video_id, "video_id")
        if self.artifact.kind is not ArtifactKind.AUDIO:
            raise ValueError("audio artifact must have kind AUDIO")
        require_non_negative_int(self.stream_index, "stream_index")
        require_positive_int(self.sample_rate_hz, "sample_rate_hz")
        require_positive_int(self.channels, "channels")
        if self.timeline_mapping.source_video_id != self.video_id:
            raise ValueError("timeline mapping must reference the source video")
        if self.timeline_mapping.derived_video_id != self.audio_id:
            raise ValueError("timeline mapping must reference the derived audio")
        if self.timeline_mapping.source_range != self.source_range:
            raise ValueError("timeline mapping source range must match source_range")


@dataclass(frozen=True, slots=True)
class VideoClipArtifact:
    clip_id: str
    video_id: str
    artifact: ArtifactRef
    requested_range: TimeRange
    actual_range: TimeRange
    timeline_mapping: TimelineMapping
    includes_audio: bool

    def __post_init__(self) -> None:
        require_text(self.clip_id, "clip_id")
        require_text(self.video_id, "video_id")
        if self.artifact.kind is not ArtifactKind.VIDEO_CLIP:
            raise ValueError("clip artifact must have kind VIDEO_CLIP")
        if self.timeline_mapping.source_video_id != self.video_id:
            raise ValueError("timeline mapping must reference the source video")
        if self.timeline_mapping.derived_video_id != self.clip_id:
            raise ValueError("timeline mapping must reference the derived clip")
        if self.timeline_mapping.source_range != self.actual_range:
            raise ValueError("timeline mapping source range must match actual_range")
