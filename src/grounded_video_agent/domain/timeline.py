"""Canonical source-video timeline objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from grounded_video_agent.domain.artifacts import ArtifactKind, ArtifactRef


def _require_identifier(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_non_negative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True, order=True, slots=True)
class TimeRange:
    """A half-open interval ``[start_ms, end_ms)`` on a media timeline."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        _require_non_negative_int(self.start_ms, "start_ms")
        _require_non_negative_int(self.end_ms, "end_ms")
        if self.start_ms >= self.end_ms:
            raise ValueError("start_ms must be less than end_ms")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def contains_timestamp(self, timestamp_ms: int) -> bool:
        _require_non_negative_int(timestamp_ms, "timestamp_ms")
        return self.start_ms <= timestamp_ms < self.end_ms

    def contains_range(self, other: TimeRange) -> bool:
        return self.start_ms <= other.start_ms and other.end_ms <= self.end_ms

    def overlaps(self, other: TimeRange) -> bool:
        return self.start_ms < other.end_ms and other.start_ms < self.end_ms

    def intersection(self, other: TimeRange) -> TimeRange | None:
        start_ms = max(self.start_ms, other.start_ms)
        end_ms = min(self.end_ms, other.end_ms)
        if start_ms >= end_ms:
            return None
        return TimeRange(start_ms=start_ms, end_ms=end_ms)


class SegmentKind(StrEnum):
    """Semantic purpose of a range on the source-video timeline."""

    SHOT = "shot"
    CHUNK = "chunk"
    CANDIDATE = "candidate"
    INSPECTION = "inspection"
    EVIDENCE = "evidence"


@dataclass(frozen=True, slots=True)
class TimelineSegment:
    """A logical range; it does not imply that a clip file exists."""

    segment_id: str
    video_id: str
    time_range: TimeRange
    kind: SegmentKind
    parent_segment_id: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.segment_id, "segment_id")
        _require_identifier(self.video_id, "video_id")
        if self.parent_segment_id is not None:
            _require_identifier(self.parent_segment_id, "parent_segment_id")
            if self.parent_segment_id == self.segment_id:
                raise ValueError("a segment cannot be its own parent")


@dataclass(frozen=True, slots=True)
class TimelineMapping:
    """A duration-preserving mapping between source and derived timelines."""

    source_video_id: str
    source_range: TimeRange
    derived_video_id: str
    derived_range: TimeRange

    def __post_init__(self) -> None:
        _require_identifier(self.source_video_id, "source_video_id")
        _require_identifier(self.derived_video_id, "derived_video_id")
        if self.source_range.duration_ms != self.derived_range.duration_ms:
            raise ValueError("timeline mappings must preserve duration")

    def to_source_timestamp(self, derived_timestamp_ms: int) -> int:
        if not self.derived_range.contains_timestamp(derived_timestamp_ms):
            raise ValueError("derived timestamp is outside the mapped range")
        return self.source_range.start_ms + derived_timestamp_ms - self.derived_range.start_ms

    def to_derived_timestamp(self, source_timestamp_ms: int) -> int:
        if not self.source_range.contains_timestamp(source_timestamp_ms):
            raise ValueError("source timestamp is outside the mapped range")
        return self.derived_range.start_ms + source_timestamp_ms - self.source_range.start_ms


@dataclass(frozen=True, slots=True)
class FrameRef:
    """Reference to a decoded frame at its actual source-video timestamp."""

    frame_id: str
    video_id: str
    timestamp_ms: int
    image: ArtifactRef
    requested_timestamp_ms: int | None = None
    segment_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.frame_id, "frame_id")
        _require_identifier(self.video_id, "video_id")
        _require_non_negative_int(self.timestamp_ms, "timestamp_ms")
        if self.requested_timestamp_ms is not None:
            _require_non_negative_int(self.requested_timestamp_ms, "requested_timestamp_ms")
        if self.image.kind is not ArtifactKind.FRAME_IMAGE:
            raise ValueError("image must reference an artifact of kind FRAME_IMAGE")
        if any(not segment_id.strip() for segment_id in self.segment_ids):
            raise ValueError("segment_ids must not contain empty values")
        if len(set(self.segment_ids)) != len(self.segment_ids):
            raise ValueError("segment_ids must be unique")
