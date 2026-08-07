"""Stable, JSON-serializable contracts exposed by the video tools."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeVar, cast

from grounded_video_agent.capabilities._support import json_value
from grounded_video_agent.domain import CapabilityUsage, TimeRange

T = TypeVar("T")


class ToolStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ToolError:
    code: str
    message: str
    retryable: bool = False
    suggested_action: str | None = None

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.message.strip():
            raise ValueError("tool error code and message must not be empty")


@dataclass(frozen=True, slots=True)
class EvidenceDelta:
    new_evidence_ids: tuple[str, ...] = ()
    reused_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolProgress:
    new_candidate_count: int = 0
    new_evidence_count: int = 0
    newly_covered_ranges: tuple[TimeRange, ...] = ()
    cache_hit: bool = False
    exhausted: bool = False
    no_information_gain: bool = False

    def __post_init__(self) -> None:
        if self.new_candidate_count < 0 or self.new_evidence_count < 0:
            raise ValueError("tool progress counts must be non-negative")


@dataclass(frozen=True, slots=True)
class ToolResult(Generic[T]):
    schema_version: str
    call_id: str
    status: ToolStatus
    data: T | None
    evidence_delta: EvidenceDelta = field(default_factory=EvidenceDelta)
    progress: ToolProgress = field(default_factory=ToolProgress)
    warnings: tuple[str, ...] = ()
    error: ToolError | None = None
    usage: CapabilityUsage = field(default_factory=CapabilityUsage)

    def __post_init__(self) -> None:
        if not self.schema_version.strip() or not self.call_id.strip():
            raise ValueError("schema_version and call_id must not be empty")
        if self.status is ToolStatus.FAILED:
            if self.data is not None or self.error is None:
                raise ValueError("failed tool results require an error and no data")
        elif self.data is None or self.error is not None:
            raise ValueError("successful or partial tool results require data and no error")
        if any(not warning.strip() for warning in self.warnings):
            raise ValueError("tool warnings must not contain empty values")

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json_value(self))

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GetVideoMetadataInput:
    include_streams: bool = True


@dataclass(frozen=True, slots=True)
class StreamSummary:
    stream_index: int
    kind: str
    codec: str | None
    language: str | None = None
    width: int | None = None
    height: int | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    is_default: bool = False


@dataclass(frozen=True, slots=True)
class VideoMetadataOutput:
    video_id: str
    display_name: str | None
    duration_ms: int | None
    format_names: tuple[str, ...]
    width: int | None
    height: int | None
    frame_rate: float | None
    validation_status: str
    processable: bool
    next_action: str
    has_audio: bool
    has_embedded_subtitles: bool
    transcript_ready: bool
    timeline_ready: bool
    sparse_search_ready: bool
    dense_search_ready: bool
    streams: tuple[StreamSummary, ...] = ()
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchVideoTranscriptInput:
    query: str
    top_k: int = 5
    within: TimeRange | None = None
    intent_id: str | None = None
    language: str | None = None
    min_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= self.top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if self.intent_id is not None and not self.intent_id.strip():
            raise ValueError("intent_id must not be empty")
        if self.language is not None and not self.language.strip():
            raise ValueError("language must not be empty")
        if self.min_score < 0:
            raise ValueError("min_score must be non-negative")


@dataclass(frozen=True, slots=True)
class TranscriptCandidate:
    candidate_id: str
    chunk_id: str
    text: str
    time_range: TimeRange
    inspection_range: TimeRange
    shot_ids: tuple[str, ...]
    evidence_id: str
    scores: tuple[tuple[str, float], ...]
    matched_queries: tuple[str, ...]
    context_may_be_needed: bool


@dataclass(frozen=True, slots=True)
class TranscriptSearchOutput:
    query: str
    retrieval_mode: str
    new_hits: tuple[TranscriptCandidate, ...]
    reused_hits: tuple[TranscriptCandidate, ...]
    exhausted: bool


class ContextDirection(StrEnum):
    BEFORE = "before"
    AFTER = "after"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class ResolveTimelineContextInput:
    candidate_ids: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    ranges: tuple[TimeRange, ...] = ()
    direction: ContextDirection = ContextDirection.BOTH
    adjacent_chunks: int = 1
    max_duration_ms: int = 120_000

    def __post_init__(self) -> None:
        if not self.candidate_ids and not self.chunk_ids and not self.ranges:
            raise ValueError("candidate_ids, chunk_ids, or ranges are required")
        for name in ("candidate_ids", "chunk_ids"):
            values = getattr(self, name)
            if any(not value.strip() for value in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique non-empty values")
        if self.adjacent_chunks < 0 or self.adjacent_chunks > 20:
            raise ValueError("adjacent_chunks must be between 0 and 20")
        if self.max_duration_ms <= 0:
            raise ValueError("max_duration_ms must be positive")


@dataclass(frozen=True, slots=True)
class SubtitleExcerpt:
    segment_id: str
    time_range: TimeRange
    text: str
    evidence_id: str


@dataclass(frozen=True, slots=True)
class TimelineContextOutput:
    context_window_id: str
    requested_ranges: tuple[TimeRange, ...]
    resolved_ranges: tuple[TimeRange, ...]
    chunk_ids: tuple[str, ...]
    shot_ids: tuple[str, ...]
    subtitles: tuple[SubtitleExcerpt, ...]
    source_evidence_ids: tuple[str, ...]


class VisualDetail(StrEnum):
    COARSE = "coarse"
    STANDARD = "standard"
    DETAILED = "detailed"


@dataclass(frozen=True, slots=True)
class InspectVisualContentInput:
    question: str
    candidate_ids: tuple[str, ...] = ()
    context_window_ids: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    ranges: tuple[TimeRange, ...] = ()
    detail: VisualDetail = VisualDetail.STANDARD
    focus: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if not any((self.candidate_ids, self.context_window_ids, self.chunk_ids, self.ranges)):
            raise ValueError("at least one visual target is required")
        if any(not item.strip() for item in self.focus):
            raise ValueError("focus values must not be empty")


@dataclass(frozen=True, slots=True)
class FrameObservation:
    frame_id: str
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class VisualObservationOutput:
    evidence_id: str
    target_id: str
    time_range: TimeRange
    text: str
    frame_ids: tuple[str, ...]
    frame_timestamps_ms: tuple[int, ...]
    tags: tuple[str, ...]
    confidence: float | None


@dataclass(frozen=True, slots=True)
class VisualInspectionOutput:
    inspected_ranges: tuple[TimeRange, ...]
    frames: tuple[FrameObservation, ...]
    observations: tuple[VisualObservationOutput, ...]
    reused_frames: bool
    reused_analysis: bool


@dataclass(frozen=True, slots=True)
class ReadScreenTextInput:
    candidate_ids: tuple[str, ...] = ()
    context_window_ids: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    ranges: tuple[TimeRange, ...] = ()
    frame_ids: tuple[str, ...] = ()
    language: str | None = None
    detail: VisualDetail = VisualDetail.DETAILED
    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        if not any(
            (
                self.candidate_ids,
                self.context_window_ids,
                self.chunk_ids,
                self.ranges,
                self.frame_ids,
            )
        ):
            raise ValueError("at least one OCR target is required")
        if self.language is not None and not self.language.strip():
            raise ValueError("language must not be empty")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class ScreenTextSpan:
    evidence_id: str
    text: str
    time_range: TimeRange
    frame_ids: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class ScreenTextOutput:
    inspected_ranges: tuple[TimeRange, ...]
    spans: tuple[ScreenTextSpan, ...]
    frame_ids: tuple[str, ...]
    reused_frames: bool
    reused_ocr: bool


@dataclass(frozen=True, slots=True)
class ScanVideoTimelineInput:
    question: str
    max_windows: int = 8
    detail: VisualDetail = VisualDetail.COARSE

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if not 1 <= self.max_windows <= 32:
            raise ValueError("max_windows must be between 1 and 32")


@dataclass(frozen=True, slots=True)
class TimelineScanCandidate:
    context_window_id: str
    time_range: TimeRange
    summary: str
    evidence_ids: tuple[str, ...]
    frame_timestamps_ms: tuple[int, ...]
    confidence: float | None


@dataclass(frozen=True, slots=True)
class TimelineScanOutput:
    candidates: tuple[TimelineScanCandidate, ...]
    covered_ranges: tuple[TimeRange, ...]
    unseen_ranges: tuple[TimeRange, ...]
    coverage_ratio: float
    exhausted: bool


class ClipGrouping(StrEnum):
    AUTO = "auto"
    SEPARATE = "separate"


@dataclass(frozen=True, slots=True)
class ExportEvidenceClipInput:
    evidence_ids: tuple[str, ...]
    context_window_ids: tuple[str, ...] = ()
    include_audio: bool = True
    grouping: ClipGrouping = ClipGrouping.AUTO
    padding_before_ms: int = 1_500
    padding_after_ms: int = 1_500

    def __post_init__(self) -> None:
        for name in ("evidence_ids", "context_window_ids"):
            values = getattr(self, name)
            if any(not value.strip() for value in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique non-empty values")
        if not self.evidence_ids:
            raise ValueError("evidence_ids must not be empty")
        if self.padding_before_ms < 0 or self.padding_after_ms < 0:
            raise ValueError("clip padding must be non-negative")


@dataclass(frozen=True, slots=True)
class EvidenceClipDelivery:
    delivery_id: str
    artifact_id: str
    catalog_entry_id: str
    filename: str
    requested_range: TimeRange
    actual_range: TimeRange
    duration_ms: int
    includes_audio: bool
    evidence_ids: tuple[str, ...]
    size_bytes: int
    reused: bool = False


@dataclass(frozen=True, slots=True)
class ClipExportFailure:
    evidence_ids: tuple[str, ...]
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ExportEvidenceClipOutput:
    export_id: str
    clips: tuple[EvidenceClipDelivery, ...]
    failures: tuple[ClipExportFailure, ...] = ()
    total_duration_ms: int = 0
