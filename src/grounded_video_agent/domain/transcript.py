"""Unified transcript records for embedded subtitles and ASR output."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from grounded_video_agent.domain._invariants import (
    require_manifest,
    require_non_negative_int,
    require_probability,
    require_text,
    require_unique_texts,
)
from grounded_video_agent.domain.artifacts import ManifestKind, ManifestRef
from grounded_video_agent.domain.timeline import TimeRange


class TranscriptSource(StrEnum):
    EMBEDDED_SUBTITLE = "embedded_subtitle"
    ASR = "asr"


@dataclass(frozen=True, slots=True)
class TranscriptWord:
    text: str
    time_range: TimeRange
    confidence: float | None = None

    def __post_init__(self) -> None:
        require_text(self.text, "text")
        require_probability(self.confidence)


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    segment_id: str
    video_id: str
    time_range: TimeRange
    raw_text: str
    normalized_text: str
    source: TranscriptSource
    language: str | None = None
    confidence: float | None = None
    words: tuple[TranscriptWord, ...] = ()
    source_stream_index: int | None = None

    def __post_init__(self) -> None:
        require_text(self.segment_id, "segment_id")
        require_text(self.video_id, "video_id")
        require_text(self.raw_text, "raw_text")
        require_text(self.normalized_text, "normalized_text")
        if self.language is not None:
            require_text(self.language, "language")
        require_probability(self.confidence)
        if self.source_stream_index is not None:
            require_non_negative_int(self.source_stream_index, "source_stream_index")
        if any(not self.time_range.contains_range(word.time_range) for word in self.words):
            raise ValueError("word ranges must be contained by the transcript segment")
        if tuple(sorted(self.words, key=lambda word: word.time_range)) != self.words:
            raise ValueError("words must be ordered by time range")


@dataclass(frozen=True, slots=True)
class TranscriptManifest:
    ref: ManifestRef
    video_id: str
    source: TranscriptSource
    segments: tuple[TranscriptSegment, ...]
    language: str | None = None

    def __post_init__(self) -> None:
        require_text(self.video_id, "video_id")
        if self.language is not None:
            require_text(self.language, "language")
        require_manifest(
            self.ref,
            kind=ManifestKind.TRANSCRIPT,
            video_id=self.video_id,
            item_count=len(self.segments),
        )
        require_unique_texts((segment.segment_id for segment in self.segments), "segment_ids")
        if any(segment.video_id != self.video_id for segment in self.segments):
            raise ValueError("all transcript segments must belong to the manifest video")
        if any(segment.source is not self.source for segment in self.segments):
            raise ValueError("all transcript segments must match the manifest source")
        if tuple(sorted(self.segments, key=lambda item: item.time_range)) != self.segments:
            raise ValueError("transcript segments must be ordered by time range")
