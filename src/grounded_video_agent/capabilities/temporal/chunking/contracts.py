from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain import (
    CapabilityRequestContext,
    ShotManifest,
    TimeRange,
    TranscriptManifest,
)


@dataclass(frozen=True, slots=True)
class TemporalChunkingRequest:
    video_id: str
    source_range: TimeRange
    shots: ShotManifest | None
    context: CapabilityRequestContext
    transcript: TranscriptManifest | None = None
    target_duration_ms: int = 15_000
    max_duration_ms: int = 30_000
    overlap_ms: int = 2_000
    target_characters: int = 240
    max_characters: int = 480
    max_silence_gap_ms: int = 2_500
    context_padding_ms: int = 500
    max_inspection_duration_ms: int = 60_000
    align_to_shots: bool = True

    def __post_init__(self) -> None:
        if self.shots is not None and self.shots.video_id != self.video_id:
            raise ValueError("shot manifest must belong to video_id")
        if self.transcript is not None and self.transcript.video_id != self.video_id:
            raise ValueError("transcript manifest must belong to video_id")
        if not 0 < self.target_duration_ms <= self.max_duration_ms:
            raise ValueError("target duration must be positive and at most max duration")
        if not 0 <= self.overlap_ms < self.target_duration_ms:
            raise ValueError("overlap must be non-negative and less than target duration")
        if not 0 < self.target_characters <= self.max_characters:
            raise ValueError("target characters must be positive and at most max characters")
        if self.max_silence_gap_ms < 0:
            raise ValueError("max_silence_gap_ms must be non-negative")
        if self.context_padding_ms < 0:
            raise ValueError("context_padding_ms must be non-negative")
        if self.max_inspection_duration_ms < self.max_duration_ms:
            raise ValueError("max inspection duration must be at least max chunk duration")
