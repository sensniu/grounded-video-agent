"""Resolved timeline context used by retrieval and inspection tools."""

from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain._invariants import require_text, require_unique_texts
from grounded_video_agent.domain.segmentation import Chunk, Shot
from grounded_video_agent.domain.timeline import TimeRange
from grounded_video_agent.domain.transcript import TranscriptSegment


@dataclass(frozen=True, slots=True)
class TimelineContext:
    video_id: str
    requested_ranges: tuple[TimeRange, ...]
    resolved_ranges: tuple[TimeRange, ...]
    chunks: tuple[Chunk, ...]
    shots: tuple[Shot, ...]
    transcript_segments: tuple[TranscriptSegment, ...]
    source_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.video_id, "video_id")
        require_unique_texts(self.source_evidence_ids, "source_evidence_ids")
        for collection in (self.chunks, self.shots, self.transcript_segments):
            if any(item.video_id != self.video_id for item in collection):
                raise ValueError("timeline context items must belong to video_id")
        if tuple(sorted(self.requested_ranges)) != self.requested_ranges:
            raise ValueError("requested_ranges must be ordered")
        if tuple(sorted(self.resolved_ranges)) != self.resolved_ranges:
            raise ValueError("resolved_ranges must be ordered")
        if tuple(sorted(self.chunks, key=lambda item: item.time_range)) != self.chunks:
            raise ValueError("chunks must be ordered")
        if tuple(sorted(self.shots, key=lambda item: item.time_range)) != self.shots:
            raise ValueError("shots must be ordered")
        if (
            tuple(sorted(self.transcript_segments, key=lambda item: item.time_range))
            != self.transcript_segments
        ):
            raise ValueError("transcript_segments must be ordered")
