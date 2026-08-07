from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain import (
    CapabilityRequestContext,
    ChunkManifest,
    RetrievalResult,
    ShotManifest,
    TimeRange,
    TranscriptManifest,
)


@dataclass(frozen=True, slots=True)
class TimelineContextRequest:
    video_id: str
    chunks: ChunkManifest
    shots: ShotManifest
    transcript: TranscriptManifest
    context: CapabilityRequestContext
    retrieval: RetrievalResult | None = None
    ranges: tuple[TimeRange, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    adjacent_chunks: int = 0

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video_id must not be empty")
        manifests = (self.chunks, self.shots, self.transcript)
        if any(manifest.video_id != self.video_id for manifest in manifests):
            raise ValueError("all manifests must belong to video_id")
        if self.retrieval is None and not self.ranges and not self.chunk_ids:
            raise ValueError("retrieval, ranges, or chunk_ids are required")
        if tuple(sorted(self.ranges)) != self.ranges:
            raise ValueError("ranges must be ordered")
        if any(not item.strip() for item in self.chunk_ids):
            raise ValueError("chunk_ids must not contain empty values")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("chunk_ids must be unique")
        known_chunk_ids = {chunk.chunk_id for chunk in self.chunks.chunks}
        if not set(self.chunk_ids).issubset(known_chunk_ids):
            raise ValueError("chunk_ids must reference the chunk manifest")
        if self.adjacent_chunks < 0:
            raise ValueError("adjacent_chunks must be non-negative")
