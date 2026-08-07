from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain import (
    CapabilityRequestContext,
    ChunkManifest,
    IndexKind,
    TranscriptManifest,
)


@dataclass(frozen=True, slots=True)
class TranscriptIndexingRequest:
    transcript: TranscriptManifest
    context: CapabilityRequestContext
    chunks: ChunkManifest | None = None
    index_kind: IndexKind = IndexKind.BM25

    def __post_init__(self) -> None:
        if self.chunks is not None and self.chunks.video_id != self.transcript.video_id:
            raise ValueError("chunks and transcript must belong to the same video")
        if self.index_kind is not IndexKind.BM25:
            raise ValueError("the local transcript index currently supports BM25")
