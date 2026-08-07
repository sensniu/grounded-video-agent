from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain import (
    CapabilityRequestContext,
    ChunkManifest,
    OCRManifest,
    TranscriptManifest,
    VisualDescriptionManifest,
)

DenseTextSource = TranscriptManifest | OCRManifest | VisualDescriptionManifest


@dataclass(frozen=True, slots=True)
class DenseIndexingRequest:
    source: DenseTextSource
    context: CapabilityRequestContext
    chunks: ChunkManifest | None = None

    def __post_init__(self) -> None:
        if self.chunks is not None and self.chunks.video_id != self.source.video_id:
            raise ValueError("chunks and dense-index source must belong to the same video")
