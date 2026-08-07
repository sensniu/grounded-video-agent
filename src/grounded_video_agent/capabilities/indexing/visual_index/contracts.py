from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain import (
    CapabilityRequestContext,
    ChunkManifest,
    IndexKind,
    ShotManifest,
    VisualDescriptionManifest,
)


@dataclass(frozen=True, slots=True)
class VisualIndexingRequest:
    descriptions: VisualDescriptionManifest
    context: CapabilityRequestContext
    chunks: ChunkManifest | None = None
    shots: ShotManifest | None = None
    index_kind: IndexKind = IndexKind.BM25

    def __post_init__(self) -> None:
        for manifest in (self.chunks, self.shots):
            if manifest is not None and manifest.video_id != self.descriptions.video_id:
                raise ValueError("all manifests must belong to the same video")
        if self.index_kind is not IndexKind.BM25:
            raise ValueError("the local visual index currently supports BM25")
