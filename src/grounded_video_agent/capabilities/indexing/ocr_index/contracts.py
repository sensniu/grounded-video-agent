from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain import (
    CapabilityRequestContext,
    ChunkManifest,
    IndexKind,
    OCRManifest,
)


@dataclass(frozen=True, slots=True)
class OCRIndexingRequest:
    ocr: OCRManifest
    context: CapabilityRequestContext
    chunks: ChunkManifest | None = None
    index_kind: IndexKind = IndexKind.BM25

    def __post_init__(self) -> None:
        if self.chunks is not None and self.chunks.video_id != self.ocr.video_id:
            raise ValueError("chunks and OCR records must belong to the same video")
        if self.index_kind is not IndexKind.BM25:
            raise ValueError("the local OCR text index currently supports BM25")
