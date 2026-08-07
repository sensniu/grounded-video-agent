from __future__ import annotations

import math
from dataclasses import dataclass

from grounded_video_agent.domain import (
    CapabilityRequestContext,
    ChunkManifest,
    EvidenceModality,
    RetrievalResult,
    ShotManifest,
)


@dataclass(frozen=True, slots=True)
class ModalityWeight:
    modality: EvidenceModality
    weight: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("modality weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class CandidateFusionRequest:
    video_id: str
    query: str
    results: tuple[RetrievalResult, ...]
    chunks: ChunkManifest
    shots: ShotManifest
    context: CapabilityRequestContext
    top_k: int = 8
    rrf_k: int = 60
    max_gap_ms: int = 2_000
    max_window_ms: int = 30_000
    align_to_chunks: bool = True
    modality_weights: tuple[ModalityWeight, ...] = ()

    def __post_init__(self) -> None:
        if not self.video_id.strip() or not self.query.strip():
            raise ValueError("video_id and query must not be empty")
        if not self.results:
            raise ValueError("candidate fusion requires retrieval results")
        if self.chunks.video_id != self.video_id or self.shots.video_id != self.video_id:
            raise ValueError("candidate fusion manifests must belong to video_id")
        if any(result.query != self.query for result in self.results):
            raise ValueError("candidate fusion result queries must match")
        if any(
            hit.item.video_id != self.video_id
            for result in self.results
            for hit in result.hits
        ):
            raise ValueError("candidate fusion evidence must belong to video_id")
        if self.top_k <= 0 or self.rrf_k <= 0 or self.max_window_ms <= 0:
            raise ValueError("top_k, rrf_k, and max_window_ms must be positive")
        if self.max_gap_ms < 0:
            raise ValueError("max_gap_ms must be non-negative")
        modalities = tuple(item.modality for item in self.modality_weights)
        if len(set(modalities)) != len(modalities):
            raise ValueError("modality weights must be unique")
