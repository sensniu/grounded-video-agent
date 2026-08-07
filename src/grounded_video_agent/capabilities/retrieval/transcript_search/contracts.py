from __future__ import annotations

import math
from dataclasses import dataclass

from grounded_video_agent.domain import CapabilityRequestContext, IndexManifest, TimeRange


@dataclass(frozen=True, slots=True)
class TranscriptRetrievalRequest:
    query: str
    index: IndexManifest
    context: CapabilityRequestContext
    top_k: int = 10
    min_score: float = 0.0
    within: TimeRange | None = None
    chunk_ids: tuple[str, ...] = ()
    language: str | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not math.isfinite(self.min_score) or self.min_score < 0:
            raise ValueError("min_score must be finite and non-negative")
        if any(not item.strip() for item in self.chunk_ids):
            raise ValueError("chunk_ids must not contain empty values")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("chunk_ids must be unique")
        if self.language is not None and not self.language.strip():
            raise ValueError("language must not be empty")
