from __future__ import annotations

import math
from dataclasses import dataclass

from grounded_video_agent.domain import CapabilityRequestContext, IndexManifest, TimeRange


@dataclass(frozen=True, slots=True)
class DenseRetrievalRequest:
    query: str
    index: IndexManifest
    context: CapabilityRequestContext
    top_k: int = 10
    min_score: float = 0.0
    within: TimeRange | None = None
    required_source_ids: tuple[str, ...] = ()
    required_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not math.isfinite(self.min_score) or not -1 <= self.min_score <= 1:
            raise ValueError("min_score must be finite and between minus one and one")
        for name in ("required_source_ids", "required_tags"):
            values = getattr(self, name)
            if any(not item.strip() for item in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique non-empty values")
