from __future__ import annotations

import math
from dataclasses import dataclass

from grounded_video_agent.domain import (
    CapabilityRequestContext,
    FrameManifest,
    IndexManifest,
    TimeRange,
)


@dataclass(frozen=True, slots=True)
class VisualRetrievalRequest:
    query: str
    index: IndexManifest
    context: CapabilityRequestContext
    top_k: int = 10
    min_score: float = 0.0
    within: TimeRange | None = None
    related_ids: tuple[str, ...] = ()
    required_tags: tuple[str, ...] = ()
    frames: FrameManifest | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not math.isfinite(self.min_score) or self.min_score < 0:
            raise ValueError("min_score must be finite and non-negative")
        filters = (("related_ids", self.related_ids), ("required_tags", self.required_tags))
        for name, values in filters:
            if any(not item.strip() for item in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique non-empty values")
        if self.frames is not None and self.frames.video_id != self.index.video_id:
            raise ValueError("frames and index must belong to the same video")
