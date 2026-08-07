from __future__ import annotations

import math
from dataclasses import dataclass

from grounded_video_agent.domain import CapabilityRequestContext, RetrievalResult


@dataclass(frozen=True, slots=True)
class HybridRetrievalRequest:
    video_id: str
    query: str
    sparse: RetrievalResult
    dense: RetrievalResult
    context: CapabilityRequestContext
    top_k: int = 10
    rrf_k: int = 60
    sparse_weight: float = 1.0
    dense_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.video_id.strip() or not self.query.strip():
            raise ValueError("video_id and query must not be empty")
        if self.sparse.query != self.query or self.dense.query != self.query:
            raise ValueError("hybrid retrieval queries must match")
        if self.sparse.searched_modalities != self.dense.searched_modalities:
            raise ValueError("hybrid results must represent the same modality")
        if len(self.sparse.searched_modalities) != 1:
            raise ValueError("hybrid retrieval requires one modality")
        if self.top_k <= 0 or self.rrf_k <= 0:
            raise ValueError("top_k and rrf_k must be positive")
        for name in ("sparse_weight", "dense_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for result in (self.sparse, self.dense):
            if any(hit.item.video_id != self.video_id for hit in result.hits):
                raise ValueError("hybrid hits must belong to video_id")
