from __future__ import annotations

import math
from dataclasses import dataclass

from grounded_video_agent.domain import CapabilityRequestContext, FrameManifest


@dataclass(frozen=True, slots=True)
class OCRExtractionRequest:
    frames: FrameManifest
    context: CapabilityRequestContext
    language: str | None = None
    min_confidence: float = 0.5
    max_merge_gap_ms: int = 1_500
    min_text_similarity: float = 0.85
    min_bbox_iou: float = 0.25
    min_span_occurrences: int = 1

    def __post_init__(self) -> None:
        if self.language is not None and not self.language.strip():
            raise ValueError("language must not be empty")
        for name in ("min_confidence", "min_text_similarity", "min_bbox_iou"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.max_merge_gap_ms < 0:
            raise ValueError("max_merge_gap_ms must be non-negative")
        if self.min_span_occurrences <= 0:
            raise ValueError("min_span_occurrences must be positive")
