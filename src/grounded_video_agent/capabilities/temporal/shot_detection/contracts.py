from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain import CapabilityRequestContext, TimeRange, VideoAsset


@dataclass(frozen=True, slots=True)
class ShotDetectionRequest:
    video_asset: VideoAsset
    source_range: TimeRange
    context: CapabilityRequestContext
    threshold: float = 27.0
    min_shot_duration_ms: int = 500

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if self.min_shot_duration_ms <= 0:
            raise ValueError("min_shot_duration_ms must be positive")
