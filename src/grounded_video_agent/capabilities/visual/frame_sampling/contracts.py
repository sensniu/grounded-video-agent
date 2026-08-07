from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain import (
    CapabilityRequestContext,
    FrameSamplingStrategy,
    ShotManifest,
    TimeRange,
    VideoAsset,
)


@dataclass(frozen=True, slots=True)
class FrameSamplingRequest:
    video_asset: VideoAsset
    ranges: tuple[TimeRange, ...]
    strategy: FrameSamplingStrategy
    context: CapabilityRequestContext
    max_frames: int = 24
    fps: float | None = None
    shots: ShotManifest | None = None
    jpeg_quality: int = 90
    deduplicate: bool = True

    def __post_init__(self) -> None:
        if not self.ranges:
            raise ValueError("ranges must not be empty")
        if tuple(sorted(self.ranges)) != self.ranges:
            raise ValueError("ranges must be ordered")
        if any(
            first.overlaps(second)
            for first, second in zip(self.ranges, self.ranges[1:], strict=False)
        ):
            raise ValueError("ranges must not overlap")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        fps_strategies = {
            FrameSamplingStrategy.FIXED_FPS,
            FrameSamplingStrategy.DENSE_WINDOW,
        }
        if self.strategy in fps_strategies and (self.fps is None or self.fps <= 0):
            raise ValueError("fps must be positive for FPS-based sampling")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be positive when provided")
        if self.strategy is FrameSamplingStrategy.SHOT_KEYFRAME and self.shots is None:
            raise ValueError("shots are required for shot-keyframe sampling")
        if self.shots is not None and self.shots.video_id != self.video_asset.video_id:
            raise ValueError("shots must belong to the requested video")
