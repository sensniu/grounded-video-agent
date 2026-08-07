from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain import CapabilityRequestContext, TimeRange, VideoAsset


@dataclass(frozen=True, slots=True)
class AudioExtractionRequest:
    video_asset: VideoAsset
    source_range: TimeRange
    stream_index: int
    context: CapabilityRequestContext
    sample_rate_hz: int = 16_000
    channels: int = 1

    def __post_init__(self) -> None:
        if self.stream_index < 0:
            raise ValueError("stream_index must be non-negative")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
