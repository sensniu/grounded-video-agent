from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain import CapabilityRequestContext, VideoAsset


@dataclass(frozen=True, slots=True)
class EmbeddedSubtitleExtractionRequest:
    video_asset: VideoAsset
    stream_index: int
    context: CapabilityRequestContext
    language: str | None = None

    def __post_init__(self) -> None:
        if self.stream_index < 0:
            raise ValueError("stream_index must be non-negative")
        if self.language is not None and not self.language.strip():
            raise ValueError("language must not be empty")
