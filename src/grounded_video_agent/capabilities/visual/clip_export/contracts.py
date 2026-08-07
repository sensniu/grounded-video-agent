from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain import CapabilityRequestContext, TimeRange, VideoAsset


@dataclass(frozen=True, slots=True)
class ClipExportRequest:
    video_asset: VideoAsset
    time_range: TimeRange
    context: CapabilityRequestContext
    include_audio: bool = True
    reencode: bool = True
