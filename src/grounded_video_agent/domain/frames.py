"""Frame sampling strategies and reproducible frame manifests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from grounded_video_agent.domain._invariants import (
    require_manifest,
    require_non_negative_int,
    require_text,
    require_unique_texts,
)
from grounded_video_agent.domain.artifacts import ManifestKind, ManifestRef
from grounded_video_agent.domain.timeline import FrameRef, TimeRange


class FrameSamplingStrategy(StrEnum):
    UNIFORM = "uniform"
    FIXED_FPS = "fixed_fps"
    SHOT_KEYFRAME = "shot_keyframe"
    DENSE_WINDOW = "dense_window"


@dataclass(frozen=True, slots=True)
class FrameManifest:
    ref: ManifestRef
    video_id: str
    strategy: FrameSamplingStrategy
    requested_ranges: tuple[TimeRange, ...]
    frames: tuple[FrameRef, ...]
    decoded_frames: int
    dropped_duplicates: int = 0

    def __post_init__(self) -> None:
        require_text(self.video_id, "video_id")
        require_manifest(
            self.ref,
            kind=ManifestKind.FRAMES,
            video_id=self.video_id,
            item_count=len(self.frames),
        )
        require_non_negative_int(self.decoded_frames, "decoded_frames")
        require_non_negative_int(self.dropped_duplicates, "dropped_duplicates")
        if not self.requested_ranges:
            raise ValueError("frame manifest requires at least one requested range")
        if len(self.frames) > self.decoded_frames:
            raise ValueError("frame count cannot exceed decoded_frames")
        require_unique_texts((frame.frame_id for frame in self.frames), "frame_ids")
        if any(frame.video_id != self.video_id for frame in self.frames):
            raise ValueError("all frames must belong to the manifest video")
        if tuple(sorted(self.frames, key=lambda frame: frame.timestamp_ms)) != self.frames:
            raise ValueError("frames must be ordered by actual timestamp")
