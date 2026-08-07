from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain import (
    CapabilityRequestContext,
    FrameManifest,
    VisualAnalysisTarget,
    VisualDescriptionMode,
)


@dataclass(frozen=True, slots=True)
class VisualContentAnalysisRequest:
    frames: FrameManifest
    targets: tuple[VisualAnalysisTarget, ...]
    mode: VisualDescriptionMode
    context: CapabilityRequestContext
    question: str | None = None

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("targets must not be empty")
        target_ids = tuple(item.target_id for item in self.targets)
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("target ids must be unique")
        known_frames = {frame.frame_id: frame for frame in self.frames.frames}
        for target in self.targets:
            if target.video_id != self.frames.video_id:
                raise ValueError("targets and frames must belong to the same video")
            if not set(target.frame_ids).issubset(known_frames):
                raise ValueError("targets must reference frames in the frame manifest")
            if any(
                not target.time_range.contains_timestamp(known_frames[frame_id].timestamp_ms)
                for frame_id in target.frame_ids
            ):
                raise ValueError("target frames must be contained by the target time range")
        if self.mode is VisualDescriptionMode.QUESTION_CONDITIONED:
            if self.question is None or not self.question.strip():
                raise ValueError("question-conditioned analysis requires a question")
        elif self.question is not None:
            raise ValueError("generic analysis must not contain a question")
