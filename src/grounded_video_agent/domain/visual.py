"""Generic and question-conditioned visual descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from grounded_video_agent.domain._invariants import (
    require_manifest,
    require_probability,
    require_text,
    require_unique_texts,
)
from grounded_video_agent.domain.artifacts import ManifestKind, ManifestRef
from grounded_video_agent.domain.timeline import TimeRange


class VisualDescriptionMode(StrEnum):
    GENERIC = "generic"
    QUESTION_CONDITIONED = "question_conditioned"


@dataclass(frozen=True, slots=True)
class VisualAnalysisTarget:
    target_id: str
    video_id: str
    time_range: TimeRange
    frame_ids: tuple[str, ...]
    chunk_id: str | None = None
    shot_id: str | None = None

    def __post_init__(self) -> None:
        require_text(self.target_id, "target_id")
        require_text(self.video_id, "video_id")
        require_unique_texts(self.frame_ids, "frame_ids")
        if not self.frame_ids:
            raise ValueError("visual analysis target requires at least one frame")
        if self.chunk_id is not None:
            require_text(self.chunk_id, "chunk_id")
        if self.shot_id is not None:
            require_text(self.shot_id, "shot_id")


@dataclass(frozen=True, slots=True)
class VisualDescription:
    description_id: str
    video_id: str
    time_range: TimeRange
    text: str
    mode: VisualDescriptionMode
    frame_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    confidence: float | None = None
    question: str | None = None

    def __post_init__(self) -> None:
        require_text(self.description_id, "description_id")
        require_text(self.video_id, "video_id")
        require_text(self.text, "text")
        require_unique_texts(self.frame_ids, "frame_ids")
        require_unique_texts(self.tags, "tags")
        require_probability(self.confidence)
        if self.mode is VisualDescriptionMode.QUESTION_CONDITIONED:
            if self.question is None:
                raise ValueError("question-conditioned description requires a question")
            require_text(self.question, "question")
        elif self.question is not None:
            raise ValueError("generic description must not contain a question")


@dataclass(frozen=True, slots=True)
class VisualDescriptionManifest:
    ref: ManifestRef
    video_id: str
    descriptions: tuple[VisualDescription, ...]

    def __post_init__(self) -> None:
        require_text(self.video_id, "video_id")
        require_manifest(
            self.ref,
            kind=ManifestKind.VISUAL_DESCRIPTIONS,
            video_id=self.video_id,
            item_count=len(self.descriptions),
        )
        require_unique_texts(
            (description.description_id for description in self.descriptions),
            "description_ids",
        )
        if any(description.video_id != self.video_id for description in self.descriptions):
            raise ValueError("all descriptions must belong to the manifest video")
