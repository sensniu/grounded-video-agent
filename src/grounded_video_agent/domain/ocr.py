"""Frame-level OCR observations and temporally merged text spans."""

from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain._invariants import (
    require_finite_number,
    require_manifest,
    require_non_negative_int,
    require_probability,
    require_text,
    require_unique_texts,
)
from grounded_video_agent.domain.artifacts import ManifestKind, ManifestRef
from grounded_video_agent.domain.timeline import TimeRange


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Normalized ``x, y, width, height`` coordinates in the range 0..1."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for field_name in ("x", "y", "width", "height"):
            require_finite_number(getattr(self, field_name), field_name)
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("bounding box coordinates and dimensions must be positive")
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("bounding box must be contained by the normalized frame")


@dataclass(frozen=True, slots=True)
class OCRObservation:
    observation_id: str
    video_id: str
    frame_id: str
    timestamp_ms: int
    raw_text: str
    normalized_text: str
    bbox: BoundingBox
    confidence: float
    language: str | None = None

    def __post_init__(self) -> None:
        require_text(self.observation_id, "observation_id")
        require_text(self.video_id, "video_id")
        require_text(self.frame_id, "frame_id")
        require_non_negative_int(self.timestamp_ms, "timestamp_ms")
        require_text(self.raw_text, "raw_text")
        require_text(self.normalized_text, "normalized_text")
        require_probability(self.confidence)
        if self.language is not None:
            require_text(self.language, "language")


@dataclass(frozen=True, slots=True)
class OCRSpan:
    span_id: str
    video_id: str
    time_range: TimeRange
    text: str
    observation_ids: tuple[str, ...]
    confidence: float

    def __post_init__(self) -> None:
        require_text(self.span_id, "span_id")
        require_text(self.video_id, "video_id")
        require_text(self.text, "text")
        require_unique_texts(self.observation_ids, "observation_ids")
        if not self.observation_ids:
            raise ValueError("OCR span requires at least one observation")
        require_probability(self.confidence)


@dataclass(frozen=True, slots=True)
class OCRManifest:
    ref: ManifestRef
    video_id: str
    observations: tuple[OCRObservation, ...]
    spans: tuple[OCRSpan, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.video_id, "video_id")
        require_manifest(
            self.ref,
            kind=ManifestKind.OCR,
            video_id=self.video_id,
            item_count=len(self.observations),
        )
        require_unique_texts(
            (observation.observation_id for observation in self.observations),
            "observation_ids",
        )
        require_unique_texts((span.span_id for span in self.spans), "span_ids")
        if any(item.video_id != self.video_id for item in self.observations):
            raise ValueError("all OCR records must belong to the manifest video")
        if any(item.video_id != self.video_id for item in self.spans):
            raise ValueError("all OCR records must belong to the manifest video")
        observation_ids = {observation.observation_id for observation in self.observations}
        if any(
            observation_id not in observation_ids
            for span in self.spans
            for observation_id in span.observation_ids
        ):
            raise ValueError("OCR spans must reference observations in the same manifest")
