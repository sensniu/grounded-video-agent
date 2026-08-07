"""Shot boundaries and retrieval-oriented logical chunks."""

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


class ChunkBasis(StrEnum):
    """The information source that determined a logical chunk boundary."""

    TEMPORAL = "temporal"
    TRANSCRIPT = "transcript"
    SHOT_FALLBACK = "shot_fallback"


@dataclass(frozen=True, slots=True)
class Shot:
    shot_id: str
    video_id: str
    time_range: TimeRange
    confidence: float | None = None

    def __post_init__(self) -> None:
        require_text(self.shot_id, "shot_id")
        require_text(self.video_id, "video_id")
        require_probability(self.confidence)


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    video_id: str
    time_range: TimeRange
    shot_ids: tuple[str, ...] = ()
    transcript_segment_ids: tuple[str, ...] = ()
    inspection_range: TimeRange | None = None
    text: str | None = None
    basis: ChunkBasis = ChunkBasis.TEMPORAL

    def __post_init__(self) -> None:
        require_text(self.chunk_id, "chunk_id")
        require_text(self.video_id, "video_id")
        require_unique_texts(self.shot_ids, "shot_ids")
        require_unique_texts(self.transcript_segment_ids, "transcript_segment_ids")
        if self.inspection_range is not None and not self.inspection_range.contains_range(
            self.time_range
        ):
            raise ValueError("inspection_range must contain the exact chunk time_range")
        if self.text is not None:
            require_text(self.text, "text")
        if self.basis is ChunkBasis.TRANSCRIPT:
            if not self.transcript_segment_ids or self.text is None:
                raise ValueError("transcript chunks require segment ids and text")
        if self.basis is ChunkBasis.SHOT_FALLBACK:
            if not self.shot_ids:
                raise ValueError("shot fallback chunks require shot ids")
            if self.transcript_segment_ids or self.text is not None:
                raise ValueError("shot fallback chunks cannot contain transcript data")

    @property
    def observation_range(self) -> TimeRange:
        """Range that downstream visual inspection should observe."""

        return self.inspection_range or self.time_range


@dataclass(frozen=True, slots=True)
class ShotManifest:
    ref: ManifestRef
    video_id: str
    shots: tuple[Shot, ...]

    def __post_init__(self) -> None:
        require_text(self.video_id, "video_id")
        require_manifest(
            self.ref,
            kind=ManifestKind.SHOTS,
            video_id=self.video_id,
            item_count=len(self.shots),
        )
        require_unique_texts((shot.shot_id for shot in self.shots), "shot_ids")
        if any(shot.video_id != self.video_id for shot in self.shots):
            raise ValueError("all shots must belong to the manifest video")
        if tuple(sorted(self.shots, key=lambda shot: shot.time_range)) != self.shots:
            raise ValueError("shots must be ordered by time range")
        adjacent_shots = zip(self.shots, self.shots[1:], strict=False)
        if any(first.time_range.overlaps(second.time_range) for first, second in adjacent_shots):
            raise ValueError("shots must not overlap")


@dataclass(frozen=True, slots=True)
class ChunkManifest:
    ref: ManifestRef
    video_id: str
    chunks: tuple[Chunk, ...]

    def __post_init__(self) -> None:
        require_text(self.video_id, "video_id")
        require_manifest(
            self.ref,
            kind=ManifestKind.CHUNKS,
            video_id=self.video_id,
            item_count=len(self.chunks),
        )
        require_unique_texts((chunk.chunk_id for chunk in self.chunks), "chunk_ids")
        if any(chunk.video_id != self.video_id for chunk in self.chunks):
            raise ValueError("all chunks must belong to the manifest video")
        if tuple(sorted(self.chunks, key=lambda chunk: chunk.time_range)) != self.chunks:
            raise ValueError("chunks must be ordered by time range")
