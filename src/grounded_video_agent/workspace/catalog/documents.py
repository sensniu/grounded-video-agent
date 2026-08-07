"""Typed singleton documents stored and versioned by the artifact catalog."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from grounded_video_agent.domain import (
    AudioArtifact,
    MediaProbe,
    ValidationReport,
    ValidationStatus,
    VideoAsset,
    VideoClipArtifact,
)
from grounded_video_agent.workspace.catalog.contracts import (
    CatalogDocumentKind,
    CatalogDocumentRef,
)


class MediaInspectionNextAction(StrEnum):
    """Framework action recommended after a successful media inspection."""

    PROCEED = "proceed"
    PROCEED_WITH_LIMITATIONS = "proceed_with_limitations"
    NORMALIZE_AND_REINSPECT = "normalize_and_reinspect"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class PrimaryStreamSelection:
    video_stream_index: int | None
    audio_stream_index: int | None
    subtitle_stream_index: int | None

    def __post_init__(self) -> None:
        values = (
            self.video_stream_index,
            self.audio_stream_index,
            self.subtitle_stream_index,
        )
        if any(value is not None and value < 0 for value in values):
            raise ValueError("primary stream indexes must be non-negative")

    @classmethod
    def from_probe(cls, probe: MediaProbe) -> PrimaryStreamSelection:
        video = probe.primary_video_stream
        audio = probe.primary_audio_stream
        subtitle = probe.primary_subtitle_stream
        return cls(
            video.stream_index if video is not None else None,
            audio.stream_index if audio is not None else None,
            subtitle.stream_index if subtitle is not None else None,
        )


@dataclass(frozen=True, slots=True)
class BasicMediaFlags:
    has_video: bool
    has_audio: bool
    has_embedded_subtitles: bool
    has_multiple_video_streams: bool
    has_multiple_audio_streams: bool
    is_variable_frame_rate: bool
    has_rotation_metadata: bool

    @classmethod
    def from_probe(cls, probe: MediaProbe) -> BasicMediaFlags:
        return cls(
            has_video=probe.primary_video_stream is not None,
            has_audio=bool(probe.audio_streams),
            has_embedded_subtitles=bool(probe.subtitle_streams),
            has_multiple_video_streams=len(probe.video_streams) > 1,
            has_multiple_audio_streams=len(probe.audio_streams) > 1,
            is_variable_frame_rate=any(
                stream.is_variable_frame_rate for stream in probe.video_streams
            ),
            has_rotation_metadata=any(
                stream.rotation_degrees % 360 != 0 for stream in probe.video_streams
            ),
        )


@dataclass(frozen=True, slots=True)
class MediaInspectionDocument:
    """Reusable successful inspection facts, independent of capability execution state."""

    ref: CatalogDocumentRef
    inspection_id: str
    video_asset: VideoAsset
    media_probe: MediaProbe
    validation: ValidationReport
    primary_streams: PrimaryStreamSelection
    basic_flags: BasicMediaFlags
    next_action: MediaInspectionNextAction

    def __post_init__(self) -> None:
        if self.ref.kind is not CatalogDocumentKind.MEDIA_INSPECTION:
            raise ValueError("media inspection document ref has the wrong kind")
        if not self.inspection_id.strip():
            raise ValueError("inspection_id must not be empty")
        video_id = self.video_asset.video_id
        if self.ref.source_video_id != video_id:
            raise ValueError("document ref must belong to the inspected video")
        if self.media_probe.video_id != video_id or self.validation.video_id != video_id:
            raise ValueError("inspection facts must belong to the inspected video")
        if self.primary_streams != PrimaryStreamSelection.from_probe(self.media_probe):
            raise ValueError("primary stream selection does not match media probe")
        if self.basic_flags != BasicMediaFlags.from_probe(self.media_probe):
            raise ValueError("basic media flags do not match media probe")
        expected_action = {
            ValidationStatus.VALID: MediaInspectionNextAction.PROCEED,
            ValidationStatus.VALID_WITH_WARNINGS: MediaInspectionNextAction.PROCEED,
            ValidationStatus.REQUIRES_NORMALIZATION: (
                MediaInspectionNextAction.NORMALIZE_AND_REINSPECT
            ),
            ValidationStatus.PARTIALLY_SUPPORTED: (
                MediaInspectionNextAction.PROCEED_WITH_LIMITATIONS
            ),
            ValidationStatus.INVALID: MediaInspectionNextAction.REJECT,
        }[self.validation.status]
        if self.next_action is not expected_action:
            raise ValueError("next_action does not match validation status")

    @property
    def video_id(self) -> str:
        return self.video_asset.video_id


@dataclass(frozen=True, slots=True)
class AudioAssetDocument:
    """Typed metadata describing one derived audio file and its source timeline."""

    ref: CatalogDocumentRef
    audio_asset: AudioArtifact

    def __post_init__(self) -> None:
        if self.ref.kind is not CatalogDocumentKind.AUDIO_ASSET:
            raise ValueError("audio asset document ref has the wrong kind")
        if self.ref.source_video_id != self.audio_asset.video_id:
            raise ValueError("audio asset document must belong to its source video")

    @property
    def video_id(self) -> str:
        return self.audio_asset.video_id


@dataclass(frozen=True, slots=True)
class VideoClipDocument:
    """Typed metadata for one source-aligned exported video clip."""

    ref: CatalogDocumentRef
    video_clip: VideoClipArtifact
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.ref.kind is not CatalogDocumentKind.VIDEO_CLIP:
            raise ValueError("video clip document ref has the wrong kind")
        if self.ref.source_video_id != self.video_clip.video_id:
            raise ValueError("video clip document must belong to its source video")
        if any(not item.strip() for item in self.evidence_ids):
            raise ValueError("video clip evidence ids must not be empty")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("video clip evidence ids must be unique")

    @property
    def video_id(self) -> str:
        return self.video_clip.video_id
