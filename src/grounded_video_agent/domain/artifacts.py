"""References and provenance for files derived from source media."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _require_text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


class ArtifactKind(StrEnum):
    """Broad media and data artifact categories."""

    SOURCE_VIDEO = "source_video"
    NORMALIZED_VIDEO = "normalized_video"
    VIDEO_CLIP = "video_clip"
    AUDIO = "audio"
    FRAME_IMAGE = "frame_image"
    TRANSCRIPT = "transcript"
    OCR = "ocr"
    EMBEDDING = "embedding"
    METADATA = "metadata"
    MANIFEST = "manifest"
    INDEX = "index"
    OTHER = "other"


class ManifestKind(StrEnum):
    """Known collections of derived domain objects."""

    SHOTS = "shots"
    CHUNKS = "chunks"
    FRAMES = "frames"
    TRANSCRIPT = "transcript"
    OCR = "ocr"
    VISUAL_DESCRIPTIONS = "visual_descriptions"
    EMBEDDINGS = "embeddings"
    INDEX = "index"
    EVIDENCE = "evidence"


@dataclass(frozen=True, slots=True)
class ProducerInfo:
    """Identity of the component that created a derived result."""

    name: str
    version: str

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.version, "version")


@dataclass(frozen=True, slots=True)
class Provenance:
    """Reproducibility metadata for a derived artifact or manifest."""

    producer: ProducerInfo
    parameters_hash: str
    source_video_id: str | None = None
    source_artifact_ids: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        _require_text(self.parameters_hash, "parameters_hash")
        if self.source_video_id is not None:
            _require_text(self.source_video_id, "source_video_id")
        if any(not artifact_id.strip() for artifact_id in self.source_artifact_ids):
            raise ValueError("source_artifact_ids must not contain empty values")
        if len(set(self.source_artifact_ids)) != len(self.source_artifact_ids):
            raise ValueError("source_artifact_ids must be unique")
        _require_aware_datetime(self.created_at, "created_at")


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A lightweight reference to content stored outside agent state."""

    artifact_id: str
    kind: ArtifactKind
    uri: str
    sha256: str | None = None
    size_bytes: int | None = None
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        _require_text(self.artifact_id, "artifact_id")
        _require_text(self.uri, "uri")
        if self.sha256 is not None and not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class ManifestRef:
    """Reference to a versioned collection of derived domain records."""

    manifest_id: str
    kind: ManifestKind
    artifact: ArtifactRef
    source_video_id: str
    item_count: int
    schema_version: str = "1"

    def __post_init__(self) -> None:
        _require_text(self.manifest_id, "manifest_id")
        _require_text(self.source_video_id, "source_video_id")
        _require_text(self.schema_version, "schema_version")
        if self.artifact.kind is not ArtifactKind.MANIFEST:
            raise ValueError("a manifest must reference an artifact of kind MANIFEST")
        if self.item_count < 0:
            raise ValueError("item_count must be non-negative")
