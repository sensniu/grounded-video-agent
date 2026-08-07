from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    IndexKind,
    IndexModality,
    ManifestKind,
    ManifestRef,
    VideoAsset,
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_VARIANT = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class CatalogResourceType(StrEnum):
    SOURCE = "source"
    ARTIFACT = "artifact"
    DOCUMENT = "document"
    MANIFEST = "manifest"
    INDEX = "index"


class CatalogDocumentKind(StrEnum):
    MEDIA_INSPECTION = "media_inspection"
    AUDIO_ASSET = "audio_asset"


@dataclass(frozen=True, slots=True)
class CatalogDocumentRef:
    document_id: str
    kind: CatalogDocumentKind
    artifact: ArtifactRef
    source_video_id: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in ("document_id", "source_video_id", "schema_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.artifact.kind is not ArtifactKind.METADATA:
            raise ValueError("a catalog document must reference a METADATA artifact")


@dataclass(frozen=True, slots=True)
class CatalogKey:
    resource_type: CatalogResourceType
    variant: str = "primary"
    artifact_kind: ArtifactKind | None = None
    manifest_kind: ManifestKind | None = None
    modality: IndexModality | None = None
    index_kind: IndexKind | None = None
    document_kind: CatalogDocumentKind | None = None

    def __post_init__(self) -> None:
        if not _VARIANT.fullmatch(self.variant):
            raise ValueError("catalog variant must be a normalized identifier")
        if self.resource_type is CatalogResourceType.SOURCE:
            if self.artifact_kind is not ArtifactKind.SOURCE_VIDEO:
                raise ValueError("source catalog keys require SOURCE_VIDEO")
            self._require_empty(
                "source", self.document_kind, self.manifest_kind, self.modality, self.index_kind
            )
        elif self.resource_type is CatalogResourceType.ARTIFACT:
            if self.artifact_kind is None:
                raise ValueError("artifact catalog keys require artifact_kind")
            self._require_empty(
                "artifact", self.document_kind, self.manifest_kind, self.modality, self.index_kind
            )
        elif self.resource_type is CatalogResourceType.DOCUMENT:
            if self.document_kind is None:
                raise ValueError("document catalog keys require document_kind")
            self._require_empty(
                "document", self.artifact_kind, self.manifest_kind, self.modality, self.index_kind
            )
        elif self.resource_type is CatalogResourceType.MANIFEST:
            if self.manifest_kind is None or self.manifest_kind is ManifestKind.INDEX:
                raise ValueError("manifest catalog keys require a non-index manifest_kind")
            self._require_empty(
                "manifest", self.artifact_kind, self.document_kind, self.modality, self.index_kind
            )
        else:
            if self.modality is None or self.index_kind is None:
                raise ValueError("index catalog keys require modality and index_kind")
            self._require_empty(
                "index", self.artifact_kind, self.document_kind, self.manifest_kind
            )

    @staticmethod
    def _require_empty(label: str, *values: object) -> None:
        if any(value is not None for value in values):
            raise ValueError(f"{label} catalog key contains incompatible fields")

    @property
    def canonical_name(self) -> str:
        components = [self.resource_type.value]
        if self.artifact_kind is not None:
            components.append(self.artifact_kind.value)
        if self.document_kind is not None:
            components.append(self.document_kind.value)
        if self.manifest_kind is not None:
            components.append(self.manifest_kind.value)
        if self.modality is not None:
            components.append(self.modality.value)
        if self.index_kind is not None:
            components.append(self.index_kind.value)
        components.append(self.variant)
        return ":".join(components)


class CatalogEntryState(StrEnum):
    AVAILABLE = "available"
    STALE = "stale"
    MISSING = "missing"
    CORRUPT = "corrupt"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    entry_id: str
    video_id: str
    key: CatalogKey
    reference: ArtifactRef | CatalogDocumentRef | ManifestRef
    operation_id: str
    producer_name: str
    producer_version: str
    dependency_entry_ids: tuple[str, ...]
    content_sha256: str
    size_bytes: int
    parameters_hash: str | None = None
    derivation_key: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        for name in (
            "entry_id",
            "video_id",
            "operation_id",
            "producer_name",
            "producer_version",
        ):
            value = getattr(self, name)
            if not value or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if len(set(self.dependency_entry_ids)) != len(self.dependency_entry_ids):
            raise ValueError("catalog dependencies must be unique")
        if any(not value.strip() for value in self.dependency_entry_ids):
            raise ValueError("catalog dependencies must not be empty")
        if not _DIGEST.fullmatch(self.content_sha256):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        for name in ("parameters_hash", "derivation_key"):
            value = getattr(self, name)
            if value is not None and not _DIGEST.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.size_bytes < 0:
            raise ValueError("catalog entry size_bytes must be non-negative")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("catalog entry created_at must be timezone-aware")
        self._validate_reference()

    def _validate_reference(self) -> None:
        if self.key.resource_type in {
            CatalogResourceType.SOURCE,
            CatalogResourceType.ARTIFACT,
        }:
            if not isinstance(self.reference, ArtifactRef):
                raise ValueError("artifact catalog keys require ArtifactRef")
            if self.reference.kind is not self.key.artifact_kind:
                raise ValueError("artifact reference kind does not match catalog key")
            return
        if self.key.resource_type is CatalogResourceType.DOCUMENT:
            if not isinstance(self.reference, CatalogDocumentRef):
                raise ValueError("document catalog keys require CatalogDocumentRef")
            if self.reference.kind is not self.key.document_kind:
                raise ValueError("document reference kind does not match catalog key")
            if self.reference.source_video_id != self.video_id:
                raise ValueError("document reference must belong to catalog video")
            return
        if not isinstance(self.reference, ManifestRef):
            raise ValueError("manifest and index catalog keys require ManifestRef")
        expected_kind = (
            ManifestKind.INDEX
            if self.key.resource_type is CatalogResourceType.INDEX
            else self.key.manifest_kind
        )
        if self.reference.kind is not expected_kind:
            raise ValueError("manifest reference kind does not match catalog key")
        if self.reference.source_video_id != self.video_id:
            raise ValueError("manifest reference must belong to catalog video")

    @property
    def artifact(self) -> ArtifactRef:
        if isinstance(self.reference, CatalogDocumentRef | ManifestRef):
            return self.reference.artifact
        return self.reference


@dataclass(frozen=True, slots=True)
class CatalogSelection:
    key: CatalogKey
    entry_id: str

    def __post_init__(self) -> None:
        if not self.entry_id.strip():
            raise ValueError("catalog selection entry_id must not be empty")


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    schema_version: str
    video_asset: VideoAsset
    revision: int
    entries: tuple[CatalogEntry, ...]
    active_selections: tuple[CatalogSelection, ...]
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("catalog schema_version must not be empty")
        if self.revision <= 0:
            raise ValueError("catalog revision must be positive")
        for value in (self.created_at, self.updated_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("catalog timestamps must be timezone-aware")
        if self.updated_at < self.created_at:
            raise ValueError("catalog updated_at must not precede created_at")
        entry_ids = tuple(entry.entry_id for entry in self.entries)
        if len(set(entry_ids)) != len(entry_ids):
            raise ValueError("catalog entry ids must be unique")
        if any(entry.video_id != self.video_asset.video_id for entry in self.entries):
            raise ValueError("catalog entries must belong to snapshot video")
        selected_names = tuple(item.key.canonical_name for item in self.active_selections)
        if len(set(selected_names)) != len(selected_names):
            raise ValueError("catalog active keys must be unique")
        entries = {entry.entry_id: entry for entry in self.entries}
        for selection in self.active_selections:
            entry = entries.get(selection.entry_id)
            if entry is None or entry.key != selection.key:
                raise ValueError("catalog selections must reference entries with matching keys")
        if any(
            dependency not in entries
            for entry in self.entries
            for dependency in entry.dependency_entry_ids
        ):
            raise ValueError("catalog dependencies must reference entries in the snapshot")

    @property
    def video_id(self) -> str:
        return self.video_asset.video_id


@dataclass(frozen=True, slots=True)
class CatalogRegistration:
    key: CatalogKey
    reference: ArtifactRef | CatalogDocumentRef | ManifestRef
    operation_id: str
    dependency_entry_ids: tuple[str, ...] = ()
    producer_name: str | None = None
    producer_version: str | None = None
    parameters_hash: str | None = None
    derivation_key: str | None = None
    activate: bool = True

    def __post_init__(self) -> None:
        if not self.operation_id.strip():
            raise ValueError("operation_id must not be empty")
        if (self.producer_name is None) != (self.producer_version is None):
            raise ValueError("producer_name and producer_version must be provided together")
        for name in ("producer_name", "producer_version"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} must not be empty")
        if len(set(self.dependency_entry_ids)) != len(self.dependency_entry_ids):
            raise ValueError("catalog registration dependencies must be unique")
        for name in ("parameters_hash", "derivation_key"):
            value = getattr(self, name)
            if value is not None and not _DIGEST.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ResolvedCatalogEntry:
    entry: CatalogEntry
    state: CatalogEntryState
    stale_dependency_entry_ids: tuple[str, ...] = ()


class CatalogAuditIssueCode(StrEnum):
    MISSING_FILE = "missing_file"
    PATH_NOT_ALLOWED = "path_not_allowed"
    HASH_MISMATCH = "hash_mismatch"
    INVALID_MANIFEST = "invalid_manifest"
    INVALID_DOCUMENT = "invalid_document"
    STALE_DEPENDENCY = "stale_dependency"


@dataclass(frozen=True, slots=True)
class CatalogAuditIssue:
    entry_id: str
    code: CatalogAuditIssueCode
    message: str


@dataclass(frozen=True, slots=True)
class CatalogEntryAudit:
    entry_id: str
    state: CatalogEntryState


@dataclass(frozen=True, slots=True)
class CatalogAuditReport:
    video_id: str
    revision: int
    deep: bool
    entries: tuple[CatalogEntryAudit, ...]
    issues: tuple[CatalogAuditIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not any(
            item.state in {CatalogEntryState.MISSING, CatalogEntryState.CORRUPT}
            for item in self.entries
        )


class CatalogErrorCode(StrEnum):
    VIDEO_NOT_FOUND = "video_not_found"
    VIDEO_ALREADY_REGISTERED = "video_already_registered"
    REVISION_CONFLICT = "revision_conflict"
    ENTRY_NOT_FOUND = "entry_not_found"
    RESOURCE_NOT_REGISTERED = "resource_not_registered"
    PATH_NOT_ALLOWED = "path_not_allowed"
    RESOURCE_MISSING = "resource_missing"
    HASH_MISMATCH = "hash_mismatch"
    TYPE_MISMATCH = "type_mismatch"
    CORRUPT_CATALOG = "corrupt_catalog"
    CORRUPT_RESOURCE = "corrupt_resource"
    DEPENDENCY_NOT_FOUND = "dependency_not_found"
    STALE_RESOURCE = "stale_resource"
    INVALID_REFERENCE = "invalid_reference"
    UNSUPPORTED_SCHEMA = "unsupported_schema"


class CatalogError(RuntimeError):
    def __init__(self, code: CatalogErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
