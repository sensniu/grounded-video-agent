from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import TypeAdapter, ValidationError

from grounded_video_agent.capabilities._support import write_json
from grounded_video_agent.domain import (
    ChunkManifest,
    EmbeddingManifest,
    FrameManifest,
    IndexManifest,
    ManifestKind,
    OCRManifest,
    ShotManifest,
    TranscriptManifest,
    VisualDescriptionManifest,
)
from grounded_video_agent.workspace.catalog.contracts import CatalogDocumentKind
from grounded_video_agent.workspace.catalog.documents import (
    AudioAssetDocument,
    MediaInspectionDocument,
    VideoClipDocument,
)

T = TypeVar("T")

_MANIFEST_TYPES: dict[type[Any], ManifestKind] = {
    ShotManifest: ManifestKind.SHOTS,
    ChunkManifest: ManifestKind.CHUNKS,
    FrameManifest: ManifestKind.FRAMES,
    TranscriptManifest: ManifestKind.TRANSCRIPT,
    OCRManifest: ManifestKind.OCR,
    VisualDescriptionManifest: ManifestKind.VISUAL_DESCRIPTIONS,
    EmbeddingManifest: ManifestKind.EMBEDDINGS,
    IndexManifest: ManifestKind.INDEX,
}
_TYPES_BY_KIND = {kind: manifest_type for manifest_type, kind in _MANIFEST_TYPES.items()}

_DOCUMENT_TYPES: dict[type[Any], CatalogDocumentKind] = {
    MediaInspectionDocument: CatalogDocumentKind.MEDIA_INSPECTION,
    AudioAssetDocument: CatalogDocumentKind.AUDIO_ASSET,
    VideoClipDocument: CatalogDocumentKind.VIDEO_CLIP,
}
_DOCUMENT_TYPES_BY_KIND = {
    kind: document_type for document_type, kind in _DOCUMENT_TYPES.items()
}


class ManifestCodecRegistry:
    def expected_kind(self, expected_type: type[Any]) -> ManifestKind:
        try:
            return _MANIFEST_TYPES[expected_type]
        except KeyError as error:
            raise TypeError(f"unsupported manifest type: {expected_type.__name__}") from error

    def load(self, path: str | Path, expected_type: type[T]) -> T:
        self.expected_kind(expected_type)
        try:
            payload = Path(path).read_bytes()
            loaded = TypeAdapter(expected_type).validate_json(payload)
        except (OSError, ValidationError, ValueError, TypeError) as error:
            raise ValueError(f"invalid {expected_type.__name__}: {error}") from error
        return cast(T, loaded)

    def type_for_kind(self, kind: ManifestKind) -> type[Any]:
        try:
            return _TYPES_BY_KIND[kind]
        except KeyError as error:
            raise TypeError(f"unsupported manifest kind: {kind.value}") from error


class DocumentCodecRegistry:
    def expected_kind(self, expected_type: type[Any]) -> CatalogDocumentKind:
        try:
            return _DOCUMENT_TYPES[expected_type]
        except KeyError as error:
            raise TypeError(f"unsupported document type: {expected_type.__name__}") from error

    def load(self, path: str | Path, expected_type: type[T]) -> T:
        self.expected_kind(expected_type)
        try:
            payload = Path(path).read_bytes()
            loaded = TypeAdapter(expected_type).validate_json(payload)
        except (OSError, ValidationError, ValueError, TypeError) as error:
            raise ValueError(f"invalid {expected_type.__name__}: {error}") from error
        return cast(T, loaded)

    def dump(self, path: str | Path, document: Any) -> None:
        self.expected_kind(type(document))
        write_json(Path(path), document)

    def type_for_kind(self, kind: CatalogDocumentKind) -> type[Any]:
        try:
            return _DOCUMENT_TYPES_BY_KIND[kind]
        except KeyError as error:
            raise TypeError(f"unsupported document kind: {kind.value}") from error
