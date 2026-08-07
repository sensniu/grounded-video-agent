"""Embedding and retrieval-index metadata; vector payloads stay in artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from grounded_video_agent.domain._invariants import (
    require_manifest,
    require_positive_int,
    require_text,
    require_unique_texts,
)
from grounded_video_agent.domain.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ManifestKind,
    ManifestRef,
)


class IndexModality(StrEnum):
    TRANSCRIPT = "transcript"
    OCR = "ocr"
    VISUAL_DESCRIPTION = "visual_description"
    VISUAL_EMBEDDING = "visual_embedding"


class IndexKind(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    VECTOR = "vector"


@dataclass(frozen=True, slots=True)
class EmbeddingManifest:
    ref: ManifestRef
    video_id: str
    modality: IndexModality
    embedding_space: str
    dimensions: int
    item_ids: tuple[str, ...]
    embedding_artifact: ArtifactRef

    def __post_init__(self) -> None:
        require_text(self.video_id, "video_id")
        require_text(self.embedding_space, "embedding_space")
        require_positive_int(self.dimensions, "dimensions")
        require_unique_texts(self.item_ids, "item_ids")
        require_manifest(
            self.ref,
            kind=ManifestKind.EMBEDDINGS,
            video_id=self.video_id,
            item_count=len(self.item_ids),
        )
        if self.embedding_artifact.kind is not ArtifactKind.EMBEDDING:
            raise ValueError("embedding_artifact must have kind EMBEDDING")


@dataclass(frozen=True, slots=True)
class IndexManifest:
    ref: ManifestRef
    video_id: str
    modality: IndexModality
    index_kind: IndexKind
    source_manifest_ids: tuple[str, ...]
    index_artifact: ArtifactRef
    embedding_manifest_id: str | None = None
    embedding_manifest: EmbeddingManifest | None = None

    def __post_init__(self) -> None:
        require_text(self.video_id, "video_id")
        require_unique_texts(self.source_manifest_ids, "source_manifest_ids")
        if not self.source_manifest_ids:
            raise ValueError("index requires at least one source manifest")
        require_manifest(
            self.ref,
            kind=ManifestKind.INDEX,
            video_id=self.video_id,
            item_count=self.ref.item_count,
        )
        if self.index_artifact.kind is not ArtifactKind.INDEX:
            raise ValueError("index_artifact must have kind INDEX")
        if self.embedding_manifest_id is not None:
            require_text(self.embedding_manifest_id, "embedding_manifest_id")
        if self.embedding_manifest is not None:
            if self.embedding_manifest.video_id != self.video_id:
                raise ValueError("embedding manifest must belong to the index video")
            if self.embedding_manifest.modality is not self.modality:
                raise ValueError("embedding manifest modality must match the index")
            if self.embedding_manifest.ref.manifest_id != self.embedding_manifest_id:
                raise ValueError("embedding manifest identity must match embedding_manifest_id")
        if self.index_kind in {IndexKind.DENSE, IndexKind.VECTOR}:
            if self.embedding_manifest_id is None:
                raise ValueError("dense and vector indexes require an embedding manifest")
