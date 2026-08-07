from grounded_video_agent.domain import ArtifactKind, IndexKind, IndexModality, ManifestKind
from grounded_video_agent.workspace.catalog import (
    CatalogDocumentKind,
    CatalogKey,
    CatalogResourceType,
)

SOURCE_KEY = CatalogKey(
    CatalogResourceType.SOURCE,
    artifact_kind=ArtifactKind.SOURCE_VIDEO,
)
MEDIA_INSPECTION_KEY = CatalogKey(
    CatalogResourceType.DOCUMENT,
    document_kind=CatalogDocumentKind.MEDIA_INSPECTION,
)
AUDIO_KEY = CatalogKey(
    CatalogResourceType.DOCUMENT,
    document_kind=CatalogDocumentKind.AUDIO_ASSET,
)
SHOTS_KEY = CatalogKey(
    CatalogResourceType.MANIFEST,
    manifest_kind=ManifestKind.SHOTS,
)
TRANSCRIPT_KEY = CatalogKey(
    CatalogResourceType.MANIFEST,
    manifest_kind=ManifestKind.TRANSCRIPT,
)
CHUNKS_KEY = CatalogKey(
    CatalogResourceType.MANIFEST,
    manifest_kind=ManifestKind.CHUNKS,
)
TRANSCRIPT_EMBEDDINGS_KEY = CatalogKey(
    CatalogResourceType.MANIFEST,
    variant="transcript",
    manifest_kind=ManifestKind.EMBEDDINGS,
)
SPARSE_INDEX_KEY = CatalogKey(
    CatalogResourceType.INDEX,
    modality=IndexModality.TRANSCRIPT,
    index_kind=IndexKind.BM25,
)
DENSE_INDEX_KEY = CatalogKey(
    CatalogResourceType.INDEX,
    modality=IndexModality.TRANSCRIPT,
    index_kind=IndexKind.DENSE,
)
