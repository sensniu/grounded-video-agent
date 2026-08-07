from __future__ import annotations

from pathlib import Path

from grounded_video_agent.capabilities.media_inspection import VideoInspectionResult
from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    AudioArtifact,
    ManifestRef,
    ProducerInfo,
    Provenance,
    VideoAsset,
)
from grounded_video_agent.pipelines.preprocessing.derivation import DerivationSpec
from grounded_video_agent.workspace.catalog import (
    ArtifactCatalog,
    AudioAssetDocument,
    BasicMediaFlags,
    CatalogDocumentKind,
    CatalogDocumentRef,
    CatalogEntry,
    CatalogKey,
    CatalogRegistration,
    DocumentCodecRegistry,
    MediaInspectionDocument,
    MediaInspectionNextAction,
    PrimaryStreamSelection,
)


class CatalogPublisher:
    """Persist singleton documents and register all successful stage outputs."""

    def __init__(
        self,
        catalog: ArtifactCatalog,
        artifact_root: str | Path,
        *,
        document_codecs: DocumentCodecRegistry | None = None,
    ) -> None:
        self._catalog = catalog
        self._artifact_root = Path(artifact_root).resolve()
        self._document_codecs = document_codecs or DocumentCodecRegistry()

    def publish_inspection(
        self,
        result: VideoInspectionResult,
        video_asset: VideoAsset,
        key: CatalogKey,
        dependencies: tuple[str, ...],
        spec: DerivationSpec,
    ) -> tuple[MediaInspectionDocument, CatalogEntry]:
        context = result.video_context
        if context is None:
            raise ValueError("cannot publish a failed media inspection")
        ref = self._document_ref(
            video_asset.video_id,
            result.inspection_id,
            CatalogDocumentKind.MEDIA_INSPECTION,
            spec,
            (video_asset.source.artifact_id,),
        )
        document = MediaInspectionDocument(
            ref=ref,
            inspection_id=result.inspection_id,
            video_asset=video_asset,
            media_probe=context.media_probe,
            validation=context.validation,
            primary_streams=PrimaryStreamSelection.from_probe(context.media_probe),
            basic_flags=BasicMediaFlags.from_probe(context.media_probe),
            next_action=MediaInspectionNextAction(result.next_action.value),
        )
        self._document_codecs.dump(ref.artifact.uri, document)
        return document, self._register(
            video_asset.video_id,
            key,
            ref,
            result.inspection_id,
            dependencies,
            spec,
        )

    def publish_audio(
        self,
        audio: AudioArtifact,
        key: CatalogKey,
        operation_id: str,
        dependencies: tuple[str, ...],
        spec: DerivationSpec,
    ) -> tuple[AudioAssetDocument, CatalogEntry]:
        ref = self._document_ref(
            audio.video_id,
            f"audio_asset_{operation_id}",
            CatalogDocumentKind.AUDIO_ASSET,
            spec,
            (audio.artifact.artifact_id,),
        )
        document = AudioAssetDocument(ref, audio)
        self._document_codecs.dump(ref.artifact.uri, document)
        return document, self._register(
            audio.video_id,
            key,
            ref,
            operation_id,
            dependencies,
            spec,
        )

    def register_manifest(
        self,
        video_id: str,
        key: CatalogKey,
        reference: ManifestRef,
        operation_id: str,
        dependencies: tuple[str, ...],
        spec: DerivationSpec,
    ) -> CatalogEntry:
        return self._register(
            video_id,
            key,
            reference,
            operation_id,
            dependencies,
            spec,
        )

    def _register(
        self,
        video_id: str,
        key: CatalogKey,
        reference: CatalogDocumentRef | ManifestRef,
        operation_id: str,
        dependencies: tuple[str, ...],
        spec: DerivationSpec,
    ) -> CatalogEntry:
        self._catalog.register(
            video_id,
            CatalogRegistration(
                key=key,
                reference=reference,
                operation_id=operation_id,
                dependency_entry_ids=dependencies,
                producer_name=spec.producer_name,
                producer_version=spec.producer_version,
                parameters_hash=spec.key,
                derivation_key=spec.key,
            ),
        )
        return self._catalog.resolve(video_id, key).entry

    def _document_ref(
        self,
        video_id: str,
        document_id: str,
        kind: CatalogDocumentKind,
        spec: DerivationSpec,
        source_artifact_ids: tuple[str, ...],
    ) -> CatalogDocumentRef:
        path = self._artifact_root / "metadata" / video_id / f"{document_id}.json"
        provenance = Provenance(
            ProducerInfo(spec.producer_name, spec.producer_version),
            spec.key,
            video_id,
            source_artifact_ids,
        )
        return CatalogDocumentRef(
            document_id=document_id,
            kind=kind,
            artifact=ArtifactRef(
                artifact_id=f"{document_id}_artifact",
                kind=ArtifactKind.METADATA,
                uri=str(path),
                provenance=provenance,
            ),
            source_video_id=video_id,
        )
