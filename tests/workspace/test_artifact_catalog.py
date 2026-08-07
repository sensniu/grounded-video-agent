from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from grounded_video_agent.capabilities._support import write_json
from grounded_video_agent.capabilities.indexing.transcript_index import (
    TranscriptIndexingCapability,
    TranscriptIndexingRequest,
)
from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    AudioArtifact,
    AudioStreamInfo,
    CapabilityRequestContext,
    ContainerInfo,
    FrameRate,
    IndexKind,
    IndexManifest,
    IndexModality,
    ManifestKind,
    ManifestRef,
    MediaProbe,
    ProducerInfo,
    Provenance,
    ShotManifest,
    SubtitleStreamInfo,
    TimelineMapping,
    TimeRange,
    TranscriptManifest,
    TranscriptSegment,
    TranscriptSource,
    ValidationReport,
    VideoAsset,
    VideoClipArtifact,
    VideoStreamInfo,
)
from grounded_video_agent.workspace.catalog import (
    AudioAssetDocument,
    BasicMediaFlags,
    CatalogAuditIssueCode,
    CatalogDocumentKind,
    CatalogDocumentRef,
    CatalogEntryState,
    CatalogError,
    CatalogErrorCode,
    CatalogKey,
    CatalogRegistration,
    CatalogResourceType,
    DocumentCodecRegistry,
    FilesystemArtifactCatalog,
    MediaInspectionDocument,
    MediaInspectionNextAction,
    PrimaryStreamSelection,
    VideoClipDocument,
)

VIDEO_ID = "video_catalog_test"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup(tmp_path: Path) -> tuple[FilesystemArtifactCatalog, Path, Path, VideoAsset]:
    input_root = tmp_path / "inputs"
    artifact_root = tmp_path / "artifacts"
    source_path = input_root / "video.mp4"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"source-video")
    source = ArtifactRef(
        "source-video",
        ArtifactKind.SOURCE_VIDEO,
        str(source_path),
        _digest(source_path),
        source_path.stat().st_size,
    )
    asset = VideoAsset(VIDEO_ID, source, "video.mp4")
    catalog = FilesystemArtifactCatalog(
        artifact_root / "catalog",
        artifact_root=artifact_root,
        input_roots=(input_root,),
    )
    return catalog, input_root, artifact_root, asset


def _transcript(
    artifact_root: Path,
    manifest_id: str,
    text: str,
) -> TranscriptManifest:
    path = artifact_root / "transcripts" / VIDEO_ID / f"{manifest_id}.json"
    provenance = Provenance(
        ProducerInfo("test-transcript", "1"),
        hashlib.sha256(text.encode()).hexdigest(),
        VIDEO_ID,
        ("source-video",),
    )
    ref = ManifestRef(
        manifest_id,
        ManifestKind.TRANSCRIPT,
        ArtifactRef(
            f"{manifest_id}-artifact",
            ArtifactKind.MANIFEST,
            str(path),
            provenance=provenance,
        ),
        VIDEO_ID,
        1,
    )
    manifest = TranscriptManifest(
        ref,
        VIDEO_ID,
        TranscriptSource.ASR,
        (
            TranscriptSegment(
                f"{manifest_id}-segment",
                VIDEO_ID,
                TimeRange(0, 1_000),
                text,
                text.lower(),
                TranscriptSource.ASR,
                language="en",
            ),
        ),
        "en",
    )
    write_json(path, manifest)
    return manifest


def _transcript_key() -> CatalogKey:
    return CatalogKey(
        CatalogResourceType.MANIFEST,
        manifest_kind=ManifestKind.TRANSCRIPT,
    )


def _index_key() -> CatalogKey:
    return CatalogKey(
        CatalogResourceType.INDEX,
        modality=IndexModality.TRANSCRIPT,
        index_kind=IndexKind.BM25,
    )


def _video_clip_key() -> CatalogKey:
    return CatalogKey(
        CatalogResourceType.DOCUMENT,
        variant="evidence_test",
        document_kind=CatalogDocumentKind.VIDEO_CLIP,
    )


def test_catalog_loads_typed_video_clip_document(tmp_path: Path) -> None:
    catalog, _, artifact_root, asset = _setup(tmp_path)
    snapshot = catalog.create_video(asset)
    source_entry = snapshot.entries[0]
    clip_path = artifact_root / "clips" / VIDEO_ID / "evidence.mp4"
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"clip")
    provenance = Provenance(
        ProducerInfo("test-clip", "1"),
        hashlib.sha256(b"clip-parameters").hexdigest(),
        VIDEO_ID,
        (asset.source.artifact_id,),
    )
    clip_id = "evidence-clip"
    source_range = TimeRange(100, 900)
    clip = VideoClipArtifact(
        clip_id,
        VIDEO_ID,
        ArtifactRef(
            "evidence-clip-artifact",
            ArtifactKind.VIDEO_CLIP,
            str(clip_path),
            _digest(clip_path),
            clip_path.stat().st_size,
            provenance,
        ),
        source_range,
        source_range,
        TimelineMapping(VIDEO_ID, source_range, clip_id, TimeRange(0, 800)),
        includes_audio=True,
    )
    ref = _document_ref(
        artifact_root,
        "clip-document",
        CatalogDocumentKind.VIDEO_CLIP,
    )
    document = VideoClipDocument(ref, clip, ("evidence-1",))
    DocumentCodecRegistry().dump(ref.artifact.uri, document)
    derivation_key = provenance.parameters_hash
    catalog.register(
        VIDEO_ID,
        CatalogRegistration(
            _video_clip_key(),
            ref,
            "clip-operation",
            (source_entry.entry_id,),
            derivation_key=derivation_key,
        ),
    )

    loaded = catalog.load_document(VIDEO_ID, _video_clip_key(), VideoClipDocument)
    assert loaded == document
    assert catalog.audit(VIDEO_ID, deep=True).is_valid


def _document_ref(
    artifact_root: Path,
    document_id: str,
    kind: CatalogDocumentKind,
) -> CatalogDocumentRef:
    path = artifact_root / "metadata" / VIDEO_ID / f"{document_id}.json"
    provenance = Provenance(
        ProducerInfo("test-document", "1"),
        hashlib.sha256(document_id.encode()).hexdigest(),
        VIDEO_ID,
        ("source-video",),
    )
    return CatalogDocumentRef(
        document_id,
        kind,
        ArtifactRef(
            f"{document_id}-artifact",
            ArtifactKind.METADATA,
            str(path),
            provenance=provenance,
        ),
        VIDEO_ID,
    )


def _inspection_document(
    artifact_root: Path,
    asset: VideoAsset,
) -> MediaInspectionDocument:
    ref = _document_ref(
        artifact_root,
        "inspection-v1",
        CatalogDocumentKind.MEDIA_INSPECTION,
    )
    probe = MediaProbe(
        VIDEO_ID,
        ContainerInfo(("mov", "mp4"), duration_ms=1_000, size_bytes=12),
        (
            VideoStreamInfo(
                0,
                "h264",
                640,
                360,
                frame_rate=FrameRate(25, 1),
                average_frame_rate=FrameRate(25, 1),
                is_default=True,
            ),
        ),
        (AudioStreamInfo(1, "aac", sample_rate_hz=16_000, channels=1, is_default=True),),
        (SubtitleStreamInfo(2, "subrip", language="en", is_default=True),),
    )
    document = MediaInspectionDocument(
        ref,
        "inspection-operation-v1",
        asset,
        probe,
        ValidationReport(VIDEO_ID, ProducerInfo("test-validator", "1")),
        PrimaryStreamSelection.from_probe(probe),
        BasicMediaFlags.from_probe(probe),
        MediaInspectionNextAction.PROCEED,
    )
    write_json(Path(ref.artifact.uri), document)
    return document


def _register_transcript(
    catalog: FilesystemArtifactCatalog,
    manifest: TranscriptManifest,
    dependency: str,
    operation_id: str,
    *,
    expected_revision: int | None = None,
):
    provenance = manifest.ref.artifact.provenance
    assert provenance is not None
    return catalog.register(
        VIDEO_ID,
        CatalogRegistration(
            _transcript_key(),
            manifest.ref,
            operation_id,
            (dependency,),
            derivation_key=provenance.parameters_hash,
        ),
        expected_revision=expected_revision,
    )


def test_catalog_versions_loads_and_marks_dependent_index_stale(tmp_path: Path) -> None:
    catalog, _, artifact_root, asset = _setup(tmp_path)
    created = catalog.create_video(asset)
    assert created.revision == 1
    assert catalog.create_video(asset).revision == 1
    source_entry = created.entries[0]

    first = _transcript(artifact_root, "transcript-v1", "A red car arrives")
    registered = _register_transcript(
        catalog,
        first,
        source_entry.entry_id,
        "transcript-operation-1",
        expected_revision=1,
    )
    assert registered.revision == 2
    first_entry = catalog.resolve(VIDEO_ID, _transcript_key()).entry
    loaded = catalog.load_manifest(VIDEO_ID, _transcript_key(), TranscriptManifest)
    assert loaded == first

    duplicate = _register_transcript(
        catalog,
        first,
        source_entry.entry_id,
        "transcript-operation-retry",
    )
    assert duplicate.revision == 2
    assert len(duplicate.entries) == 2

    index_result = TranscriptIndexingCapability(artifact_root).execute(
        TranscriptIndexingRequest(
            first,
            CapabilityRequestContext("catalog-index"),
        )
    )
    assert index_result.data is not None
    indexed = catalog.register(
        VIDEO_ID,
        CatalogRegistration(
            _index_key(),
            index_result.data.ref,
            "catalog-index",
            (first_entry.entry_id,),
        ),
    )
    assert indexed.revision == 3
    loaded_index = catalog.load_manifest(VIDEO_ID, _index_key(), IndexManifest)
    assert loaded_index.index_kind is IndexKind.BM25

    second = _transcript(artifact_root, "transcript-v2", "Someone opens a door")
    updated = _register_transcript(
        catalog,
        second,
        source_entry.entry_id,
        "transcript-operation-2",
    )
    assert updated.revision == 4
    with pytest.raises(CatalogError) as stale:
        catalog.resolve(VIDEO_ID, _index_key())
    assert stale.value.code is CatalogErrorCode.STALE_RESOURCE

    resolved_stale = catalog.resolve(VIDEO_ID, _index_key(), require_fresh=False)
    assert resolved_stale.state is CatalogEntryState.STALE
    report = catalog.audit(VIDEO_ID, deep=True)
    states = {entry.entry_id: entry.state for entry in report.entries}
    assert states[first_entry.entry_id] is CatalogEntryState.SUPERSEDED
    assert states[resolved_stale.entry.entry_id] is CatalogEntryState.STALE
    assert any(issue.code is CatalogAuditIssueCode.STALE_DEPENDENCY for issue in report.issues)

    restored = catalog.activate(VIDEO_ID, _transcript_key(), first_entry.entry_id)
    assert restored.revision == 5
    assert catalog.resolve(VIDEO_ID, _index_key()).state is CatalogEntryState.AVAILABLE


def test_catalog_rejects_revision_conflict_and_outside_path(tmp_path: Path) -> None:
    catalog, _, artifact_root, asset = _setup(tmp_path)
    created = catalog.create_video(asset)
    transcript = _transcript(artifact_root, "transcript", "hello")
    with pytest.raises(CatalogError) as conflict:
        _register_transcript(
            catalog,
            transcript,
            created.entries[0].entry_id,
            "conflict",
            expected_revision=99,
        )
    assert conflict.value.code is CatalogErrorCode.REVISION_CONFLICT

    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(CatalogError) as forbidden:
        catalog.register(
            VIDEO_ID,
            CatalogRegistration(
                CatalogKey(
                    CatalogResourceType.ARTIFACT,
                    artifact_kind=ArtifactKind.OTHER,
                ),
                ArtifactRef("outside", ArtifactKind.OTHER, str(outside)),
                "outside-operation",
                producer_name="test",
                producer_version="1",
            ),
        )
    assert forbidden.value.code is CatalogErrorCode.PATH_NOT_ALLOWED


def test_catalog_detects_missing_and_tampered_resources(tmp_path: Path) -> None:
    catalog, _, artifact_root, asset = _setup(tmp_path)
    source_entry = catalog.create_video(asset).entries[0]
    transcript = _transcript(artifact_root, "transcript", "original")
    registered = _register_transcript(
        catalog,
        transcript,
        source_entry.entry_id,
        "register-transcript",
    )
    transcript_entry = catalog.resolve(VIDEO_ID, _transcript_key()).entry
    path = Path(transcript.ref.artifact.uri)
    path.write_text("{}", encoding="utf-8")

    report = catalog.audit(VIDEO_ID, deep=True)
    states = {entry.entry_id: entry.state for entry in report.entries}
    assert states[transcript_entry.entry_id] is CatalogEntryState.CORRUPT
    assert any(issue.code is CatalogAuditIssueCode.HASH_MISMATCH for issue in report.issues)
    with pytest.raises(CatalogError) as changed:
        catalog.load_manifest(VIDEO_ID, _transcript_key(), TranscriptManifest)
    assert changed.value.code is CatalogErrorCode.HASH_MISMATCH

    replacement = _transcript(artifact_root, "transcript-2", "replacement")
    registered = _register_transcript(
        catalog,
        replacement,
        source_entry.entry_id,
        "replacement",
        expected_revision=registered.revision,
    )
    replacement_entry = catalog.resolve(VIDEO_ID, _transcript_key()).entry
    Path(replacement.ref.artifact.uri).unlink()
    resolved = catalog.resolve(VIDEO_ID, _transcript_key(), require_fresh=False)
    assert resolved.state is CatalogEntryState.MISSING
    missing_report = catalog.audit(VIDEO_ID)
    states = {entry.entry_id: entry.state for entry in missing_report.entries}
    assert states[replacement_entry.entry_id] is CatalogEntryState.MISSING
    assert registered.revision == 3


def test_catalog_rejects_wrong_manifest_type(tmp_path: Path) -> None:
    catalog, _, artifact_root, asset = _setup(tmp_path)
    source_entry = catalog.create_video(asset).entries[0]
    transcript = _transcript(artifact_root, "transcript", "hello")
    _register_transcript(
        catalog,
        transcript,
        source_entry.entry_id,
        "register-transcript",
    )

    with pytest.raises(CatalogError) as mismatch:
        catalog.load_manifest(VIDEO_ID, _transcript_key(), ShotManifest)
    assert mismatch.value.code is CatalogErrorCode.TYPE_MISMATCH


def test_catalog_registers_and_loads_typed_documents(tmp_path: Path) -> None:
    catalog, _, artifact_root, asset = _setup(tmp_path)
    source_entry = catalog.create_video(asset).entries[0]
    inspection = _inspection_document(artifact_root, asset)
    inspection_key = CatalogKey(
        CatalogResourceType.DOCUMENT,
        document_kind=CatalogDocumentKind.MEDIA_INSPECTION,
    )
    catalog.register(
        VIDEO_ID,
        CatalogRegistration(
            inspection_key,
            inspection.ref,
            "register-inspection",
            (source_entry.entry_id,),
        ),
    )

    assert catalog.load_document(VIDEO_ID, inspection_key, MediaInspectionDocument) == inspection

    audio_path = artifact_root / "audio" / VIDEO_ID / "audio.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"test-wave")
    audio_provenance = Provenance(
        ProducerInfo("test-audio", "1"),
        hashlib.sha256(b"audio-parameters").hexdigest(),
        VIDEO_ID,
        ("source-video",),
    )
    source_range = TimeRange(0, 1_000)
    audio = AudioArtifact(
        "audio-v1",
        VIDEO_ID,
        ArtifactRef(
            "audio-v1-artifact",
            ArtifactKind.AUDIO,
            str(audio_path),
            _digest(audio_path),
            audio_path.stat().st_size,
            audio_provenance,
        ),
        source_range,
        TimelineMapping(VIDEO_ID, source_range, "audio-v1", TimeRange(0, 1_000)),
        1,
        16_000,
        1,
    )
    audio_ref = _document_ref(artifact_root, "audio-document-v1", CatalogDocumentKind.AUDIO_ASSET)
    audio_document = AudioAssetDocument(audio_ref, audio)
    write_json(Path(audio_ref.artifact.uri), audio_document)
    audio_key = CatalogKey(
        CatalogResourceType.DOCUMENT,
        document_kind=CatalogDocumentKind.AUDIO_ASSET,
    )
    catalog.register(
        VIDEO_ID,
        CatalogRegistration(
            audio_key,
            audio_ref,
            "register-audio-document",
            (source_entry.entry_id,),
        ),
    )

    assert catalog.load_document(VIDEO_ID, audio_key, AudioAssetDocument) == audio_document
    assert catalog.audit(VIDEO_ID, deep=True).is_valid
    with pytest.raises(CatalogError) as mismatch:
        catalog.load_document(VIDEO_ID, inspection_key, AudioAssetDocument)
    assert mismatch.value.code is CatalogErrorCode.TYPE_MISMATCH
    with pytest.raises(CatalogError) as direct_artifact:
        catalog.load_artifact(VIDEO_ID, audio_key)
    assert direct_artifact.value.code is CatalogErrorCode.TYPE_MISMATCH


def test_catalog_finds_only_intact_reusable_resources(tmp_path: Path) -> None:
    catalog, _, artifact_root, asset = _setup(tmp_path)
    source_entry = catalog.create_video(asset).entries[0]
    transcript = _transcript(artifact_root, "reusable-transcript", "reusable text")
    provenance = transcript.ref.artifact.provenance
    assert provenance is not None
    derivation_key = provenance.parameters_hash
    dependencies = (source_entry.entry_id,)
    _register_transcript(
        catalog,
        transcript,
        source_entry.entry_id,
        "register-reusable-transcript",
    )

    reusable = catalog.find_reusable(
        VIDEO_ID,
        _transcript_key(),
        derivation_key,
        dependencies,
    )
    assert reusable is not None
    assert reusable.entry.reference == transcript.ref
    assert reusable.state is CatalogEntryState.AVAILABLE
    assert (
        catalog.find_reusable(VIDEO_ID, _transcript_key(), "f" * 64, dependencies) is None
    )

    Path(transcript.ref.artifact.uri).write_text("tampered", encoding="utf-8")
    assert (
        catalog.find_reusable(
            VIDEO_ID,
            _transcript_key(),
            derivation_key,
            dependencies,
        )
        is None
    )
