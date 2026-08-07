import pytest

from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    EmbeddingManifest,
    EvidenceAction,
    EvidenceBundle,
    EvidenceConflict,
    EvidenceItem,
    EvidenceModality,
    EvidenceScore,
    EvidenceVerificationReport,
    EvidenceVerificationStatus,
    IndexKind,
    IndexManifest,
    IndexModality,
    ManifestKind,
    ManifestRef,
    RetrievalHit,
    RetrievalResult,
    TimeRange,
)


def _manifest_ref(kind: ManifestKind, item_count: int) -> ManifestRef:
    artifact = ArtifactRef("manifest-file", ArtifactKind.MANIFEST, "manifests/data.json")
    return ManifestRef(f"{kind.value}-1", kind, artifact, "video-1", item_count)


def _evidence(evidence_id: str, text: str, start_ms: int) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        video_id="video-1",
        time_range=TimeRange(start_ms, start_ms + 1_000),
        modality=EvidenceModality.TRANSCRIPT,
        source_ids=(f"source-{evidence_id}",),
        text=text,
        scores=(EvidenceScore("bm25", 1.5),),
        confidence=0.8,
    )


def test_embedding_manifest_references_external_vector_artifact() -> None:
    vectors = ArtifactRef("vectors-1", ArtifactKind.EMBEDDING, "vectors/data.bin")
    manifest = EmbeddingManifest(
        ref=_manifest_ref(ManifestKind.EMBEDDINGS, 2),
        video_id="video-1",
        modality=IndexModality.TRANSCRIPT,
        embedding_space="text-v1",
        dimensions=768,
        item_ids=("segment-1", "segment-2"),
        embedding_artifact=vectors,
    )

    assert manifest.embedding_artifact == vectors


def test_dense_index_requires_embedding_manifest() -> None:
    index_artifact = ArtifactRef("index-file", ArtifactKind.INDEX, "indexes/data")

    with pytest.raises(ValueError, match="embedding manifest"):
        IndexManifest(
            ref=_manifest_ref(ManifestKind.INDEX, 2),
            video_id="video-1",
            modality=IndexModality.TRANSCRIPT,
            index_kind=IndexKind.DENSE,
            source_manifest_ids=("transcript-1",),
            index_artifact=index_artifact,
        )


def test_retrieval_ranks_are_consecutive() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        RetrievalResult(
            query="what happened?",
            hits=(RetrievalHit(2, _evidence("evidence-1", "A cup moved.", 0)),),
            searched_modalities=(EvidenceModality.TRANSCRIPT,),
        )


def test_evidence_bundle_preserves_conflict() -> None:
    first = _evidence("evidence-1", "The cup is full.", 0)
    second = _evidence("evidence-2", "The cup is empty.", 2_000)
    conflict = EvidenceConflict(
        conflict_id="conflict-1",
        evidence_ids=(first.evidence_id, second.evidence_id),
        description="The two statements disagree.",
    )

    bundle = EvidenceBundle(
        bundle_id="bundle-1",
        question="Is the cup full?",
        items=(first, second),
        conflicts=(conflict,),
    )

    assert bundle.modalities == frozenset({EvidenceModality.TRANSCRIPT})


def test_sufficient_verification_requires_answer_action() -> None:
    with pytest.raises(ValueError, match="directly support"):
        EvidenceVerificationReport(
            status=EvidenceVerificationStatus.SUFFICIENT,
            direct_support=True,
            temporal_coverage=1.0,
            cross_modal_consistency=1.0,
            confidence=0.9,
            recommended_actions=(),
        )

    report = EvidenceVerificationReport(
        status=EvidenceVerificationStatus.SUFFICIENT,
        direct_support=True,
        temporal_coverage=1.0,
        cross_modal_consistency=1.0,
        confidence=0.9,
        recommended_actions=(EvidenceAction.ANSWER,),
    )
    assert report.recommended_actions == (EvidenceAction.ANSWER,)
