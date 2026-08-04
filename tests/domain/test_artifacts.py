from datetime import UTC, datetime

import pytest

from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    ManifestKind,
    ManifestRef,
    ProducerInfo,
    Provenance,
)


def test_artifact_ref_accepts_source_without_provenance() -> None:
    artifact = ArtifactRef(
        artifact_id="source-1",
        kind=ArtifactKind.SOURCE_VIDEO,
        uri="videos/source.mp4",
        sha256="a" * 64,
        size_bytes=1024,
    )

    assert artifact.provenance is None


def test_derived_artifact_records_reproducible_provenance() -> None:
    provenance = Provenance(
        producer=ProducerInfo(name="frame-sampler", version="1.0.0"),
        parameters_hash="params-123",
        source_video_id="video-1",
        source_artifact_ids=("source-1",),
        created_at=datetime(2026, 8, 3, tzinfo=UTC),
    )
    artifact = ArtifactRef(
        artifact_id="frame-1-image",
        kind=ArtifactKind.FRAME_IMAGE,
        uri="frames/frame-1.jpg",
        provenance=provenance,
    )

    assert artifact.provenance == provenance


@pytest.mark.parametrize("sha256", ["abc", "g" * 64, "a" * 63])
def test_artifact_ref_rejects_invalid_sha256(sha256: str) -> None:
    with pytest.raises(ValueError, match="sha256"):
        ArtifactRef(
            artifact_id="source-1",
            kind=ArtifactKind.SOURCE_VIDEO,
            uri="videos/source.mp4",
            sha256=sha256,
        )


def test_manifest_requires_manifest_artifact() -> None:
    image = ArtifactRef(
        artifact_id="image-1",
        kind=ArtifactKind.FRAME_IMAGE,
        uri="frames/1.jpg",
    )

    with pytest.raises(ValueError, match="kind MANIFEST"):
        ManifestRef(
            manifest_id="frames-1",
            kind=ManifestKind.FRAMES,
            artifact=image,
            source_video_id="video-1",
            item_count=1,
        )
