import pytest

from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    Chunk,
    ChunkManifest,
    FrameManifest,
    FrameRef,
    FrameSamplingStrategy,
    ManifestKind,
    ManifestRef,
    Shot,
    ShotManifest,
    TimeRange,
)


def _manifest_ref(kind: ManifestKind, item_count: int) -> ManifestRef:
    artifact = ArtifactRef("manifest-file", ArtifactKind.MANIFEST, "manifests/data.json")
    return ManifestRef(f"{kind.value}-1", kind, artifact, "video-1", item_count)


def test_shot_manifest_accepts_ordered_non_overlapping_shots() -> None:
    shots = (
        Shot("shot-1", "video-1", TimeRange(0, 1_000), 0.9),
        Shot("shot-2", "video-1", TimeRange(1_000, 2_000), 0.8),
    )

    manifest = ShotManifest(
        ref=_manifest_ref(ManifestKind.SHOTS, 2),
        video_id="video-1",
        shots=shots,
    )

    assert manifest.shots == shots


def test_shot_manifest_rejects_overlapping_shots() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        ShotManifest(
            ref=_manifest_ref(ManifestKind.SHOTS, 2),
            video_id="video-1",
            shots=(
                Shot("shot-1", "video-1", TimeRange(0, 1_100)),
                Shot("shot-2", "video-1", TimeRange(1_000, 2_000)),
            ),
        )


def test_chunks_can_overlap_and_reference_shots_and_transcript() -> None:
    chunks = (
        Chunk(
            "chunk-1",
            "video-1",
            TimeRange(0, 2_000),
            shot_ids=("shot-1",),
            transcript_segment_ids=("transcript-1",),
        ),
        Chunk("chunk-2", "video-1", TimeRange(1_800, 3_000), shot_ids=("shot-2",)),
    )

    manifest = ChunkManifest(
        ref=_manifest_ref(ManifestKind.CHUNKS, 2),
        video_id="video-1",
        chunks=chunks,
    )

    assert manifest.chunks == chunks


def test_frame_manifest_keeps_requested_and_actual_timestamps() -> None:
    image = ArtifactRef("frame-image", ArtifactKind.FRAME_IMAGE, "frames/1.jpg")
    frame = FrameRef(
        frame_id="frame-1",
        video_id="video-1",
        requested_timestamp_ms=1_000,
        timestamp_ms=1_042,
        image=image,
    )

    manifest = FrameManifest(
        ref=_manifest_ref(ManifestKind.FRAMES, 1),
        video_id="video-1",
        strategy=FrameSamplingStrategy.SHOT_KEYFRAME,
        requested_ranges=(TimeRange(0, 2_000),),
        frames=(frame,),
        decoded_frames=10,
        dropped_duplicates=2,
    )

    assert manifest.frames[0].timestamp_ms == 1_042


def test_manifest_count_must_match_contained_frames() -> None:
    with pytest.raises(ValueError, match="item_count"):
        FrameManifest(
            ref=_manifest_ref(ManifestKind.FRAMES, 1),
            video_id="video-1",
            strategy=FrameSamplingStrategy.UNIFORM,
            requested_ranges=(TimeRange(0, 1_000),),
            frames=(),
            decoded_frames=0,
        )


def test_frame_manifest_requires_requested_range() -> None:
    with pytest.raises(ValueError, match="requested range"):
        FrameManifest(
            ref=_manifest_ref(ManifestKind.FRAMES, 0),
            video_id="video-1",
            strategy=FrameSamplingStrategy.UNIFORM,
            requested_ranges=(),
            frames=(),
            decoded_frames=0,
        )
