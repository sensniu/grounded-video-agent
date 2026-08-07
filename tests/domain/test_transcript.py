import pytest

from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    ManifestKind,
    ManifestRef,
    TimeRange,
    TranscriptManifest,
    TranscriptSegment,
    TranscriptSource,
    TranscriptWord,
)


def _manifest_ref(kind: ManifestKind, item_count: int) -> ManifestRef:
    artifact = ArtifactRef("manifest-file", ArtifactKind.MANIFEST, "manifests/data.json")
    return ManifestRef("manifest-1", kind, artifact, "video-1", item_count)


def test_transcript_unifies_word_timestamps_and_source() -> None:
    words = (
        TranscriptWord("hello", TimeRange(1_000, 1_400), 0.9),
        TranscriptWord("world", TimeRange(1_400, 2_000), 0.8),
    )
    segment = TranscriptSegment(
        segment_id="segment-1",
        video_id="video-1",
        time_range=TimeRange(1_000, 2_000),
        raw_text="Hello world",
        normalized_text="hello world",
        source=TranscriptSource.ASR,
        language="en",
        words=words,
    )
    manifest = TranscriptManifest(
        ref=_manifest_ref(ManifestKind.TRANSCRIPT, 1),
        video_id="video-1",
        source=TranscriptSource.ASR,
        segments=(segment,),
        language="en",
    )

    assert manifest.segments[0].words == words


def test_transcript_word_must_be_inside_segment() -> None:
    with pytest.raises(ValueError, match="contained"):
        TranscriptSegment(
            segment_id="segment-1",
            video_id="video-1",
            time_range=TimeRange(1_000, 2_000),
            raw_text="hello",
            normalized_text="hello",
            source=TranscriptSource.ASR,
            words=(TranscriptWord("hello", TimeRange(900, 1_100)),),
        )


def test_transcript_manifest_rejects_mixed_sources() -> None:
    segment = TranscriptSegment(
        segment_id="segment-1",
        video_id="video-1",
        time_range=TimeRange(0, 1_000),
        raw_text="subtitle",
        normalized_text="subtitle",
        source=TranscriptSource.EMBEDDED_SUBTITLE,
    )

    with pytest.raises(ValueError, match="manifest source"):
        TranscriptManifest(
            ref=_manifest_ref(ManifestKind.TRANSCRIPT, 1),
            video_id="video-1",
            source=TranscriptSource.ASR,
            segments=(segment,),
        )
