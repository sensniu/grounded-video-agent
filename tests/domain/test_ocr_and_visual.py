import pytest

from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    BoundingBox,
    ManifestKind,
    ManifestRef,
    OCRManifest,
    OCRObservation,
    OCRSpan,
    TimeRange,
    VisualDescription,
    VisualDescriptionManifest,
    VisualDescriptionMode,
)


def _manifest_ref(kind: ManifestKind, item_count: int) -> ManifestRef:
    artifact = ArtifactRef("manifest-file", ArtifactKind.MANIFEST, "manifests/data.json")
    return ManifestRef(f"{kind.value}-1", kind, artifact, "video-1", item_count)


def test_bounding_box_uses_normalized_coordinates() -> None:
    assert BoundingBox(0.1, 0.2, 0.3, 0.4).width == 0.3

    with pytest.raises(ValueError, match="contained"):
        BoundingBox(0.8, 0.2, 0.3, 0.4)


def test_ocr_manifest_links_temporal_span_to_observations() -> None:
    observation = OCRObservation(
        observation_id="observation-1",
        video_id="video-1",
        frame_id="frame-1",
        timestamp_ms=1_000,
        raw_text="OPENAI",
        normalized_text="openai",
        bbox=BoundingBox(0.1, 0.1, 0.4, 0.2),
        confidence=0.95,
        language="en",
    )
    span = OCRSpan(
        span_id="span-1",
        video_id="video-1",
        time_range=TimeRange(900, 1_200),
        text="openai",
        observation_ids=("observation-1",),
        confidence=0.95,
    )

    manifest = OCRManifest(
        ref=_manifest_ref(ManifestKind.OCR, 1),
        video_id="video-1",
        observations=(observation,),
        spans=(span,),
    )

    assert manifest.spans[0].observation_ids == ("observation-1",)


def test_ocr_span_cannot_reference_unknown_observation() -> None:
    span = OCRSpan(
        span_id="span-1",
        video_id="video-1",
        time_range=TimeRange(0, 100),
        text="text",
        observation_ids=("missing",),
        confidence=0.5,
    )

    with pytest.raises(ValueError, match="same manifest"):
        OCRManifest(
            ref=_manifest_ref(ManifestKind.OCR, 0),
            video_id="video-1",
            observations=(),
            spans=(span,),
        )


def test_question_conditioned_description_requires_question() -> None:
    with pytest.raises(ValueError, match="requires a question"):
        VisualDescription(
            description_id="description-1",
            video_id="video-1",
            time_range=TimeRange(0, 1_000),
            text="A person raises a cup.",
            mode=VisualDescriptionMode.QUESTION_CONDITIONED,
        )


def test_visual_description_manifest() -> None:
    description = VisualDescription(
        description_id="description-1",
        video_id="video-1",
        time_range=TimeRange(0, 1_000),
        text="A person raises a cup.",
        mode=VisualDescriptionMode.GENERIC,
        frame_ids=("frame-1",),
    )

    manifest = VisualDescriptionManifest(
        ref=_manifest_ref(ManifestKind.VISUAL_DESCRIPTIONS, 1),
        video_id="video-1",
        descriptions=(description,),
    )

    assert manifest.descriptions == (description,)
