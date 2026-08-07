from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from grounded_video_agent.capabilities.audio.extraction import AudioExtractionRequest
from grounded_video_agent.capabilities.subtitles.embedded_extraction import (
    EmbeddedSubtitleExtractionCapability,
)
from grounded_video_agent.capabilities.subtitles.speech_transcription import (
    SpeechTranscriptionCapability,
    SpeechTranscriptionRequest,
)
from grounded_video_agent.capabilities.subtitles.speech_transcription.backend import (
    ASRSegment,
    ASRTranscript,
    ASRWord,
)
from grounded_video_agent.capabilities.temporal.chunking import (
    TemporalChunkingCapability,
    TemporalChunkingRequest,
)
from grounded_video_agent.capabilities.visual.frame_sampling import (
    FrameSamplingCapability,
    FrameSamplingRequest,
)
from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    AudioArtifact,
    CapabilityRequestContext,
    CapabilityStatus,
    ChunkBasis,
    FrameSamplingStrategy,
    ManifestKind,
    ManifestRef,
    Shot,
    ShotManifest,
    TimelineMapping,
    TimeRange,
    TranscriptManifest,
    TranscriptSegment,
    TranscriptSource,
    VideoAsset,
)


def _artifact(artifact_id: str, kind: ArtifactKind, uri: str = "/tmp/source") -> ArtifactRef:
    return ArtifactRef(artifact_id=artifact_id, kind=kind, uri=uri)


def _asset(video_id: str = "video-test") -> VideoAsset:
    return VideoAsset(
        video_id=video_id,
        source=_artifact("source-test", ArtifactKind.SOURCE_VIDEO),
        registered_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _shot_manifest(video_id: str = "video-test") -> ShotManifest:
    artifact = _artifact("shots-artifact", ArtifactKind.MANIFEST, "/tmp/shots.json")
    ref = ManifestRef("shots", ManifestKind.SHOTS, artifact, video_id, 3)
    return ShotManifest(
        ref=ref,
        video_id=video_id,
        shots=(
            Shot("shot-1", video_id, TimeRange(0, 8_000)),
            Shot("shot-2", video_id, TimeRange(8_000, 18_000)),
            Shot("shot-3", video_id, TimeRange(18_000, 30_000)),
        ),
    )


def _audio() -> AudioArtifact:
    artifact = _artifact("audio-artifact", ArtifactKind.AUDIO, "/tmp/audio.wav")
    source_range = TimeRange(10_000, 15_000)
    return AudioArtifact(
        audio_id="audio-test",
        video_id="video-test",
        artifact=artifact,
        source_range=source_range,
        timeline_mapping=TimelineMapping(
            "video-test",
            source_range,
            "audio-test",
            TimeRange(0, 5_000),
        ),
        stream_index=1,
        sample_rate_hz=16_000,
        channels=1,
    )


class _FakeBackend:
    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None,
        word_timestamps: bool,
    ) -> ASRTranscript:
        return ASRTranscript(
            language="zh",
            segments=(
                ASRSegment(
                    " 测试 字幕 ",
                    0.5,
                    2.0,
                    (ASRWord("测试", 0.5, 1.0, 0.9), ASRWord("字幕", 1.0, 2.0, 0.8)),
                ),
                ASRSegment("超出范围", 4.5, 6.0, (ASRWord("超出", 4.5, 6.0, 0.7),)),
            ),
        )


def test_audio_request_rejects_invalid_output_format() -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        AudioExtractionRequest(
            _asset(),
            TimeRange(0, 1_000),
            0,
            CapabilityRequestContext("audio"),
            sample_rate_hz=0,
        )


def test_embedded_subtitle_parser_normalizes_webvtt(tmp_path: Path) -> None:
    path = tmp_path / "subtitle.vtt"
    path.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.500\n<c.yellow>Hello &amp; world</c>\n",
        encoding="utf-8",
    )

    segments = EmbeddedSubtitleExtractionCapability._parse_vtt(
        path,
        video_id="video-test",
        operation_id="subtitle",
        stream_index=2,
        language="en",
    )

    assert len(segments) == 1
    assert segments[0].time_range == TimeRange(1_000, 2_500)
    assert segments[0].normalized_text == "Hello & world"
    assert segments[0].source is TranscriptSource.EMBEDDED_SUBTITLE


def test_speech_transcription_maps_to_source_timeline_and_clamps(tmp_path: Path) -> None:
    request = SpeechTranscriptionRequest(_audio(), CapabilityRequestContext("asr"))

    result = SpeechTranscriptionCapability(tmp_path, backend=_FakeBackend()).execute(request)

    assert result.status is CapabilityStatus.SUCCESS
    assert result.data is not None
    assert tuple(segment.time_range for segment in result.data.segments) == (
        TimeRange(10_500, 12_000),
        TimeRange(14_500, 15_000),
    )
    assert result.data.segments[1].words[0].time_range == TimeRange(14_500, 15_000)
    assert Path(result.data.ref.artifact.uri).is_file()


def test_temporal_chunking_falls_back_to_shot_chunks(tmp_path: Path) -> None:
    request = TemporalChunkingRequest(
        video_id="video-test",
        source_range=TimeRange(0, 30_000),
        shots=_shot_manifest(),
        context=CapabilityRequestContext("chunks"),
        target_duration_ms=7_000,
        max_duration_ms=15_000,
        overlap_ms=2_000,
    )

    result = TemporalChunkingCapability(tmp_path).execute(request)

    assert result.status is CapabilityStatus.PARTIAL
    assert result.data is not None
    assert tuple(chunk.time_range for chunk in result.data.chunks) == (
        TimeRange(0, 8_000),
        TimeRange(8_000, 18_000),
        TimeRange(18_000, 30_000),
    )
    assert result.data.chunks[0].shot_ids == ("shot-1",)
    assert result.data.chunks[0].basis is ChunkBasis.SHOT_FALLBACK
    assert Path(result.data.ref.artifact.uri).is_file()


def test_temporal_chunking_groups_transcript_and_aligns_inspection_range(
    tmp_path: Path,
) -> None:
    segments = (
        TranscriptSegment(
            "segment-1",
            "video-test",
            TimeRange(1_000, 4_000),
            "First sentence.",
            "First sentence.",
            TranscriptSource.ASR,
        ),
        TranscriptSegment(
            "segment-2",
            "video-test",
            TimeRange(4_200, 7_000),
            "Second sentence.",
            "Second sentence.",
            TranscriptSource.ASR,
        ),
    )
    transcript = TranscriptManifest(
        ManifestRef(
            "transcript",
            ManifestKind.TRANSCRIPT,
            _artifact("transcript-artifact", ArtifactKind.MANIFEST, "/tmp/transcript.json"),
            "video-test",
            2,
        ),
        "video-test",
        TranscriptSource.ASR,
        segments,
    )
    request = TemporalChunkingRequest(
        "video-test",
        TimeRange(0, 30_000),
        _shot_manifest(),
        CapabilityRequestContext("transcript-chunks"),
        transcript,
        target_duration_ms=20_000,
        max_duration_ms=30_000,
        overlap_ms=0,
        target_characters=10,
        max_characters=100,
    )

    result = TemporalChunkingCapability(tmp_path).execute(request)

    assert result.status is CapabilityStatus.SUCCESS
    assert result.data is not None
    assert tuple(chunk.time_range for chunk in result.data.chunks) == (
        TimeRange(1_000, 4_000),
        TimeRange(4_200, 7_000),
    )
    assert result.data.chunks[0].inspection_range == TimeRange(0, 8_000)
    assert result.data.chunks[0].text == "First sentence."
    assert result.data.chunks[0].basis is ChunkBasis.TRANSCRIPT


def test_frame_sampling_timestamp_strategies() -> None:
    uniform = FrameSamplingRequest(
        _asset(),
        (TimeRange(0, 1_000), TimeRange(2_000, 3_000)),
        FrameSamplingStrategy.UNIFORM,
        CapabilityRequestContext("uniform"),
        max_frames=4,
    )
    fixed = FrameSamplingRequest(
        _asset(),
        (TimeRange(0, 2_000),),
        FrameSamplingStrategy.FIXED_FPS,
        CapabilityRequestContext("fixed"),
        max_frames=4,
        fps=2,
    )
    keyframes = FrameSamplingRequest(
        _asset(),
        (TimeRange(0, 30_000),),
        FrameSamplingStrategy.SHOT_KEYFRAME,
        CapabilityRequestContext("keyframes"),
        max_frames=2,
        shots=_shot_manifest(),
    )

    assert FrameSamplingCapability._requested_timestamps(uniform, 4) == (250, 750, 2_250, 2_750)
    assert FrameSamplingCapability._requested_timestamps(fixed, 4) == (0, 500, 1_000, 1_500)
    assert FrameSamplingCapability._requested_timestamps(keyframes, 2) == (4_000, 13_000)


def test_frame_sampling_request_requires_strategy_inputs() -> None:
    with pytest.raises(ValueError, match="fps"):
        FrameSamplingRequest(
            _asset(),
            (TimeRange(0, 1_000),),
            FrameSamplingStrategy.FIXED_FPS,
            CapabilityRequestContext("missing-fps"),
        )
    with pytest.raises(ValueError, match="shots"):
        FrameSamplingRequest(
            _asset(),
            (TimeRange(0, 1_000),),
            FrameSamplingStrategy.SHOT_KEYFRAME,
            CapabilityRequestContext("missing-shots"),
        )
