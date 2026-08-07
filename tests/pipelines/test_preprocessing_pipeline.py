from __future__ import annotations

from pathlib import Path
from typing import Any

from grounded_video_agent.capabilities._support import (
    file_artifact,
    make_provenance,
    manifest_ref,
    write_json,
)
from grounded_video_agent.capabilities.indexing.dense_index import DenseIndexingCapability
from grounded_video_agent.capabilities.indexing.transcript_index import (
    TranscriptIndexingCapability,
)
from grounded_video_agent.capabilities.media_inspection import MediaInspectionCapability
from grounded_video_agent.capabilities.media_inspection.ffprobe import RawProbeResult
from grounded_video_agent.capabilities.temporal.chunking import TemporalChunkingCapability
from grounded_video_agent.domain import (
    ArtifactKind,
    AudioArtifact,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    ChunkBasis,
    ChunkManifest,
    ManifestKind,
    Shot,
    ShotManifest,
    TimelineMapping,
    TimeRange,
    TranscriptManifest,
    TranscriptSegment,
    TranscriptSource,
)
from grounded_video_agent.infrastructure.embeddings import EmbeddingModelInfo
from grounded_video_agent.input import VideoRegistrar
from grounded_video_agent.pipelines.preprocessing import (
    DenseIndexPolicy,
    PipelineStage,
    PipelineStageStatus,
    PipelineStatus,
    PreprocessingConfig,
    PreprocessingPipeline,
    PreprocessingRequest,
)
from grounded_video_agent.pipelines.preprocessing.dependencies import (
    PreprocessingDependencies,
)
from grounded_video_agent.pipelines.preprocessing.keys import CHUNKS_KEY
from grounded_video_agent.pipelines.preprocessing.publication import CatalogPublisher
from grounded_video_agent.workspace.catalog import FilesystemArtifactCatalog


def _probe_payload(
    *,
    include_audio: bool = True,
    include_subtitles: bool = True,
) -> dict[str, Any]:
    streams = [
            {
                "index": 0,
                "codec_name": "h264",
                "codec_type": "video",
                "width": 640,
                "height": 360,
                "r_frame_rate": "25/1",
                "avg_frame_rate": "25/1",
                "duration": "12.000",
                "disposition": {"default": 1, "attached_pic": 0},
            },
    ]
    if include_audio:
        streams.append(
            {
                "index": 1,
                "codec_name": "aac",
                "codec_type": "audio",
                "sample_rate": "48000",
                "channels": 2,
                "duration": "12.000",
                "disposition": {"default": 1},
            }
        )
    if include_subtitles:
        streams.append(
            {
                "index": 2,
                "codec_name": "mov_text",
                "codec_type": "subtitle",
                "duration": "12.000",
                "disposition": {"default": 1},
                "tags": {"language": "eng"},
            }
        )
    return {
        "streams": streams,
        "format": {
            "format_name": "mov,mp4",
            "duration": "12.000",
            "size": "5",
            "probe_score": 100,
        },
    }


class _ProbeRunner:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.calls = 0
        self._payload = payload or _probe_payload()

    def probe(self, source_path: str | Path) -> RawProbeResult:
        self.calls += 1
        return RawProbeResult(self._payload, "", 1)


class _ShotDetector:
    VERSION = "test-1"

    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root
        self.calls = 0

    def execute(self, request: Any) -> CapabilityResult[ShotManifest]:
        self.calls += 1
        provenance = make_provenance(
            "test-shot-detector",
            self.VERSION,
            {"threshold": request.threshold},
            video_id=request.video_asset.video_id,
            source_artifact_ids=(request.video_asset.source.artifact_id,),
        )
        path = (
            self._artifact_root
            / "manifests"
            / request.video_asset.video_id
            / f"shots_{request.context.operation_id}.json"
        )
        ref = manifest_ref(
            path,
            manifest_id=f"shots_{request.context.operation_id}",
            kind=ManifestKind.SHOTS,
            video_id=request.video_asset.video_id,
            item_count=2,
            provenance=provenance,
        )
        manifest = ShotManifest(
            ref,
            request.video_asset.video_id,
            (
                Shot("shot-1", request.video_asset.video_id, TimeRange(0, 6_000)),
                Shot("shot-2", request.video_asset.video_id, TimeRange(6_000, 12_000)),
            ),
        )
        write_json(path, manifest)
        return CapabilityResult(
            CapabilityStatus.SUCCESS,
            manifest,
            CapabilityUsage(input_items=1, output_items=2),
            (ref.artifact,),
            provenance=provenance,
        )


class _SubtitleExtractor:
    VERSION = "test-1"

    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root
        self.calls = 0

    def execute(self, request: Any) -> CapabilityResult[TranscriptManifest]:
        self.calls += 1
        provenance = make_provenance(
            "test-subtitle-extractor",
            self.VERSION,
            {"stream_index": request.stream_index},
            video_id=request.video_asset.video_id,
            source_artifact_ids=(request.video_asset.source.artifact_id,),
        )
        path = (
            self._artifact_root
            / "subtitles"
            / request.video_asset.video_id
            / f"transcript_{request.context.operation_id}.json"
        )
        ref = manifest_ref(
            path,
            manifest_id=f"transcript_{request.context.operation_id}",
            kind=ManifestKind.TRANSCRIPT,
            video_id=request.video_asset.video_id,
            item_count=2,
            provenance=provenance,
        )
        segments = (
            TranscriptSegment(
                "segment-1",
                request.video_asset.video_id,
                TimeRange(500, 3_000),
                "A red car arrives.",
                "a red car arrives.",
                TranscriptSource.EMBEDDED_SUBTITLE,
                language="eng",
                source_stream_index=request.stream_index,
            ),
            TranscriptSegment(
                "segment-2",
                request.video_asset.video_id,
                TimeRange(6_500, 9_000),
                "Someone opens the door.",
                "someone opens the door.",
                TranscriptSource.EMBEDDED_SUBTITLE,
                language="eng",
                source_stream_index=request.stream_index,
            ),
        )
        manifest = TranscriptManifest(
            ref,
            request.video_asset.video_id,
            TranscriptSource.EMBEDDED_SUBTITLE,
            segments,
            "eng",
        )
        write_json(path, manifest)
        return CapabilityResult(
            CapabilityStatus.SUCCESS,
            manifest,
            CapabilityUsage(input_items=1, output_items=2),
            (ref.artifact,),
            provenance=provenance,
        )


class _UnexpectedCapability:
    def execute(self, request: Any) -> Any:
        raise AssertionError("unexpected fallback capability call")


class _EmbeddingBackend:
    def get_model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo("test-embedding", "1", "test:embedding", 2)

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0)


class _AudioExtractor:
    VERSION = "test-1"

    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root

    def execute(self, request: Any) -> CapabilityResult[AudioArtifact]:
        provenance = make_provenance(
            "test-audio-extractor",
            self.VERSION,
            {"stream_index": request.stream_index},
            video_id=request.video_asset.video_id,
            source_artifact_ids=(request.video_asset.source.artifact_id,),
        )
        audio_id = f"audio_{request.context.operation_id}"
        path = self._artifact_root / "audio" / request.video_asset.video_id / f"{audio_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"wave")
        artifact = file_artifact(
            path,
            artifact_id=f"{audio_id}_artifact",
            kind=ArtifactKind.AUDIO,
            provenance=provenance,
        )
        audio = AudioArtifact(
            audio_id,
            request.video_asset.video_id,
            artifact,
            request.source_range,
            TimelineMapping(
                request.video_asset.video_id,
                request.source_range,
                audio_id,
                TimeRange(0, request.source_range.duration_ms),
            ),
            request.stream_index,
            request.sample_rate_hz,
            request.channels,
        )
        return CapabilityResult(
            CapabilityStatus.SUCCESS,
            audio,
            CapabilityUsage(input_items=1, output_items=1),
            (artifact,),
            provenance=provenance,
        )


class _SpeechTranscriber:
    VERSION = "test-1"

    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root

    def execute(self, request: Any) -> CapabilityResult[TranscriptManifest]:
        provenance = make_provenance(
            "test-speech-transcriber",
            self.VERSION,
            {"language": request.language_hint},
            video_id=request.audio.video_id,
            source_artifact_ids=(request.audio.artifact.artifact_id,),
        )
        path = (
            self._artifact_root
            / "transcripts"
            / request.audio.video_id
            / f"asr_{request.context.operation_id}.json"
        )
        ref = manifest_ref(
            path,
            manifest_id=f"asr_{request.context.operation_id}",
            kind=ManifestKind.TRANSCRIPT,
            video_id=request.audio.video_id,
            item_count=1,
            provenance=provenance,
        )
        manifest = TranscriptManifest(
            ref,
            request.audio.video_id,
            TranscriptSource.ASR,
            (
                TranscriptSegment(
                    "asr-segment",
                    request.audio.video_id,
                    TimeRange(1_000, 5_000),
                    "Fallback speech transcript.",
                    "fallback speech transcript.",
                    TranscriptSource.ASR,
                    language="eng",
                    source_stream_index=request.audio.stream_index,
                ),
            ),
            "eng",
        )
        write_json(path, manifest)
        return CapabilityResult(
            CapabilityStatus.SUCCESS,
            manifest,
            CapabilityUsage(input_items=1, output_items=1, model_calls=1),
            (ref.artifact,),
            provenance=provenance,
        )


def test_preprocessing_pipeline_builds_catalog_and_reuses_every_stage(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    artifact_root = tmp_path / "artifacts"
    input_root.mkdir()
    (input_root / "video.mp4").write_bytes(b"video")
    registrar = VideoRegistrar(input_root)
    probe = _ProbeRunner()
    shot_detector = _ShotDetector(artifact_root)
    subtitle_extractor = _SubtitleExtractor(artifact_root)
    catalog = FilesystemArtifactCatalog(
        artifact_root / "catalog",
        artifact_root=artifact_root,
        input_roots=(input_root,),
    )
    pipeline = PreprocessingPipeline(
        catalog,
        CatalogPublisher(catalog, artifact_root),
        PreprocessingDependencies(
            registrar=registrar,
            inspector=MediaInspectionCapability(
                input_root=input_root,
                registrar=registrar,
                probe_runner=probe,
            ),
            shot_detector=shot_detector,
            embedded_subtitle_extractor=subtitle_extractor,
            audio_extractor=_UnexpectedCapability(),
            speech_transcriber=_UnexpectedCapability(),
            chunker=TemporalChunkingCapability(artifact_root),
            transcript_indexer=TranscriptIndexingCapability(artifact_root),
            dense_indexer=DenseIndexingCapability(_EmbeddingBackend(), artifact_root),
        ),
        PreprocessingConfig(dense_index_policy=DenseIndexPolicy.OPTIONAL),
    )

    first = pipeline.run(PreprocessingRequest("video.mp4"))

    assert first.status is PipelineStatus.READY
    assert first.video_id is not None
    assert first.readiness.media_ready
    assert first.readiness.shots_ready
    assert first.readiness.transcript_ready
    assert first.readiness.timeline_ready
    assert first.readiness.sparse_search_ready
    assert first.readiness.dense_search_ready
    assert first.entries.embedding_entry_id is not None
    assert first.entries.dense_index_entry_id is not None
    chunks = catalog.load_manifest(first.video_id, CHUNKS_KEY, ChunkManifest)
    assert chunks.chunks[0].basis is ChunkBasis.TRANSCRIPT
    assert chunks.chunks[0].text == "a red car arrives."
    assert chunks.chunks[0].time_range == TimeRange(500, 3_000)
    assert chunks.chunks[0].inspection_range == TimeRange(0, 6_000)

    second = pipeline.run("video.mp4")

    assert second.status is PipelineStatus.READY
    cache_stages = {
        report.stage
        for report in second.stages
        if report.status is PipelineStageStatus.CACHE_HIT
    }
    assert {
        PipelineStage.MEDIA_INSPECTION,
        PipelineStage.SHOT_DETECTION,
        PipelineStage.EMBEDDED_SUBTITLES,
        PipelineStage.CHUNKING,
        PipelineStage.SPARSE_INDEXING,
        PipelineStage.DENSE_INDEXING,
    }.issubset(cache_stages)
    assert probe.calls == 1
    assert shot_detector.calls == 1
    assert subtitle_extractor.calls == 1


def test_preprocessing_pipeline_uses_asr_when_embedded_subtitles_are_absent(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "inputs"
    artifact_root = tmp_path / "artifacts"
    input_root.mkdir()
    (input_root / "video.mp4").write_bytes(b"video")
    registrar = VideoRegistrar(input_root)
    catalog = FilesystemArtifactCatalog(
        artifact_root / "catalog",
        artifact_root=artifact_root,
        input_roots=(input_root,),
    )
    pipeline = PreprocessingPipeline(
        catalog,
        CatalogPublisher(catalog, artifact_root),
        PreprocessingDependencies(
            registrar=registrar,
            inspector=MediaInspectionCapability(
                input_root=input_root,
                registrar=registrar,
                probe_runner=_ProbeRunner(_probe_payload(include_subtitles=False)),
            ),
            shot_detector=_ShotDetector(artifact_root),
            embedded_subtitle_extractor=_UnexpectedCapability(),
            audio_extractor=_AudioExtractor(artifact_root),
            speech_transcriber=_SpeechTranscriber(artifact_root),
            chunker=TemporalChunkingCapability(artifact_root),
            transcript_indexer=TranscriptIndexingCapability(artifact_root),
        ),
    )

    result = pipeline.run("video.mp4")

    assert result.status is PipelineStatus.READY
    assert result.entries.audio_entry_id is not None
    assert result.entries.transcript_entry_id is not None
    statuses = {report.stage: report.status for report in result.stages}
    assert statuses[PipelineStage.EMBEDDED_SUBTITLES] is PipelineStageStatus.SKIPPED
    assert statuses[PipelineStage.AUDIO_EXTRACTION] is PipelineStageStatus.SUCCEEDED
    assert statuses[PipelineStage.SPEECH_TRANSCRIPTION] is PipelineStageStatus.SUCCEEDED


def test_preprocessing_pipeline_returns_partial_shot_timeline_without_text(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "inputs"
    artifact_root = tmp_path / "artifacts"
    input_root.mkdir()
    (input_root / "video.mp4").write_bytes(b"video")
    registrar = VideoRegistrar(input_root)
    catalog = FilesystemArtifactCatalog(
        artifact_root / "catalog",
        artifact_root=artifact_root,
        input_roots=(input_root,),
    )
    pipeline = PreprocessingPipeline(
        catalog,
        CatalogPublisher(catalog, artifact_root),
        PreprocessingDependencies(
            registrar=registrar,
            inspector=MediaInspectionCapability(
                input_root=input_root,
                registrar=registrar,
                probe_runner=_ProbeRunner(
                    _probe_payload(include_audio=False, include_subtitles=False)
                ),
            ),
            shot_detector=_ShotDetector(artifact_root),
            embedded_subtitle_extractor=_UnexpectedCapability(),
            audio_extractor=_UnexpectedCapability(),
            speech_transcriber=_UnexpectedCapability(),
            chunker=TemporalChunkingCapability(artifact_root),
            transcript_indexer=TranscriptIndexingCapability(artifact_root),
        ),
    )

    result = pipeline.run("video.mp4")

    assert result.status is PipelineStatus.PARTIAL
    assert result.readiness.shots_ready
    assert result.readiness.timeline_ready
    assert not result.readiness.transcript_ready
    assert not result.readiness.sparse_search_ready
    assert result.video_id is not None
    chunks = catalog.load_manifest(result.video_id, CHUNKS_KEY, ChunkManifest)
    assert all(chunk.basis is ChunkBasis.SHOT_FALLBACK for chunk in chunks.chunks)
