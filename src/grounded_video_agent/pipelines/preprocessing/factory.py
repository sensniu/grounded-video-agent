from __future__ import annotations

from pathlib import Path

from grounded_video_agent.capabilities.audio.extraction import AudioExtractionCapability
from grounded_video_agent.capabilities.indexing.dense_index import DenseIndexingCapability
from grounded_video_agent.capabilities.indexing.transcript_index import (
    TranscriptIndexingCapability,
)
from grounded_video_agent.capabilities.media_inspection import MediaInspectionCapability
from grounded_video_agent.capabilities.subtitles.embedded_extraction import (
    EmbeddedSubtitleExtractionCapability,
)
from grounded_video_agent.capabilities.subtitles.speech_transcription import (
    SpeechTranscriptionCapability,
)
from grounded_video_agent.capabilities.temporal.chunking import TemporalChunkingCapability
from grounded_video_agent.capabilities.temporal.shot_detection import ShotDetectionCapability
from grounded_video_agent.infrastructure.embeddings import TextEmbeddingBackend
from grounded_video_agent.input import VideoRegistrar
from grounded_video_agent.pipelines.preprocessing.config import (
    DenseIndexPolicy,
    PreprocessingConfig,
)
from grounded_video_agent.pipelines.preprocessing.dependencies import (
    PreprocessingDependencies,
)
from grounded_video_agent.pipelines.preprocessing.pipeline import PreprocessingPipeline
from grounded_video_agent.pipelines.preprocessing.publication import CatalogPublisher
from grounded_video_agent.workspace.catalog import FilesystemArtifactCatalog


def build_local_preprocessing_pipeline(
    *,
    input_root: str | Path = "analyzed_video",
    artifact_root: str | Path = "artifacts",
    catalog_root: str | Path | None = None,
    config: PreprocessingConfig | None = None,
    embedding_backend: TextEmbeddingBackend | None = None,
) -> PreprocessingPipeline:
    resolved_config = config or PreprocessingConfig()
    if (
        resolved_config.dense_index_policy is DenseIndexPolicy.REQUIRED
        and embedding_backend is None
    ):
        raise ValueError("a dense embedding backend is required by preprocessing config")
    artifact_path = Path(artifact_root).resolve()
    registrar = VideoRegistrar(input_root)
    catalog = FilesystemArtifactCatalog(
        catalog_root or artifact_path / "catalog",
        artifact_root=artifact_path,
        input_roots=(input_root,),
    )
    dense_indexer = (
        DenseIndexingCapability(embedding_backend, artifact_path)
        if embedding_backend is not None
        else None
    )
    dependencies = PreprocessingDependencies(
        registrar=registrar,
        inspector=MediaInspectionCapability(input_root=input_root, registrar=registrar),
        shot_detector=ShotDetectionCapability(artifact_path),
        embedded_subtitle_extractor=EmbeddedSubtitleExtractionCapability(artifact_path),
        audio_extractor=AudioExtractionCapability(artifact_path),
        speech_transcriber=SpeechTranscriptionCapability(artifact_path),
        chunker=TemporalChunkingCapability(artifact_path),
        transcript_indexer=TranscriptIndexingCapability(artifact_path),
        dense_indexer=dense_indexer,
    )
    return PreprocessingPipeline(
        catalog,
        CatalogPublisher(catalog, artifact_path),
        dependencies,
        resolved_config,
    )
