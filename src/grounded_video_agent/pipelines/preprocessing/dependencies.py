from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from grounded_video_agent.capabilities.audio.extraction import AudioExtractionRequest
from grounded_video_agent.capabilities.indexing.dense_index import DenseIndexingRequest
from grounded_video_agent.capabilities.indexing.transcript_index import TranscriptIndexingRequest
from grounded_video_agent.capabilities.media_inspection import VideoInspectionResult
from grounded_video_agent.capabilities.subtitles.embedded_extraction import (
    EmbeddedSubtitleExtractionRequest,
)
from grounded_video_agent.capabilities.subtitles.speech_transcription import (
    SpeechTranscriptionRequest,
)
from grounded_video_agent.capabilities.temporal.chunking import TemporalChunkingRequest
from grounded_video_agent.capabilities.temporal.shot_detection import ShotDetectionRequest
from grounded_video_agent.domain import (
    AudioArtifact,
    CapabilityResult,
    ChunkManifest,
    IndexManifest,
    ShotManifest,
    TranscriptManifest,
)
from grounded_video_agent.infrastructure.embeddings import EmbeddingModelInfo
from grounded_video_agent.input import VideoRegistrationResult


class VideoRegistrarProtocol(Protocol):
    def register(self, filename: str) -> VideoRegistrationResult: ...


class MediaInspectorProtocol(Protocol):
    def inspect_registered(
        self,
        registration: VideoRegistrationResult,
    ) -> VideoInspectionResult: ...


class ShotDetectorProtocol(Protocol):
    def execute(self, request: ShotDetectionRequest) -> CapabilityResult[ShotManifest]: ...


class EmbeddedSubtitleExtractorProtocol(Protocol):
    def execute(
        self,
        request: EmbeddedSubtitleExtractionRequest,
    ) -> CapabilityResult[TranscriptManifest]: ...


class AudioExtractorProtocol(Protocol):
    def execute(self, request: AudioExtractionRequest) -> CapabilityResult[AudioArtifact]: ...


class SpeechTranscriberProtocol(Protocol):
    def execute(
        self,
        request: SpeechTranscriptionRequest,
    ) -> CapabilityResult[TranscriptManifest]: ...


class ChunkerProtocol(Protocol):
    def execute(self, request: TemporalChunkingRequest) -> CapabilityResult[ChunkManifest]: ...


class TranscriptIndexerProtocol(Protocol):
    def execute(self, request: TranscriptIndexingRequest) -> CapabilityResult[IndexManifest]: ...


class DenseIndexerProtocol(Protocol):
    def get_model_info(self) -> EmbeddingModelInfo: ...

    def execute(self, request: DenseIndexingRequest) -> CapabilityResult[IndexManifest]: ...


@dataclass(frozen=True, slots=True)
class PreprocessingDependencies:
    registrar: VideoRegistrarProtocol
    inspector: MediaInspectorProtocol
    shot_detector: ShotDetectorProtocol
    embedded_subtitle_extractor: EmbeddedSubtitleExtractorProtocol
    audio_extractor: AudioExtractorProtocol
    speech_transcriber: SpeechTranscriberProtocol
    chunker: ChunkerProtocol
    transcript_indexer: TranscriptIndexerProtocol
    dense_indexer: DenseIndexerProtocol | None = None
