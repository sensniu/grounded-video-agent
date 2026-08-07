"""Capability dependencies composed by the agent-facing tool layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from grounded_video_agent.capabilities.ocr.extraction import OCRExtractionRequest
from grounded_video_agent.capabilities.retrieval.dense_search import DenseRetrievalRequest
from grounded_video_agent.capabilities.retrieval.hybrid_search import HybridRetrievalRequest
from grounded_video_agent.capabilities.retrieval.timeline_context import TimelineContextRequest
from grounded_video_agent.capabilities.retrieval.transcript_search import (
    TranscriptRetrievalRequest,
)
from grounded_video_agent.capabilities.visual.clip_export import ClipExportRequest
from grounded_video_agent.capabilities.visual.content_analysis import (
    VisualContentAnalysisRequest,
)
from grounded_video_agent.capabilities.visual.frame_sampling import FrameSamplingRequest
from grounded_video_agent.domain import (
    CapabilityResult,
    FrameManifest,
    OCRManifest,
    RetrievalResult,
    TimelineContext,
    VideoClipArtifact,
    VisualDescriptionManifest,
)


class TranscriptSearchProtocol(Protocol):
    def execute(
        self, request: TranscriptRetrievalRequest
    ) -> CapabilityResult[RetrievalResult]: ...


class DenseSearchProtocol(Protocol):
    def execute(self, request: DenseRetrievalRequest) -> CapabilityResult[RetrievalResult]: ...


class HybridSearchProtocol(Protocol):
    def execute(self, request: HybridRetrievalRequest) -> CapabilityResult[RetrievalResult]: ...


class TimelineContextProtocol(Protocol):
    def execute(self, request: TimelineContextRequest) -> CapabilityResult[TimelineContext]: ...


class FrameSamplerProtocol(Protocol):
    def execute(self, request: FrameSamplingRequest) -> CapabilityResult[FrameManifest]: ...


class VisualAnalyzerProtocol(Protocol):
    def execute(
        self, request: VisualContentAnalysisRequest
    ) -> CapabilityResult[VisualDescriptionManifest]: ...


class OCRExtractorProtocol(Protocol):
    def execute(self, request: OCRExtractionRequest) -> CapabilityResult[OCRManifest]: ...


class ClipExporterProtocol(Protocol):
    VERSION: str

    def execute(
        self, request: ClipExportRequest
    ) -> CapabilityResult[VideoClipArtifact]: ...


@dataclass(frozen=True, slots=True)
class VideoToolDependencies:
    transcript_search: TranscriptSearchProtocol
    hybrid_search: HybridSearchProtocol
    timeline_context: TimelineContextProtocol
    frame_sampler: FrameSamplerProtocol
    clip_exporter: ClipExporterProtocol
    dense_search: DenseSearchProtocol | None = None
    visual_analyzer: VisualAnalyzerProtocol | None = None
    ocr_extractor: OCRExtractorProtocol | None = None
