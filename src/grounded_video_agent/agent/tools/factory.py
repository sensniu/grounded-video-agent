from __future__ import annotations

from pathlib import Path

from grounded_video_agent.agent.tools.dependencies import VideoToolDependencies
from grounded_video_agent.agent.tools.evidence_clip import ExportEvidenceClipTool
from grounded_video_agent.agent.tools.media_operations import (
    FrameProvider,
    OCROperation,
    VisualOperation,
)
from grounded_video_agent.agent.tools.metadata import GetVideoMetadataTool
from grounded_video_agent.agent.tools.screen_text import ReadScreenTextTool
from grounded_video_agent.agent.tools.suite import VideoToolSuite
from grounded_video_agent.agent.tools.timeline_context import ResolveTimelineContextTool
from grounded_video_agent.agent.tools.timeline_scan import ScanVideoTimelineTool
from grounded_video_agent.agent.tools.transcript_search import SearchVideoTranscriptTool
from grounded_video_agent.agent.tools.visual_inspection import InspectVisualContentTool
from grounded_video_agent.capabilities.ocr.extraction import OCRExtractionCapability
from grounded_video_agent.capabilities.retrieval.dense_search import DenseRetrievalCapability
from grounded_video_agent.capabilities.retrieval.hybrid_search import HybridRetrievalCapability
from grounded_video_agent.capabilities.retrieval.timeline_context import TimelineContextCapability
from grounded_video_agent.capabilities.retrieval.transcript_search import (
    TranscriptRetrievalCapability,
)
from grounded_video_agent.capabilities.visual.clip_export import ClipExportCapability
from grounded_video_agent.capabilities.visual.content_analysis import (
    VisualContentAnalysisCapability,
)
from grounded_video_agent.capabilities.visual.frame_sampling import FrameSamplingCapability
from grounded_video_agent.infrastructure.embeddings import TextEmbeddingBackend
from grounded_video_agent.infrastructure.ocr import OCRBackend
from grounded_video_agent.infrastructure.visual_model import VisualModelBackend


def build_video_tool_suite(
    *,
    artifact_root: str | Path = "artifacts",
    embedding_backend: TextEmbeddingBackend | None = None,
    visual_backend: VisualModelBackend | None = None,
    ocr_backend: OCRBackend | None = None,
) -> VideoToolSuite:
    dependencies = VideoToolDependencies(
        transcript_search=TranscriptRetrievalCapability(),
        dense_search=(
            DenseRetrievalCapability(embedding_backend)
            if embedding_backend is not None
            else None
        ),
        hybrid_search=HybridRetrievalCapability(),
        timeline_context=TimelineContextCapability(),
        frame_sampler=FrameSamplingCapability(artifact_root),
        clip_exporter=ClipExportCapability(artifact_root),
        visual_analyzer=(
            VisualContentAnalysisCapability(visual_backend, artifact_root)
            if visual_backend is not None
            else None
        ),
        ocr_extractor=(
            OCRExtractionCapability(ocr_backend, artifact_root)
            if ocr_backend is not None
            else None
        ),
    )
    frames = FrameProvider(dependencies)
    visual = VisualOperation(dependencies, frames)
    ocr = OCROperation(dependencies, frames)
    return VideoToolSuite(
        (
            GetVideoMetadataTool(),
            SearchVideoTranscriptTool(dependencies),
            ResolveTimelineContextTool(dependencies),
            InspectVisualContentTool(visual, enabled=visual_backend is not None),
            ReadScreenTextTool(ocr, enabled=ocr_backend is not None),
            ScanVideoTimelineTool(visual, enabled=visual_backend is not None),
            ExportEvidenceClipTool(dependencies, artifact_root),
        )
    )
