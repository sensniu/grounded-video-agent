from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grounded_video_agent.agent.tools import (
    DeliveryPolicy,
    ExportEvidenceClipInput,
    GetVideoMetadataInput,
    InspectVisualContentInput,
    ReadScreenTextInput,
    ResolveTimelineContextInput,
    ScanVideoTimelineInput,
    SearchVideoTranscriptInput,
    ToolRuntimeContext,
    ToolStatus,
    VisualDetail,
)
from grounded_video_agent.agent.tools.dependencies import VideoToolDependencies
from grounded_video_agent.agent.tools.evidence_clip import ExportEvidenceClipTool
from grounded_video_agent.agent.tools.factory import build_video_tool_suite
from grounded_video_agent.agent.tools.media_operations import (
    FrameProvider,
    OCROperation,
    VisualOperation,
)
from grounded_video_agent.agent.tools.metadata import GetVideoMetadataTool
from grounded_video_agent.agent.tools.screen_text import ReadScreenTextTool
from grounded_video_agent.agent.tools.timeline_context import ResolveTimelineContextTool
from grounded_video_agent.agent.tools.timeline_scan import ScanVideoTimelineTool
from grounded_video_agent.agent.tools.transcript_search import SearchVideoTranscriptTool
from grounded_video_agent.agent.tools.visual_inspection import InspectVisualContentTool
from grounded_video_agent.capabilities._support import file_artifact, make_provenance
from grounded_video_agent.capabilities.indexing.transcript_index import (
    TranscriptIndexingCapability,
    TranscriptIndexingRequest,
)
from grounded_video_agent.capabilities.ocr.extraction import OCRExtractionCapability
from grounded_video_agent.capabilities.retrieval.hybrid_search import HybridRetrievalCapability
from grounded_video_agent.capabilities.retrieval.timeline_context import TimelineContextCapability
from grounded_video_agent.capabilities.retrieval.transcript_search import (
    TranscriptRetrievalCapability,
)
from grounded_video_agent.capabilities.visual.content_analysis import (
    VisualContentAnalysisCapability,
)
from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    CapabilityRequestContext,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    Chunk,
    ChunkBasis,
    ChunkManifest,
    ContainerInfo,
    FrameManifest,
    FrameRate,
    FrameRef,
    FrameSamplingStrategy,
    ManifestKind,
    ManifestRef,
    MediaProbe,
    ProducerInfo,
    Shot,
    ShotManifest,
    TimelineMapping,
    TimeRange,
    TranscriptManifest,
    TranscriptSegment,
    TranscriptSource,
    ValidationReport,
    VideoAsset,
    VideoClipArtifact,
    VideoStreamInfo,
)
from grounded_video_agent.infrastructure.ocr import (
    OCRDetection,
    OCRFrameResult,
    OCRModelInfo,
)
from grounded_video_agent.infrastructure.visual_model import (
    VisualModelInfo,
    VisualModelObservation,
    VisualModelResponse,
)
from grounded_video_agent.pipelines.preprocessing.keys import (
    CHUNKS_KEY,
    MEDIA_INSPECTION_KEY,
    SHOTS_KEY,
    SOURCE_KEY,
    SPARSE_INDEX_KEY,
    TRANSCRIPT_KEY,
)
from grounded_video_agent.workspace.catalog import (
    BasicMediaFlags,
    CatalogDocumentKind,
    CatalogDocumentRef,
    CatalogError,
    CatalogErrorCode,
    DocumentCodecRegistry,
    MediaInspectionDocument,
    MediaInspectionNextAction,
    PrimaryStreamSelection,
)

VIDEO_ID = "video-tools"


def _artifact(artifact_id: str, kind: ArtifactKind, uri: str) -> ArtifactRef:
    return ArtifactRef(artifact_id, kind, uri)


def _manifest_ref(
    manifest_id: str,
    kind: ManifestKind,
    count: int,
    path: Path,
) -> ManifestRef:
    return ManifestRef(
        manifest_id,
        kind,
        _artifact(f"{manifest_id}-artifact", ArtifactKind.MANIFEST, str(path)),
        VIDEO_ID,
        count,
    )


@dataclass(frozen=True)
class _Snapshot:
    video_asset: VideoAsset


@dataclass(frozen=True)
class _Entry:
    entry_id: str
    derivation_key: str | None = None
    dependency_entry_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Resolved:
    entry: _Entry


class _Catalog:
    def __init__(self, asset: VideoAsset, resources: dict[object, object]) -> None:
        self._asset = asset
        self._resources = resources
        self._registrations: dict[object, Any] = {}
        self._entries: dict[object, _Entry] = {}

    def get_snapshot(self, video_id: str) -> _Snapshot:
        assert video_id == VIDEO_ID
        return _Snapshot(self._asset)

    def resolve(self, video_id: str, key: object, **_: object) -> object:
        if key == SOURCE_KEY:
            return _Resolved(_Entry("entry-source"))
        if key in self._entries:
            return _Resolved(self._entries[key])
        self._get(video_id, key)
        return _Resolved(_Entry(f"entry-resource-{len(self._resources)}"))

    def load_manifest(
        self,
        video_id: str,
        key: object,
        expected_type: type[Any],
        **_: object,
    ) -> Any:
        value = self._get(video_id, key)
        assert isinstance(value, expected_type)
        return value

    def load_document(
        self,
        video_id: str,
        key: object,
        expected_type: type[Any],
        **_: object,
    ) -> Any:
        if key in self._registrations:
            reference = self._registrations[key].reference
            return DocumentCodecRegistry().load(reference.artifact.uri, expected_type)
        value = self._get(video_id, key)
        assert isinstance(value, expected_type)
        return value

    def register(self, video_id: str, registration: Any, **_: object) -> _Snapshot:
        assert video_id == VIDEO_ID
        self._registrations[registration.key] = registration
        self._entries[registration.key] = _Entry(
            f"entry-generated-{len(self._entries) + 1}",
            registration.derivation_key,
            registration.dependency_entry_ids,
        )
        return _Snapshot(self._asset)

    def find_reusable(
        self,
        video_id: str,
        key: object,
        derivation_key: str,
        dependency_entry_ids: tuple[str, ...],
    ) -> _Resolved | None:
        assert video_id == VIDEO_ID
        entry = self._entries.get(key)
        if (
            entry is not None
            and entry.derivation_key == derivation_key
            and entry.dependency_entry_ids == dependency_entry_ids
        ):
            return _Resolved(entry)
        return None

    def activate(
        self,
        video_id: str,
        key: object,
        entry_id: str,
        **_: object,
    ) -> _Snapshot:
        assert video_id == VIDEO_ID
        assert self._entries[key].entry_id == entry_id
        return _Snapshot(self._asset)

    def _get(self, video_id: str, key: object) -> object:
        assert video_id == VIDEO_ID
        try:
            return self._resources[key]
        except KeyError as error:
            raise CatalogError(CatalogErrorCode.RESOURCE_NOT_REGISTERED, str(key)) from error


class _FrameSampler:
    def __init__(self, root: Path) -> None:
        self._root = root
        self.calls = 0

    def execute(self, request: Any) -> CapabilityResult[FrameManifest]:
        self.calls += 1
        frames: list[FrameRef] = []
        if request.strategy is FrameSamplingStrategy.SHOT_KEYFRAME:
            timestamps = tuple(
                (overlap.start_ms + overlap.end_ms) // 2
                for shot in request.shots.shots
                for item in request.ranges
                if (overlap := shot.time_range.intersection(item)) is not None
            )
        else:
            timestamps = tuple((item.start_ms + item.end_ms) // 2 for item in request.ranges)
        for index, timestamp in enumerate(timestamps):
            image = self._root / f"{request.context.operation_id}-{index}.jpg"
            image.write_bytes(b"image")
            frames.append(
                FrameRef(
                    f"frame-{request.context.operation_id}-{index}",
                    VIDEO_ID,
                    timestamp,
                    _artifact(
                        f"frame-image-{request.context.operation_id}-{index}",
                        ArtifactKind.FRAME_IMAGE,
                        str(image),
                    ),
                )
            )
        manifest = FrameManifest(
            _manifest_ref(
                f"frames-{request.context.operation_id}",
                ManifestKind.FRAMES,
                len(frames),
                self._root / f"frames-{request.context.operation_id}.json",
            ),
            VIDEO_ID,
            request.strategy,
            request.ranges,
            tuple(frames),
            len(frames),
        )
        return CapabilityResult(
            CapabilityStatus.SUCCESS,
            manifest,
            CapabilityUsage(
                input_items=len(request.ranges),
                output_items=len(frames),
                decoded_frames=len(frames),
                returned_frames=len(frames),
            ),
        )


class _VisualBackend:
    def get_model_info(self) -> VisualModelInfo:
        return VisualModelInfo("fake-vlm", "1")

    def analyze(self, request: Any) -> VisualModelResponse:
        return VisualModelResponse(
            self.get_model_info(),
            tuple(
                VisualModelObservation(
                    target.target_id,
                    f"visible content for {target.target_id}",
                    target.frame_ids,
                    ("visible",),
                    0.9,
                )
                for target in request.targets
            ),
        )


class _OCRBackend:
    def get_model_info(self) -> OCRModelInfo:
        return OCRModelInfo("fake-ocr", "1", "test")

    def recognize(self, frames: tuple[Any, ...]) -> tuple[OCRFrameResult, ...]:
        return tuple(
            OCRFrameResult(
                frame.frame_id,
                100,
                100,
                (
                    OCRDetection(
                        "SALE 50%",
                        ((10, 10), (90, 10), (90, 30), (10, 30)),
                        0.95,
                    ),
                ),
            )
            for frame in frames
        )


class _ClipExporter:
    VERSION = "test-1"

    def __init__(self, root: Path) -> None:
        self._root = root
        self.calls = 0

    def execute(self, request: Any) -> CapabilityResult[VideoClipArtifact]:
        self.calls += 1
        provenance = make_provenance(
            "test-clip-export",
            self.VERSION,
            request,
            video_id=VIDEO_ID,
            source_artifact_ids=(request.video_asset.source.artifact_id,),
        )
        clip_id = f"clip-{request.context.operation_id}"
        path = self._root / "clips" / f"{clip_id}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"clip")
        artifact = file_artifact(
            path,
            artifact_id=f"{clip_id}-artifact",
            kind=ArtifactKind.VIDEO_CLIP,
            provenance=provenance,
        )
        clip = VideoClipArtifact(
            clip_id,
            VIDEO_ID,
            artifact,
            request.time_range,
            request.time_range,
            TimelineMapping(
                VIDEO_ID,
                request.time_range,
                clip_id,
                TimeRange(0, request.time_range.duration_ms),
            ),
            request.include_audio,
        )
        return CapabilityResult(
            CapabilityStatus.SUCCESS,
            clip,
            CapabilityUsage(
                input_items=1,
                output_items=1,
                processed_duration_ms=request.time_range.duration_ms,
            ),
            (artifact,),
            provenance=provenance,
        )


def _domain(tmp_path: Path) -> tuple[VideoAsset, dict[object, object]]:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    asset = VideoAsset(
        VIDEO_ID,
        _artifact("source", ArtifactKind.SOURCE_VIDEO, str(source)),
        "video.mp4",
    )
    chunks = ChunkManifest(
        _manifest_ref("chunks", ManifestKind.CHUNKS, 2, tmp_path / "chunks.json"),
        VIDEO_ID,
        (
            Chunk(
                "chunk-1",
                VIDEO_ID,
                TimeRange(0, 4_000),
                ("shot-1",),
                ("segment-1",),
                TimeRange(0, 5_000),
                "a red car arrives",
                ChunkBasis.TRANSCRIPT,
            ),
            Chunk(
                "chunk-2",
                VIDEO_ID,
                TimeRange(5_000, 9_000),
                ("shot-2",),
                ("segment-2",),
                TimeRange(5_000, 10_000),
                "a person opens a door",
                ChunkBasis.TRANSCRIPT,
            ),
        ),
    )
    shots = ShotManifest(
        _manifest_ref("shots", ManifestKind.SHOTS, 2, tmp_path / "shots.json"),
        VIDEO_ID,
        (
            Shot("shot-1", VIDEO_ID, TimeRange(0, 5_000)),
            Shot("shot-2", VIDEO_ID, TimeRange(5_000, 10_000)),
        ),
    )
    transcript = TranscriptManifest(
        _manifest_ref(
            "transcript", ManifestKind.TRANSCRIPT, 2, tmp_path / "transcript.json"
        ),
        VIDEO_ID,
        TranscriptSource.ASR,
        (
            TranscriptSegment(
                "segment-1",
                VIDEO_ID,
                TimeRange(0, 4_000),
                "A red car arrives",
                "a red car arrives",
                TranscriptSource.ASR,
                "en",
            ),
            TranscriptSegment(
                "segment-2",
                VIDEO_ID,
                TimeRange(5_000, 9_000),
                "A person opens a door",
                "a person opens a door",
                TranscriptSource.ASR,
                "en",
            ),
        ),
        "en",
    )
    indexed = TranscriptIndexingCapability(tmp_path).execute(
        TranscriptIndexingRequest(
            transcript,
            CapabilityRequestContext("index"),
            chunks,
        )
    )
    assert indexed.data is not None
    probe = MediaProbe(
        VIDEO_ID,
        ContainerInfo(("mp4",), duration_ms=10_000),
        (
            VideoStreamInfo(
                0,
                "h264",
                640,
                360,
                average_frame_rate=FrameRate(25, 1),
                is_default=True,
            ),
        ),
    )
    validation = ValidationReport(VIDEO_ID, ProducerInfo("test", "1"))
    document_ref = CatalogDocumentRef(
        "inspection",
        CatalogDocumentKind.MEDIA_INSPECTION,
        _artifact("inspection-artifact", ArtifactKind.METADATA, str(tmp_path / "media.json")),
        VIDEO_ID,
    )
    inspection = MediaInspectionDocument(
        document_ref,
        "inspection",
        asset,
        probe,
        validation,
        PrimaryStreamSelection.from_probe(probe),
        BasicMediaFlags.from_probe(probe),
        MediaInspectionNextAction.PROCEED,
    )
    return asset, {
        MEDIA_INSPECTION_KEY: inspection,
        CHUNKS_KEY: chunks,
        SHOTS_KEY: shots,
        TRANSCRIPT_KEY: transcript,
        SPARSE_INDEX_KEY: indexed.data,
    }


def _tools(
    tmp_path: Path,
) -> tuple[_Catalog, VideoToolDependencies, _FrameSampler, _ClipExporter]:
    asset, resources = _domain(tmp_path)
    catalog = _Catalog(asset, resources)
    frame_sampler = _FrameSampler(tmp_path)
    clip_exporter = _ClipExporter(tmp_path)
    dependencies = VideoToolDependencies(
        transcript_search=TranscriptRetrievalCapability(),
        hybrid_search=HybridRetrievalCapability(),
        timeline_context=TimelineContextCapability(),
        frame_sampler=frame_sampler,
        clip_exporter=clip_exporter,
        visual_analyzer=VisualContentAnalysisCapability(_VisualBackend(), tmp_path),
        ocr_extractor=OCRExtractionCapability(_OCRBackend(), tmp_path),
    )
    return catalog, dependencies, frame_sampler, clip_exporter


def test_metadata_search_dedup_and_context(tmp_path: Path) -> None:
    catalog, dependencies, _, _ = _tools(tmp_path)
    runtime = ToolRuntimeContext(VIDEO_ID, catalog)  # type: ignore[arg-type]

    metadata = GetVideoMetadataTool().execute(GetVideoMetadataInput(), runtime)
    assert metadata.status is ToolStatus.SUCCESS
    assert metadata.data is not None
    assert metadata.data.duration_ms == 10_000
    assert metadata.data.sparse_search_ready

    search_tool = SearchVideoTranscriptTool(dependencies)
    first = search_tool.execute(SearchVideoTranscriptInput("red car", top_k=1), runtime)
    second = search_tool.execute(SearchVideoTranscriptInput("red car", top_k=1), runtime)
    third = search_tool.execute(SearchVideoTranscriptInput("red car", top_k=1), runtime)
    assert first.data is not None and second.data is not None
    assert len(first.data.new_hits) == 1
    assert not second.data.new_hits
    assert second.data.reused_hits[0].candidate_id == first.data.new_hits[0].candidate_id
    assert second.progress.cache_hit
    assert second.progress.no_information_gain
    assert third.progress.no_information_gain
    assert runtime.requires_replan

    context = ResolveTimelineContextTool(dependencies).execute(
        ResolveTimelineContextInput(
            candidate_ids=(first.data.new_hits[0].candidate_id,),
            adjacent_chunks=1,
        ),
        runtime,
    )
    assert context.status is ToolStatus.SUCCESS
    assert context.data is not None
    assert context.data.chunk_ids == ("chunk-1", "chunk-2")
    assert len(context.data.subtitles) == 2
    bundle = runtime.build_evidence_bundle("What happened?")
    assert bundle.items
    assert first.data.new_hits[0].evidence_id in {
        item.evidence_id for item in bundle.items
    }


def test_visual_ocr_cache_and_timeline_scan(tmp_path: Path) -> None:
    catalog, dependencies, frame_sampler, _ = _tools(tmp_path)
    frames = FrameProvider(dependencies)
    visual_operation = VisualOperation(dependencies, frames)
    visual_tool = InspectVisualContentTool(visual_operation)
    ocr_tool = ReadScreenTextTool(OCROperation(dependencies, frames))
    runtime = ToolRuntimeContext(VIDEO_ID, catalog)  # type: ignore[arg-type]

    request = InspectVisualContentInput(
        "What is visible?",
        chunk_ids=("chunk-1",),
        detail=VisualDetail.STANDARD,
    )
    first = visual_tool.execute(request, runtime)
    second = visual_tool.execute(request, runtime)
    assert first.data is not None and first.data.observations
    assert second.data is not None and second.data.reused_analysis
    assert frame_sampler.calls == 1

    ocr = ocr_tool.execute(
        ReadScreenTextInput(frame_ids=(first.data.frames[0].frame_id,)),
        runtime,
    )
    assert ocr.data is not None
    assert ocr.data.spans[0].text == "SALE 50%"
    assert ocr.data.reused_frames
    assert frame_sampler.calls == 1

    scan_runtime = ToolRuntimeContext(VIDEO_ID, catalog)  # type: ignore[arg-type]
    scan = ScanVideoTimelineTool(visual_operation).execute(
        ScanVideoTimelineInput("Summarize the scene", max_windows=2),
        scan_runtime,
    )
    assert scan.data is not None
    assert len(scan.data.candidates) == 2
    assert scan.data.coverage_ratio == 1.0
    assert scan.data.exhausted


def test_evidence_clip_requires_authorization_and_reuses_catalog_entry(
    tmp_path: Path,
) -> None:
    catalog, dependencies, _, clip_exporter = _tools(tmp_path)
    runtime = ToolRuntimeContext(VIDEO_ID, catalog)  # type: ignore[arg-type]
    search = SearchVideoTranscriptTool(dependencies).execute(
        SearchVideoTranscriptInput("red car", top_k=1),
        runtime,
    )
    assert search.data is not None
    evidence_id = search.data.new_hits[0].evidence_id
    tool = ExportEvidenceClipTool(dependencies, tmp_path)
    request = ExportEvidenceClipInput(
        (evidence_id,),
        padding_before_ms=0,
        padding_after_ms=0,
    )

    unauthorized = tool.execute(request, runtime)
    assert unauthorized.status is ToolStatus.FAILED
    assert unauthorized.error is not None
    assert unauthorized.error.code == "CLIP_EXPORT_NOT_AUTHORIZED"

    runtime.delivery_policy = DeliveryPolicy(evidence_clip_requested=True)
    unverified = tool.execute(request, runtime)
    assert unverified.status is ToolStatus.FAILED
    assert unverified.error is not None
    assert unverified.error.code == "EVIDENCE_NOT_VERIFIED"

    runtime.delivery_policy = DeliveryPolicy(
        evidence_clip_requested=True,
        verified_evidence_ids=frozenset((evidence_id,)),
    )
    assert tool.is_available(runtime)
    over_limit = tool.execute(
        ExportEvidenceClipInput(
            (evidence_id,),
            padding_before_ms=6_000,
            padding_after_ms=0,
        ),
        runtime,
    )
    assert over_limit.status is ToolStatus.FAILED
    assert over_limit.error is not None
    assert over_limit.error.code == "EXPORT_LIMIT_EXCEEDED"
    first = tool.execute(request, runtime)
    second = tool.execute(request, runtime)

    assert first.status is ToolStatus.SUCCESS
    assert first.data is not None
    assert first.data.clips[0].requested_range == TimeRange(0, 4_000)
    assert not first.data.clips[0].includes_audio
    assert first.data.clips[0].artifact_id
    assert first.data.clips[0].catalog_entry_id
    assert str(tmp_path) not in first.to_json()
    assert second.data is not None
    assert second.data.clips[0].reused
    assert second.progress.cache_hit
    assert clip_exporter.calls == 1
    delivery = runtime.deliveries.get(first.data.clips[0].delivery_id)
    assert delivery.artifact.uri.endswith(".mp4")
    assert delivery.evidence_ids == (evidence_id,)


def test_suite_exposes_seven_framework_neutral_tool_schemas(tmp_path: Path) -> None:
    suite = build_video_tool_suite(artifact_root=tmp_path)
    assert tuple(spec.name for spec in suite.specs) == (
        "get_video_metadata",
        "search_video_transcript",
        "resolve_timeline_context",
        "inspect_visual_content",
        "read_screen_text",
        "scan_video_timeline",
        "export_evidence_clip",
    )
    assert all(spec.input_schema["type"] == "object" for spec in suite.specs)
    assert tuple(spec.name for spec in suite.available_specs) == (
        "get_video_metadata",
        "search_video_transcript",
        "resolve_timeline_context",
    )

    asset, resources = _domain(tmp_path)
    runtime = ToolRuntimeContext(VIDEO_ID, _Catalog(asset, resources))  # type: ignore[arg-type]
    runtime.delivery_policy = DeliveryPolicy(
        evidence_clip_requested=True,
        verified_evidence_ids=frozenset(("evidence-ready",)),
    )
    assert "export_evidence_clip" in {
        spec.name for spec in suite.available_specs_for(runtime)
    }
    invalid = suite.invoke("search_video_transcript", {"query": ""}, runtime)
    assert invalid.status is ToolStatus.FAILED
    assert invalid.error is not None
    assert invalid.error.code == "INVALID_TOOL_INPUT"

    limited_runtime = ToolRuntimeContext(
        VIDEO_ID,
        _Catalog(asset, resources),  # type: ignore[arg-type]
        max_tool_calls=1,
    )
    first = suite.invoke("get_video_metadata", {}, limited_runtime)
    second = suite.invoke("get_video_metadata", {}, limited_runtime)
    assert first.status is ToolStatus.SUCCESS
    assert second.status is ToolStatus.FAILED
    assert second.error is not None
    assert second.error.code == "TOOL_BUDGET_EXHAUSTED"
