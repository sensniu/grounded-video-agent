from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from grounded_video_agent.capabilities.indexing.transcript_index import (
    TranscriptIndexingCapability,
    TranscriptIndexingRequest,
)
from grounded_video_agent.capabilities.indexing.visual_index import (
    VisualIndexingCapability,
    VisualIndexingRequest,
)
from grounded_video_agent.capabilities.retrieval.timeline_context import (
    TimelineContextCapability,
    TimelineContextRequest,
)
from grounded_video_agent.capabilities.retrieval.transcript_search import (
    TranscriptRetrievalCapability,
    TranscriptRetrievalRequest,
)
from grounded_video_agent.capabilities.retrieval.visual_search import (
    VisualRetrievalCapability,
    VisualRetrievalRequest,
)
from grounded_video_agent.capabilities.visual.content_analysis import (
    VisualContentAnalysisCapability,
    VisualContentAnalysisRequest,
)
from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    CapabilityRequestContext,
    CapabilityStatus,
    Chunk,
    ChunkBasis,
    ChunkManifest,
    FrameManifest,
    FrameRef,
    FrameSamplingStrategy,
    ManifestKind,
    ManifestRef,
    Shot,
    ShotManifest,
    TimeRange,
    TranscriptManifest,
    TranscriptSegment,
    TranscriptSource,
    VisualAnalysisTarget,
    VisualDescriptionMode,
)
from grounded_video_agent.infrastructure.visual_model import (
    FastAPIVisualModelClient,
    VisualModelFrame,
    VisualModelInfo,
    VisualModelObservation,
    VisualModelRequest,
    VisualModelResponse,
    VisualModelTarget,
)
from grounded_video_agent.services.visual_model_api import create_app

VIDEO_ID = "video-search"


def _artifact(artifact_id: str, kind: ArtifactKind, uri: str) -> ArtifactRef:
    return ArtifactRef(artifact_id, kind, uri)


def _manifest_ref(
    manifest_id: str,
    kind: ManifestKind,
    item_count: int,
    uri: str,
) -> ManifestRef:
    return ManifestRef(
        manifest_id,
        kind,
        _artifact(f"{manifest_id}-artifact", ArtifactKind.MANIFEST, uri),
        VIDEO_ID,
        item_count,
    )


def _transcript() -> TranscriptManifest:
    segments = (
        TranscriptSegment(
            "segment-car",
            VIDEO_ID,
            TimeRange(0, 4_000),
            "A red car enters",
            "a red car enters",
            TranscriptSource.ASR,
            language="en",
        ),
        TranscriptSegment(
            "segment-door",
            VIDEO_ID,
            TimeRange(4_000, 8_000),
            "A person opens the door",
            "a person opens the door",
            TranscriptSource.ASR,
            language="en",
        ),
    )
    return TranscriptManifest(
        _manifest_ref("transcript", ManifestKind.TRANSCRIPT, 2, "/tmp/transcript.json"),
        VIDEO_ID,
        TranscriptSource.ASR,
        segments,
        "en",
    )


def _chunks() -> ChunkManifest:
    chunks = (
        Chunk(
            "chunk-1",
            VIDEO_ID,
            TimeRange(0, 4_000),
            ("shot-1",),
            ("segment-car",),
            TimeRange(0, 5_000),
            "a red car enters",
            ChunkBasis.TRANSCRIPT,
        ),
        Chunk(
            "chunk-2",
            VIDEO_ID,
            TimeRange(4_000, 8_000),
            ("shot-2",),
            ("segment-door",),
            TimeRange(4_000, 10_000),
            "a person opens the door",
            ChunkBasis.TRANSCRIPT,
        ),
    )
    return ChunkManifest(
        _manifest_ref("chunks", ManifestKind.CHUNKS, 2, "/tmp/chunks.json"),
        VIDEO_ID,
        chunks,
    )


def _shots() -> ShotManifest:
    shots = (
        Shot("shot-1", VIDEO_ID, TimeRange(0, 5_000)),
        Shot("shot-2", VIDEO_ID, TimeRange(5_000, 10_000)),
    )
    return ShotManifest(
        _manifest_ref("shots", ManifestKind.SHOTS, 2, "/tmp/shots.json"),
        VIDEO_ID,
        shots,
    )


def _frames(tmp_path: Path) -> FrameManifest:
    first = tmp_path / "frame-1.jpg"
    second = tmp_path / "frame-2.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    frames = (
        FrameRef(
            "frame-1",
            VIDEO_ID,
            1_000,
            _artifact("frame-1-image", ArtifactKind.FRAME_IMAGE, str(first)),
        ),
        FrameRef(
            "frame-2",
            VIDEO_ID,
            6_000,
            _artifact("frame-2-image", ArtifactKind.FRAME_IMAGE, str(second)),
        ),
    )
    return FrameManifest(
        _manifest_ref("frames", ManifestKind.FRAMES, 2, "/tmp/frames.json"),
        VIDEO_ID,
        FrameSamplingStrategy.SHOT_KEYFRAME,
        (TimeRange(0, 10_000),),
        frames,
        decoded_frames=2,
    )


class _FakeVisualBackend:
    def get_model_info(self) -> VisualModelInfo:
        return VisualModelInfo("fake-vlm", "1")

    def analyze(self, request: VisualModelRequest) -> VisualModelResponse:
        observations = tuple(
            VisualModelObservation(
                target.target_id,
                "a red car" if target.target_id == "target-1" else "a person opens a door",
                target.frame_ids,
                ("car",) if target.target_id == "target-1" else ("person", "door"),
                0.9,
            )
            for target in request.targets
        )
        return VisualModelResponse(self.get_model_info(), observations)


def test_transcript_index_search_and_timeline_context(tmp_path: Path) -> None:
    transcript = _transcript()
    chunks = _chunks()
    index = TranscriptIndexingCapability(tmp_path).execute(
        TranscriptIndexingRequest(
            transcript,
            CapabilityRequestContext("transcript-index"),
            chunks,
        )
    )
    assert index.status is CapabilityStatus.SUCCESS
    assert index.data is not None

    retrieval = TranscriptRetrievalCapability().execute(
        TranscriptRetrievalRequest(
            "opens door",
            index.data,
            CapabilityRequestContext("transcript-search"),
            chunk_ids=("chunk-2",),
            language="en",
        )
    )
    assert retrieval.status is CapabilityStatus.SUCCESS
    assert retrieval.data is not None
    assert len(retrieval.data.hits) == 1
    assert retrieval.data.hits[0].item.source_ids == (
        "chunk-2",
        "segment-door",
        "shot-2",
    )

    timeline = TimelineContextCapability().execute(
        TimelineContextRequest(
            VIDEO_ID,
            chunks,
            _shots(),
            transcript,
            CapabilityRequestContext("timeline"),
            retrieval=retrieval.data,
            adjacent_chunks=1,
        )
    )
    assert timeline.status is CapabilityStatus.SUCCESS
    assert timeline.data is not None
    assert tuple(chunk.chunk_id for chunk in timeline.data.chunks) == ("chunk-1", "chunk-2")
    assert timeline.data.resolved_ranges == (TimeRange(0, 10_000),)


def test_visual_analysis_index_and_search(tmp_path: Path) -> None:
    frames = _frames(tmp_path)
    targets = (
        VisualAnalysisTarget("target-1", VIDEO_ID, TimeRange(0, 5_000), ("frame-1",)),
        VisualAnalysisTarget("target-2", VIDEO_ID, TimeRange(5_000, 10_000), ("frame-2",)),
    )
    analyzed = VisualContentAnalysisCapability(_FakeVisualBackend(), tmp_path).execute(
        VisualContentAnalysisRequest(
            frames,
            targets,
            VisualDescriptionMode.GENERIC,
            CapabilityRequestContext("visual-analysis"),
        )
    )
    assert analyzed.status is CapabilityStatus.SUCCESS
    assert analyzed.data is not None

    indexed = VisualIndexingCapability(tmp_path).execute(
        VisualIndexingRequest(
            analyzed.data,
            CapabilityRequestContext("visual-index"),
            _chunks(),
            _shots(),
        )
    )
    assert indexed.status is CapabilityStatus.SUCCESS
    assert indexed.data is not None

    retrieval = VisualRetrievalCapability().execute(
        VisualRetrievalRequest(
            "red car",
            indexed.data,
            CapabilityRequestContext("visual-search"),
            related_ids=("chunk-1",),
            frames=frames,
        )
    )
    assert retrieval.status is CapabilityStatus.SUCCESS
    assert retrieval.data is not None
    assert len(retrieval.data.hits) == 1
    assert retrieval.data.hits[0].item.text == "a red car"
    assert retrieval.data.hits[0].item.artifacts == (frames.frames[0].image,)


@pytest.mark.asyncio
async def test_visual_model_fastapi_service_validates_paths(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"image")
    app = create_app(_FakeVisualBackend(), allowed_roots=(tmp_path,))
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")

    health = await client.get("/health")
    response = await client.post(
        "/v1/analyze",
        json={
            "operation_id": "api-test",
            "mode": "generic",
            "frames": [
                {"frame_id": "frame-1", "uri": str(frame_path), "timestamp_ms": 1000}
            ],
            "targets": [
                {
                    "target_id": "target-1",
                    "start_ms": 0,
                    "end_ms": 2000,
                    "frame_ids": ["frame-1"],
                }
            ],
        },
    )
    denied = await client.post(
        "/v1/analyze",
        json={
            "operation_id": "api-denied",
            "mode": "generic",
            "frames": [
                {"frame_id": "outside", "uri": "/etc/hosts", "timestamp_ms": 1000}
            ],
            "targets": [
                {
                    "target_id": "target-1",
                    "start_ms": 0,
                    "end_ms": 2000,
                    "frame_ids": ["outside"],
                }
            ],
        },
    )
    await client.aclose()

    assert health.json()["model_name"] == "fake-vlm"
    assert response.status_code == 200
    assert response.json()["observations"][0]["text"] == "a red car"
    assert denied.status_code == 400


def test_fastapi_visual_model_client_maps_transport_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"model_name": "fake-vlm", "model_version": "1"})
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": {"model_name": "fake-vlm", "model_version": "1"},
                "observations": [
                    {
                        "target_id": payload["targets"][0]["target_id"],
                        "text": "a red car",
                        "frame_ids": ["frame"],
                        "tags": ["car"],
                        "confidence": 0.9,
                    }
                ],
                "warnings": [],
            },
        )

    client = FastAPIVisualModelClient(
        "http://visual-model",
        transport=httpx.MockTransport(handler),
    )
    result = client.analyze(
        VisualModelRequest(
            "client-test",
            "generic",
            (VisualModelFrame("frame", "/tmp/frame.jpg", 0),),
            (VisualModelTarget("target", 0, 1_000, ("frame",)),),
        )
    )

    assert client.get_model_info().model_name == "fake-vlm"
    assert result.observations[0].tags == ("car",)
