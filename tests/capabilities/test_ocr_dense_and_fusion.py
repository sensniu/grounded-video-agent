from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grounded_video_agent.capabilities.indexing.dense_index import (
    DenseIndexingCapability,
    DenseIndexingRequest,
)
from grounded_video_agent.capabilities.indexing.ocr_index import (
    OCRIndexingCapability,
    OCRIndexingRequest,
)
from grounded_video_agent.capabilities.indexing.transcript_index import (
    TranscriptIndexingCapability,
    TranscriptIndexingRequest,
)
from grounded_video_agent.capabilities.ocr import OCRExtractionCapability, OCRExtractionRequest
from grounded_video_agent.capabilities.retrieval.candidate_fusion import (
    CandidateFusionCapability,
    CandidateFusionRequest,
)
from grounded_video_agent.capabilities.retrieval.dense_search import (
    DenseRetrievalCapability,
    DenseRetrievalRequest,
)
from grounded_video_agent.capabilities.retrieval.hybrid_search import (
    HybridRetrievalCapability,
    HybridRetrievalRequest,
)
from grounded_video_agent.capabilities.retrieval.ocr_search import (
    OCRRetrievalCapability,
    OCRRetrievalRequest,
)
from grounded_video_agent.capabilities.retrieval.transcript_search import (
    TranscriptRetrievalCapability,
    TranscriptRetrievalRequest,
)
from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    CapabilityRequestContext,
    CapabilityStatus,
    Chunk,
    ChunkBasis,
    ChunkManifest,
    EvidenceItem,
    EvidenceModality,
    FrameManifest,
    FrameRef,
    FrameSamplingStrategy,
    ManifestKind,
    ManifestRef,
    RetrievalHit,
    RetrievalResult,
    Shot,
    ShotManifest,
    TimeRange,
    TranscriptManifest,
    TranscriptSegment,
    TranscriptSource,
)
from grounded_video_agent.infrastructure.embeddings import EmbeddingModelInfo
from grounded_video_agent.infrastructure.ocr import (
    OCRDetection,
    OCRFrameInput,
    OCRFrameResult,
    OCRModelInfo,
    RapidOCRBackend,
)

VIDEO_ID = "video-closed-loops"


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


def _frames(tmp_path: Path) -> FrameManifest:
    frames: list[FrameRef] = []
    for index, timestamp in enumerate((1_000, 2_000), start=1):
        path = tmp_path / f"frame-{index}.jpg"
        path.write_bytes(f"frame-{index}".encode())
        frames.append(
            FrameRef(
                f"frame-{index}",
                VIDEO_ID,
                timestamp,
                _artifact(f"frame-{index}-image", ArtifactKind.FRAME_IMAGE, str(path)),
            )
        )
    return FrameManifest(
        _manifest_ref("frames", ManifestKind.FRAMES, 2, str(tmp_path / "frames.json")),
        VIDEO_ID,
        FrameSamplingStrategy.FIXED_FPS,
        (TimeRange(0, 3_000),),
        tuple(frames),
        decoded_frames=2,
    )


def _chunks() -> ChunkManifest:
    chunks = (
        Chunk(
            "chunk-1",
            VIDEO_ID,
            TimeRange(0, 5_000),
            transcript_segment_ids=("segment-car",),
            inspection_range=TimeRange(0, 5_000),
            text="a red car arrives",
            basis=ChunkBasis.TRANSCRIPT,
        ),
        Chunk(
            "chunk-2",
            VIDEO_ID,
            TimeRange(15_000, 25_000),
            transcript_segment_ids=("segment-door",),
            inspection_range=TimeRange(15_000, 25_000),
            text="someone opens a door",
            basis=ChunkBasis.TRANSCRIPT,
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
        Shot("shot-2", VIDEO_ID, TimeRange(15_000, 25_000)),
    )
    return ShotManifest(
        _manifest_ref("shots", ManifestKind.SHOTS, 2, "/tmp/shots.json"),
        VIDEO_ID,
        shots,
    )


class _FakeOCRBackend:
    def get_model_info(self) -> OCRModelInfo:
        return OCRModelInfo("fake-ocr", "1", "tests")

    def recognize(self, frames: tuple[OCRFrameInput, ...]) -> tuple[OCRFrameResult, ...]:
        return tuple(
            OCRFrameResult(
                frame.frame_id,
                200,
                100,
                (
                    OCRDetection(
                        "正在直播",
                        ((20, 70), (180, 70), (180, 90), (20, 90)),
                        0.95,
                    ),
                ),
            )
            for frame in frames
        )


class _FakeEmbeddingBackend:
    def __init__(self, space: str = "fake:semantic") -> None:
        self._space = space

    def get_model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo("fake-embedding", "1", self._space, 3)

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._vector(text) for text in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        lowered = text.lower()
        if any(token in lowered for token in ("car", "vehicle", "汽车")):
            return (1.0, 0.0, 0.0)
        if any(token in lowered for token in ("door", "门")):
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)


def _transcript() -> TranscriptManifest:
    segments = (
        TranscriptSegment(
            "segment-car",
            VIDEO_ID,
            TimeRange(0, 4_000),
            "A red car arrives",
            "a red car arrives",
            TranscriptSource.ASR,
            language="en",
        ),
        TranscriptSegment(
            "segment-door",
            VIDEO_ID,
            TimeRange(15_000, 20_000),
            "Someone opens a door",
            "someone opens a door",
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


def test_ocr_extraction_merge_index_and_search(tmp_path: Path) -> None:
    frames = _frames(tmp_path)
    extracted = OCRExtractionCapability(_FakeOCRBackend(), tmp_path).execute(
        OCRExtractionRequest(
            frames,
            CapabilityRequestContext("ocr-extract"),
            language="zh",
        )
    )
    assert extracted.status is CapabilityStatus.SUCCESS
    assert extracted.data is not None
    assert len(extracted.data.observations) == 2
    assert len(extracted.data.spans) == 1
    assert extracted.data.spans[0].time_range == TimeRange(1_000, 2_001)

    indexed = OCRIndexingCapability(tmp_path).execute(
        OCRIndexingRequest(
            extracted.data,
            CapabilityRequestContext("ocr-index"),
            _chunks(),
        )
    )
    assert indexed.status is CapabilityStatus.SUCCESS
    assert indexed.data is not None
    searched = OCRRetrievalCapability().execute(
        OCRRetrievalRequest(
            "直播",
            indexed.data,
            CapabilityRequestContext("ocr-search"),
            related_ids=("chunk-1",),
            frames=frames,
        )
    )
    assert searched.status is CapabilityStatus.SUCCESS
    assert searched.data is not None
    assert len(searched.data.hits) == 1
    assert searched.data.hits[0].item.modality is EvidenceModality.OCR
    assert len(searched.data.hits[0].item.artifacts) == 2


def test_dense_index_search_and_hybrid_fusion(tmp_path: Path) -> None:
    transcript = _transcript()
    backend = _FakeEmbeddingBackend()
    dense_index = DenseIndexingCapability(backend, tmp_path).execute(
        DenseIndexingRequest(
            transcript,
            CapabilityRequestContext("dense-index"),
            _chunks(),
        )
    )
    assert dense_index.status is CapabilityStatus.SUCCESS
    assert dense_index.data is not None
    assert dense_index.data.embedding_manifest_id is not None
    dense = DenseRetrievalCapability(backend).execute(
        DenseRetrievalRequest(
            "vehicle",
            dense_index.data,
            CapabilityRequestContext("dense-search"),
        )
    )
    assert dense.status is CapabilityStatus.SUCCESS
    assert dense.data is not None
    assert dense.data.hits[0].item.source_ids[0] == "chunk-1"

    sparse_index = TranscriptIndexingCapability(tmp_path).execute(
        TranscriptIndexingRequest(
            transcript,
            CapabilityRequestContext("sparse-index"),
            _chunks(),
        )
    )
    assert sparse_index.data is not None
    sparse = TranscriptRetrievalCapability().execute(
        TranscriptRetrievalRequest(
            "vehicle",
            sparse_index.data,
            CapabilityRequestContext("sparse-search"),
        )
    )
    assert sparse.data is not None
    hybrid = HybridRetrievalCapability().execute(
        HybridRetrievalRequest(
            VIDEO_ID,
            "vehicle",
            sparse.data,
            dense.data,
            CapabilityRequestContext("hybrid-search"),
        )
    )
    assert hybrid.status is CapabilityStatus.SUCCESS
    assert hybrid.data is not None
    assert hybrid.data.hits[0].item.source_ids[0] == "chunk-1"
    assert hybrid.data.hits[0].item.scores[-1].name == "hybrid_rrf"


def test_dense_retrieval_rejects_embedding_space_mismatch(tmp_path: Path) -> None:
    built = DenseIndexingCapability(_FakeEmbeddingBackend(), tmp_path).execute(
        DenseIndexingRequest(_transcript(), CapabilityRequestContext("dense-space"))
    )
    assert built.data is not None
    result = DenseRetrievalCapability(_FakeEmbeddingBackend("fake:other")).execute(
        DenseRetrievalRequest(
            "vehicle",
            built.data,
            CapabilityRequestContext("dense-space-search"),
        )
    )
    assert result.status is CapabilityStatus.FAILED
    assert result.error is not None
    assert "embedding spaces" in result.error.message


def _retrieval(
    query: str,
    modality: EvidenceModality,
    source_id: str,
    time_range: TimeRange,
    text: str,
) -> RetrievalResult:
    return RetrievalResult(
        query,
        (
            RetrievalHit(
                1,
                EvidenceItem(
                    f"{source_id}-evidence",
                    VIDEO_ID,
                    time_range,
                    modality,
                    (source_id,),
                    text=text,
                ),
            ),
        ),
        (modality,),
        (time_range,),
    )


def test_candidate_fusion_builds_ranked_multimodal_windows() -> None:
    query = "what happened"
    result = CandidateFusionCapability().execute(
        CandidateFusionRequest(
            VIDEO_ID,
            query,
            (
                _retrieval(
                    query,
                    EvidenceModality.TRANSCRIPT,
                    "segment-1",
                    TimeRange(1_000, 2_000),
                    "a person speaks",
                ),
                _retrieval(
                    query,
                    EvidenceModality.OCR,
                    "ocr-1",
                    TimeRange(2_500, 3_000),
                    "live",
                ),
                _retrieval(
                    query,
                    EvidenceModality.VISUAL_DESCRIPTION,
                    "visual-1",
                    TimeRange(18_000, 19_000),
                    "a door opens",
                ),
            ),
            _chunks(),
            _shots(),
            CapabilityRequestContext("candidate-fusion"),
            max_gap_ms=100,
        )
    )
    assert result.status is CapabilityStatus.SUCCESS
    assert result.data is not None
    assert len(result.data.windows) == 2
    first = result.data.windows[0]
    assert first.time_range == TimeRange(0, 5_000)
    assert first.modalities == (EvidenceModality.OCR, EvidenceModality.TRANSCRIPT)
    assert first.chunk_ids == ("chunk-1",)
    assert set(first.evidence_ids).issubset(
        {item.evidence_id for item in result.data.evidence.items}
    )


@dataclass
class _RapidOutput:
    img: np.ndarray
    boxes: np.ndarray
    txts: tuple[str, ...]
    scores: tuple[float, ...]


class _RapidEngine:
    def __call__(self, path: Path) -> _RapidOutput:
        assert path.is_file()
        return _RapidOutput(
            np.zeros((100, 200, 3), dtype=np.uint8),
            np.asarray([[[10, 10], [80, 10], [80, 30], [10, 30]]], dtype=np.float32),
            ("hello",),
            (0.98,),
        )


def test_rapidocr_backend_adapts_package_output(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    backend = RapidOCRBackend(engine=_RapidEngine())

    result = backend.recognize((OCRFrameInput("frame-1", str(image)),))

    assert result[0].width == 200
    assert result[0].height == 100
    assert result[0].detections[0].text == "hello"
    assert result[0].detections[0].confidence == 0.98
