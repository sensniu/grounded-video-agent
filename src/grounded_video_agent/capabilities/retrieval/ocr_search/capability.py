from __future__ import annotations

from time import perf_counter

from grounded_video_agent.capabilities._support import make_provenance
from grounded_video_agent.capabilities.indexing._bm25 import search_index
from grounded_video_agent.capabilities.retrieval.ocr_search.contracts import OCRRetrievalRequest
from grounded_video_agent.domain import (
    ArtifactRef,
    CapabilityError,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    EvidenceItem,
    EvidenceModality,
    EvidenceScore,
    IndexKind,
    IndexModality,
    RetrievalHit,
    RetrievalResult,
)


class OCRRetrievalCapability:
    VERSION = "1.0.0"

    def execute(self, request: OCRRetrievalRequest) -> CapabilityResult[RetrievalResult]:
        started = perf_counter()
        if (
            request.index.modality is not IndexModality.OCR
            or request.index.index_kind is not IndexKind.BM25
        ):
            return self._failure("Expected a BM25 OCR index.", started)
        try:
            scored = search_index(
                request.index.index_artifact.uri,
                request.query,
                top_k=request.top_k,
                min_score=request.min_score,
                within=request.within,
                required_source_ids=frozenset(request.related_ids),
                required_tags=(
                    frozenset((request.language,))
                    if request.language is not None
                    else frozenset()
                ),
                expected_video_id=request.index.video_id,
                expected_modality=IndexModality.OCR.value,
            )
        except (OSError, ValueError) as error:
            return self._failure(str(error), started)
        frame_artifacts = self._frame_artifacts(request)
        hits = tuple(
            RetrievalHit(
                rank,
                EvidenceItem(
                    evidence_id=f"ocr_hit_{request.context.operation_id}_{rank:06d}",
                    video_id=request.index.video_id,
                    time_range=item.document.time_range,
                    modality=EvidenceModality.OCR,
                    source_ids=item.document.source_ids,
                    text=item.document.text,
                    artifacts=tuple(
                        frame_artifacts[source_id]
                        for source_id in item.document.source_ids
                        if source_id in frame_artifacts
                    ),
                    scores=(EvidenceScore("bm25", item.score),),
                ),
            )
            for rank, item in enumerate(scored, start=1)
        )
        result = RetrievalResult(
            request.query,
            hits,
            (EvidenceModality.OCR,),
            tuple(hit.item.time_range for hit in hits),
        )
        provenance = make_provenance(
            "ocr-retrieval",
            self.VERSION,
            request,
            video_id=request.index.video_id,
            source_artifact_ids=(request.index.index_artifact.artifact_id,),
        )
        return CapabilityResult(
            status=CapabilityStatus.SUCCESS,
            data=result,
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=1,
                output_items=len(hits),
            ),
            provenance=provenance,
        )

    @staticmethod
    def _frame_artifacts(request: OCRRetrievalRequest) -> dict[str, ArtifactRef]:
        if request.frames is None:
            return {}
        return {frame.frame_id: frame.image for frame in request.frames.frames}

    @staticmethod
    def _failure(message: str, started: float) -> CapabilityResult[RetrievalResult]:
        return CapabilityResult(
            status=CapabilityStatus.FAILED,
            data=None,
            error=CapabilityError("OCR_RETRIEVAL_FAILED", message, "local_index"),
            usage=CapabilityUsage(wall_time_ms=round((perf_counter() - started) * 1000)),
        )
