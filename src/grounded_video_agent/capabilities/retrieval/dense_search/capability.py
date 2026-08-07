from __future__ import annotations

from time import perf_counter

from grounded_video_agent.capabilities._support import make_provenance
from grounded_video_agent.capabilities.indexing._dense import search_dense_index
from grounded_video_agent.capabilities.retrieval.dense_search.contracts import (
    DenseRetrievalRequest,
)
from grounded_video_agent.domain import (
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
from grounded_video_agent.infrastructure.embeddings import TextEmbeddingBackend


class DenseRetrievalCapability:
    VERSION = "1.0.0"

    def __init__(self, backend: TextEmbeddingBackend) -> None:
        self._backend = backend

    def execute(self, request: DenseRetrievalRequest) -> CapabilityResult[RetrievalResult]:
        started = perf_counter()
        if request.index.index_kind is not IndexKind.DENSE:
            return self._failure("Expected a dense index.", started)
        try:
            model_info = self._backend.get_model_info()
            query_vector = self._backend.embed_query(request.query)
            scored = search_dense_index(
                request.index.index_artifact.uri,
                query_vector,
                embedding_space=model_info.embedding_space,
                dimensions=model_info.dimensions,
                top_k=request.top_k,
                min_score=request.min_score,
                within=request.within,
                required_source_ids=frozenset(request.required_source_ids),
                required_tags=frozenset(request.required_tags),
                expected_video_id=request.index.video_id,
                expected_modality=request.index.modality.value,
            )
            evidence_modality = _evidence_modality(request.index.modality)
        except Exception as error:
            return self._failure(str(error), started)
        hits = tuple(
            RetrievalHit(
                rank,
                EvidenceItem(
                    evidence_id=f"dense_hit_{request.context.operation_id}_{rank:06d}",
                    video_id=request.index.video_id,
                    time_range=item.document.time_range,
                    modality=evidence_modality,
                    source_ids=item.document.source_ids,
                    text=item.document.text,
                    scores=(EvidenceScore("cosine", item.score),),
                ),
            )
            for rank, item in enumerate(scored, start=1)
        )
        result = RetrievalResult(
            request.query,
            hits,
            (evidence_modality,),
            tuple(hit.item.time_range for hit in hits),
        )
        provenance = make_provenance(
            "dense-retrieval",
            self.VERSION,
            {"request": request, "model": model_info},
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
                model_calls=1,
            ),
            provenance=provenance,
        )

    @staticmethod
    def _failure(message: str, started: float) -> CapabilityResult[RetrievalResult]:
        return CapabilityResult(
            status=CapabilityStatus.FAILED,
            data=None,
            error=CapabilityError("DENSE_RETRIEVAL_FAILED", message, "embedding"),
            usage=CapabilityUsage(wall_time_ms=round((perf_counter() - started) * 1000)),
        )


def _evidence_modality(modality: IndexModality) -> EvidenceModality:
    mapping = {
        IndexModality.TRANSCRIPT: EvidenceModality.TRANSCRIPT,
        IndexModality.OCR: EvidenceModality.OCR,
        IndexModality.VISUAL_DESCRIPTION: EvidenceModality.VISUAL_DESCRIPTION,
    }
    try:
        return mapping[modality]
    except KeyError as error:
        raise ValueError(f"unsupported dense text modality: {modality}") from error
