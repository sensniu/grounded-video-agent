from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from grounded_video_agent.capabilities._support import make_provenance
from grounded_video_agent.capabilities.retrieval.hybrid_search.contracts import (
    HybridRetrievalRequest,
)
from grounded_video_agent.domain import (
    ArtifactRef,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    EvidenceItem,
    EvidenceScore,
    RetrievalHit,
    RetrievalResult,
)


@dataclass(slots=True)
class _Entry:
    item: EvidenceItem
    sparse_rank: int | None = None
    dense_rank: int | None = None


class HybridRetrievalCapability:
    VERSION = "1.0.0"

    def execute(self, request: HybridRetrievalRequest) -> CapabilityResult[RetrievalResult]:
        started = perf_counter()
        entries: dict[tuple[object, str], _Entry] = {}
        for hit in request.sparse.hits:
            key = (hit.item.modality, hit.item.source_ids[0])
            entry = entries.setdefault(key, _Entry(hit.item))
            entry.sparse_rank = hit.rank
        for hit in request.dense.hits:
            key = (hit.item.modality, hit.item.source_ids[0])
            entry = entries.setdefault(key, _Entry(hit.item))
            entry.dense_rank = hit.rank
            if entry.item.text is None and hit.item.text is not None:
                entry.item = hit.item
        ranked = sorted(
            entries.values(),
            key=lambda entry: (
                -self._score(entry, request),
                entry.item.time_range,
                entry.item.source_ids[0],
            ),
        )[: request.top_k]
        hits = tuple(
            RetrievalHit(
                rank,
                self._evidence(entry, request, rank),
            )
            for rank, entry in enumerate(ranked, start=1)
        )
        modality = request.sparse.searched_modalities[0]
        result = RetrievalResult(
            request.query,
            hits,
            (modality,),
            tuple(hit.item.time_range for hit in hits),
        )
        provenance = make_provenance(
            "hybrid-retrieval",
            self.VERSION,
            request,
            video_id=request.video_id,
        )
        return CapabilityResult(
            status=CapabilityStatus.SUCCESS,
            data=result,
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=len(request.sparse.hits) + len(request.dense.hits),
                output_items=len(hits),
            ),
            provenance=provenance,
        )

    @staticmethod
    def _score(entry: _Entry, request: HybridRetrievalRequest) -> float:
        score = 0.0
        if entry.sparse_rank is not None:
            score += request.sparse_weight / (request.rrf_k + entry.sparse_rank)
        if entry.dense_rank is not None:
            score += request.dense_weight / (request.rrf_k + entry.dense_rank)
        return score

    def _evidence(
        self,
        entry: _Entry,
        request: HybridRetrievalRequest,
        rank: int,
    ) -> EvidenceItem:
        related = [
            hit.item
            for result in (request.sparse, request.dense)
            for hit in result.hits
            if hit.item.modality is entry.item.modality
            and hit.item.source_ids[0] == entry.item.source_ids[0]
        ]
        source_ids = tuple(
            dict.fromkeys(source_id for item in related for source_id in item.source_ids)
        )
        artifacts = _unique_artifacts(tuple(a for item in related for a in item.artifacts))
        scores = {
            score.name: score.value for item in related for score in item.scores
        }
        if entry.sparse_rank is not None:
            scores["rrf_sparse"] = request.sparse_weight / (
                request.rrf_k + entry.sparse_rank
            )
        if entry.dense_rank is not None:
            scores["rrf_dense"] = request.dense_weight / (
                request.rrf_k + entry.dense_rank
            )
        scores["hybrid_rrf"] = self._score(entry, request)
        return EvidenceItem(
            evidence_id=f"hybrid_hit_{request.context.operation_id}_{rank:06d}",
            video_id=request.video_id,
            time_range=entry.item.time_range,
            modality=entry.item.modality,
            source_ids=source_ids,
            text=entry.item.text,
            artifacts=artifacts,
            scores=tuple(EvidenceScore(name, value) for name, value in scores.items()),
            confidence=entry.item.confidence,
        )


def _unique_artifacts(artifacts: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
    return tuple({artifact.artifact_id: artifact for artifact in artifacts}.values())
