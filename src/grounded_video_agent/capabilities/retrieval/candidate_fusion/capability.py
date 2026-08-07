from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from grounded_video_agent.capabilities._support import make_provenance
from grounded_video_agent.capabilities.retrieval.candidate_fusion.contracts import (
    CandidateFusionRequest,
)
from grounded_video_agent.domain import (
    CandidateWindow,
    CandidateWindowSet,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    EvidenceBundle,
    EvidenceItem,
    EvidenceModality,
    EvidenceScore,
    TimeRange,
)


@dataclass(slots=True)
class _FusedEvidence:
    item: EvidenceItem
    score: float


class CandidateFusionCapability:
    VERSION = "1.0.0"

    def execute(
        self,
        request: CandidateFusionRequest,
    ) -> CapabilityResult[CandidateWindowSet]:
        started = perf_counter()
        evidence = self._fuse_evidence(request)
        clusters = self._cluster(evidence, request)
        ranked_clusters = sorted(
            clusters,
            key=lambda cluster: (
                -sum(item.score for item in cluster),
                self._range(cluster),
            ),
        )[: request.top_k]
        selected_items: list[EvidenceItem] = []
        windows: list[CandidateWindow] = []
        for rank, cluster in enumerate(ranked_clusters, start=1):
            candidate_range = self._aligned_range(cluster, request)
            cluster_items = tuple(item.item for item in cluster)
            selected_items.extend(cluster_items)
            modalities = tuple(
                sorted({item.modality for item in cluster_items}, key=lambda item: item.value)
            )
            chunk_ids = tuple(
                chunk.chunk_id
                for chunk in request.chunks.chunks
                if chunk.time_range.overlaps(candidate_range)
            )
            shot_ids = tuple(
                shot.shot_id
                for shot in request.shots.shots
                if shot.time_range.overlaps(candidate_range)
            )
            windows.append(
                CandidateWindow(
                    candidate_id=f"candidate_{request.context.operation_id}_{rank:06d}",
                    rank=rank,
                    video_id=request.video_id,
                    time_range=candidate_range,
                    evidence_ids=tuple(item.evidence_id for item in cluster_items),
                    modalities=modalities,
                    chunk_ids=chunk_ids,
                    shot_ids=shot_ids,
                    scores=(
                        EvidenceScore("fusion_rrf", sum(item.score for item in cluster)),
                        EvidenceScore("modality_count", float(len(modalities))),
                    ),
                )
            )
        unique_items = tuple(
            {item.evidence_id: item for item in selected_items}.values()
        )
        bundle = EvidenceBundle(
            bundle_id=f"evidence_{request.context.operation_id}",
            question=request.query,
            items=unique_items,
            covered_ranges=tuple(window.time_range for window in windows),
        )
        result = CandidateWindowSet(request.query, request.video_id, tuple(windows), bundle)
        provenance = make_provenance(
            "candidate-fusion",
            self.VERSION,
            request,
            video_id=request.video_id,
        )
        status = CapabilityStatus.SUCCESS if windows else CapabilityStatus.PARTIAL
        warnings = () if windows else ("Retrieval produced no candidate windows.",)
        return CapabilityResult(
            status=status,
            data=result,
            warnings=warnings,
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=sum(len(result.hits) for result in request.results),
                output_items=len(windows),
                processed_duration_ms=sum(window.time_range.duration_ms for window in windows),
            ),
            provenance=provenance,
        )

    @staticmethod
    def _fuse_evidence(request: CandidateFusionRequest) -> tuple[_FusedEvidence, ...]:
        weights = {item.modality: item.weight for item in request.modality_weights}
        accumulated: dict[tuple[EvidenceModality, str], _FusedEvidence] = {}
        for result in request.results:
            for hit in result.hits:
                item = hit.item
                weight = weights.get(item.modality, 1.0)
                contribution = weight / (request.rrf_k + hit.rank)
                key = (item.modality, item.source_ids[0])
                existing = accumulated.get(key)
                if existing is None:
                    scores = {score.name: score.value for score in item.scores}
                    scores["multimodal_rrf"] = contribution
                    accumulated[key] = _FusedEvidence(
                        EvidenceItem(
                            evidence_id=(
                                f"fusion_evidence_{request.context.operation_id}_"
                                f"{len(accumulated):06d}"
                            ),
                            video_id=item.video_id,
                            time_range=item.time_range,
                            modality=item.modality,
                            source_ids=item.source_ids,
                            text=item.text,
                            artifacts=item.artifacts,
                            scores=tuple(
                                EvidenceScore(name, value) for name, value in scores.items()
                            ),
                            confidence=item.confidence,
                        ),
                        contribution,
                    )
                else:
                    existing.score += contribution
                    scores = {
                        score.name: score.value
                        for score in existing.item.scores
                        if score.name != "multimodal_rrf"
                    }
                    scores["multimodal_rrf"] = existing.score
                    existing.item = EvidenceItem(
                        evidence_id=existing.item.evidence_id,
                        video_id=existing.item.video_id,
                        time_range=existing.item.time_range,
                        modality=existing.item.modality,
                        source_ids=existing.item.source_ids,
                        text=existing.item.text,
                        artifacts=existing.item.artifacts,
                        scores=tuple(EvidenceScore(name, value) for name, value in scores.items()),
                        confidence=existing.item.confidence,
                    )
        return tuple(
            sorted(
                accumulated.values(),
                key=lambda item: (item.item.time_range, -item.score),
            )
        )

    @classmethod
    def _cluster(
        cls,
        evidence: tuple[_FusedEvidence, ...],
        request: CandidateFusionRequest,
    ) -> tuple[tuple[_FusedEvidence, ...], ...]:
        clusters: list[list[_FusedEvidence]] = []
        for item in evidence:
            if not clusters:
                clusters.append([item])
                continue
            current_range = cls._range(tuple(clusters[-1]))
            combined = TimeRange(
                min(current_range.start_ms, item.item.time_range.start_ms),
                max(current_range.end_ms, item.item.time_range.end_ms),
            )
            near = item.item.time_range.start_ms <= current_range.end_ms + request.max_gap_ms
            same_chunk = any(
                chunk.time_range.overlaps(current_range)
                and chunk.time_range.overlaps(item.item.time_range)
                for chunk in request.chunks.chunks
            )
            if (near or same_chunk) and combined.duration_ms <= request.max_window_ms:
                clusters[-1].append(item)
            else:
                clusters.append([item])
        return tuple(tuple(cluster) for cluster in clusters)

    @staticmethod
    def _range(cluster: tuple[_FusedEvidence, ...]) -> TimeRange:
        return TimeRange(
            min(item.item.time_range.start_ms for item in cluster),
            max(item.item.time_range.end_ms for item in cluster),
        )

    @classmethod
    def _aligned_range(
        cls,
        cluster: tuple[_FusedEvidence, ...],
        request: CandidateFusionRequest,
    ) -> TimeRange:
        raw = cls._range(cluster)
        if not request.align_to_chunks:
            return raw
        overlapping = tuple(
            chunk for chunk in request.chunks.chunks if chunk.time_range.overlaps(raw)
        )
        if not overlapping:
            return raw
        aligned = TimeRange(
            min(chunk.time_range.start_ms for chunk in overlapping),
            max(chunk.time_range.end_ms for chunk in overlapping),
        )
        return aligned if aligned.duration_ms <= request.max_window_ms else raw
