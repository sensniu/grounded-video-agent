from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from grounded_video_agent.agent.tools._support import (
    SCHEMA_VERSION,
    add_usage,
    capability_failure,
    catalog_failure,
    start_tool,
)
from grounded_video_agent.agent.tools.contracts import (
    EvidenceDelta,
    SearchVideoTranscriptInput,
    ToolProgress,
    ToolResult,
    ToolStatus,
    TranscriptCandidate,
    TranscriptSearchOutput,
)
from grounded_video_agent.agent.tools.dependencies import VideoToolDependencies
from grounded_video_agent.agent.tools.runtime import (
    CandidateState,
    SearchAttempt,
    ToolRuntimeContext,
    fingerprint,
    stable_id,
)
from grounded_video_agent.capabilities.retrieval.dense_search import DenseRetrievalRequest
from grounded_video_agent.capabilities.retrieval.hybrid_search import HybridRetrievalRequest
from grounded_video_agent.capabilities.retrieval.transcript_search import (
    TranscriptRetrievalRequest,
)
from grounded_video_agent.domain import (
    CapabilityStatus,
    Chunk,
    ChunkManifest,
    EvidenceItem,
    IndexManifest,
    RetrievalResult,
)
from grounded_video_agent.pipelines.preprocessing.keys import (
    CHUNKS_KEY,
    DENSE_INDEX_KEY,
    SPARSE_INDEX_KEY,
)
from grounded_video_agent.workspace.catalog import CatalogError


@dataclass(frozen=True, slots=True)
class _CachedRetrieval:
    result: RetrievalResult
    mode: str
    warnings: tuple[str, ...] = ()


class SearchVideoTranscriptTool:
    name = "search_video_transcript"
    enabled = True
    description = (
        "Search transcript chunks for the current video. Returns novel and previously seen "
        "candidates separately with exact citation ranges and wider visual inspection ranges."
    )
    input_type = SearchVideoTranscriptInput

    def __init__(self, dependencies: VideoToolDependencies, *, overfetch_factor: int = 3) -> None:
        if overfetch_factor <= 0:
            raise ValueError("overfetch_factor must be positive")
        self._dependencies = dependencies
        self._overfetch_factor = overfetch_factor

    def execute(
        self,
        request: SearchVideoTranscriptInput,
        runtime: ToolRuntimeContext,
    ) -> ToolResult[TranscriptSearchOutput]:
        call_id, early = start_tool(runtime, self.name)
        if early is not None:
            return cast(ToolResult[TranscriptSearchOutput], early)
        assert call_id is not None
        try:
            chunks = runtime.catalog.load_manifest(runtime.video_id, CHUNKS_KEY, ChunkManifest)
            sparse_index = runtime.catalog.load_manifest(
                runtime.video_id, SPARSE_INDEX_KEY, IndexManifest
            )
        except CatalogError as error:
            return cast(ToolResult[TranscriptSearchOutput], catalog_failure(call_id, error))

        dense_index = self._load_dense_index(runtime)
        overfetch = min(50, max(request.top_k, request.top_k * self._overfetch_factor))
        retrieval_key = "retrieval:" + fingerprint(
            (
                request,
                sparse_index.ref.manifest_id,
                dense_index.ref.manifest_id if dense_index is not None else None,
                overfetch,
            )
        )
        cached = runtime.memory.get(retrieval_key, _CachedRetrieval)
        cache_hit = cached is not None
        usages = []
        warnings: list[str] = []
        if cached is not None:
            retrieval = cached.result
            mode = cached.mode
            warnings.extend(cached.warnings)
        else:
            sparse = self._dependencies.transcript_search.execute(
                TranscriptRetrievalRequest(
                    request.query,
                    sparse_index,
                    runtime.capability_context(call_id, "sparse"),
                    top_k=overfetch,
                    min_score=request.min_score,
                    within=request.within,
                    language=request.language,
                )
            )
            if sparse.status is CapabilityStatus.FAILED:
                return cast(
                    ToolResult[TranscriptSearchOutput], capability_failure(call_id, sparse)
                )
            assert sparse.data is not None
            retrieval = sparse.data
            mode = "sparse"
            usages.append(sparse.usage)
            if self._dependencies.dense_search is not None and dense_index is not None:
                dense = self._dependencies.dense_search.execute(
                    DenseRetrievalRequest(
                        request.query,
                        dense_index,
                        runtime.capability_context(call_id, "dense"),
                        top_k=overfetch,
                        min_score=request.min_score,
                        within=request.within,
                        required_tags=((request.language,) if request.language else ()),
                    )
                )
                usages.append(dense.usage)
                if dense.status is not CapabilityStatus.FAILED and dense.data is not None:
                    hybrid = self._dependencies.hybrid_search.execute(
                        HybridRetrievalRequest(
                            runtime.video_id,
                            request.query,
                            sparse.data,
                            dense.data,
                            runtime.capability_context(call_id, "hybrid"),
                            top_k=overfetch,
                        )
                    )
                    usages.append(hybrid.usage)
                    if hybrid.status is not CapabilityStatus.FAILED and hybrid.data is not None:
                        retrieval = hybrid.data
                        mode = "hybrid"
                    elif hybrid.error is not None:
                        warnings.append(
                            f"Hybrid retrieval failed; sparse results were used: "
                            f"{hybrid.error.message}"
                        )
                elif dense.error is not None:
                    warnings.append(
                        f"Dense retrieval failed; sparse results were used: {dense.error.message}"
                    )
            elif dense_index is not None:
                warnings.append(
                    "A dense transcript index exists, but no embedding backend is configured; "
                    "sparse retrieval was used."
                )
            runtime.memory.put(
                retrieval_key,
                _CachedRetrieval(retrieval, mode, tuple(warnings)),
            )

        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks.chunks}
        new_hits: list[TranscriptCandidate] = []
        reused_hits: list[TranscriptCandidate] = []
        new_evidence: list[str] = []
        reused_evidence: list[str] = []
        returned_ids: list[str] = []
        for hit in retrieval.hits:
            chunk = self._matching_chunk(hit.item.source_ids, chunk_by_id)
            if chunk is None or chunk.text is None:
                continue
            evidence = self._canonical_evidence(hit.item, chunk)
            evidence_is_new = runtime.evidence.add(evidence)
            candidate_id = stable_id("candidate", (runtime.video_id, chunk.chunk_id))
            state, candidate_is_new = runtime.candidates.register(
                CandidateState(
                    candidate_id,
                    chunk.chunk_id,
                    chunk.time_range,
                    chunk.observation_range,
                    evidence.evidence_id,
                ),
                request.query,
            )
            candidate = TranscriptCandidate(
                state.candidate_id,
                chunk.chunk_id,
                chunk.text,
                chunk.time_range,
                chunk.observation_range,
                chunk.shot_ids,
                evidence.evidence_id,
                tuple((score.name, score.value) for score in hit.item.scores),
                tuple(state.matched_queries),
                chunk.observation_range != chunk.time_range,
            )
            returned_ids.append(candidate_id)
            if evidence_is_new:
                new_evidence.append(evidence.evidence_id)
            else:
                reused_evidence.append(evidence.evidence_id)
            if candidate_is_new and len(new_hits) < request.top_k:
                new_hits.append(candidate)
            elif len(reused_hits) < request.top_k:
                reused_hits.append(candidate)

        exhausted = len(new_hits) < request.top_k and (
            len(retrieval.hits) < overfetch or not new_hits
        )
        attempt = SearchAttempt(
            fingerprint((request.query, request.within, request.language, request.intent_id)),
            tuple(dict.fromkeys(returned_ids)),
            tuple(item.candidate_id for item in new_hits),
            exhausted,
        )
        runtime.search_attempts.append(attempt)
        gained = bool(new_hits or new_evidence)
        runtime.record_information_gain(gained)
        usage = add_usage(*usages)
        runtime.record_usage(usage)
        output = TranscriptSearchOutput(
            request.query,
            mode,
            tuple(new_hits),
            tuple(reused_hits),
            exhausted,
        )
        return ToolResult(
            SCHEMA_VERSION,
            call_id,
            ToolStatus.SUCCESS,
            output,
            EvidenceDelta(
                tuple(dict.fromkeys(new_evidence)),
                tuple(dict.fromkeys(reused_evidence)),
            ),
            ToolProgress(
                new_candidate_count=len(new_hits),
                new_evidence_count=len(set(new_evidence)),
                cache_hit=cache_hit,
                exhausted=exhausted,
                no_information_gain=not gained,
            ),
            tuple(warnings),
            usage=usage,
        )

    def _load_dense_index(self, runtime: ToolRuntimeContext) -> IndexManifest | None:
        try:
            return runtime.catalog.load_manifest(runtime.video_id, DENSE_INDEX_KEY, IndexManifest)
        except CatalogError:
            return None

    @staticmethod
    def _matching_chunk(
        source_ids: tuple[str, ...],
        chunks: dict[str, Chunk],
    ) -> Chunk | None:
        return next((chunks[item] for item in source_ids if item in chunks), None)

    @staticmethod
    def _canonical_evidence(item: EvidenceItem, chunk: Chunk) -> EvidenceItem:
        evidence_id = stable_id(
            "evidence",
            (
                item.video_id,
                item.modality,
                chunk.chunk_id,
                chunk.time_range,
                chunk.text,
            ),
        )
        return EvidenceItem(
            evidence_id,
            item.video_id,
            chunk.time_range,
            item.modality,
            item.source_ids,
            text=chunk.text,
            artifacts=item.artifacts,
            scores=item.scores,
            confidence=item.confidence,
        )
