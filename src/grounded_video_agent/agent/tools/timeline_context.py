from __future__ import annotations

from typing import cast

from grounded_video_agent.agent.tools._support import (
    SCHEMA_VERSION,
    capability_failure,
    catalog_failure,
    failed_result,
    start_tool,
)
from grounded_video_agent.agent.tools.contracts import (
    ContextDirection,
    EvidenceDelta,
    ResolveTimelineContextInput,
    SubtitleExcerpt,
    TimelineContextOutput,
    ToolProgress,
    ToolResult,
    ToolStatus,
)
from grounded_video_agent.agent.tools.dependencies import VideoToolDependencies
from grounded_video_agent.agent.tools.runtime import (
    ContextWindowState,
    ToolRuntimeContext,
    stable_id,
)
from grounded_video_agent.capabilities.retrieval.timeline_context import TimelineContextRequest
from grounded_video_agent.domain import (
    CapabilityStatus,
    ChunkManifest,
    EvidenceItem,
    EvidenceModality,
    ShotManifest,
    TranscriptManifest,
)
from grounded_video_agent.pipelines.preprocessing.keys import (
    CHUNKS_KEY,
    SHOTS_KEY,
    TRANSCRIPT_KEY,
)
from grounded_video_agent.workspace.catalog import CatalogError


class ResolveTimelineContextTool:
    name = "resolve_timeline_context"
    enabled = True
    description = (
        "Expand transcript candidates or explicit time ranges into adjacent timeline context. "
        "Use this when a result may be out of context or when order and causality matter."
    )
    input_type = ResolveTimelineContextInput

    def __init__(self, dependencies: VideoToolDependencies) -> None:
        self._dependencies = dependencies

    def execute(
        self,
        request: ResolveTimelineContextInput,
        runtime: ToolRuntimeContext,
    ) -> ToolResult[TimelineContextOutput]:
        call_id, early = start_tool(runtime, self.name)
        if early is not None:
            return cast(ToolResult[TimelineContextOutput], early)
        assert call_id is not None
        try:
            chunks = runtime.catalog.load_manifest(runtime.video_id, CHUNKS_KEY, ChunkManifest)
            shots = runtime.catalog.load_manifest(runtime.video_id, SHOTS_KEY, ShotManifest)
            transcript = runtime.catalog.load_manifest(
                runtime.video_id, TRANSCRIPT_KEY, TranscriptManifest
            )
            selected_ids, truncated = self._select_chunks(request, runtime, chunks)
        except CatalogError as error:
            return cast(ToolResult[TimelineContextOutput], catalog_failure(call_id, error))
        except KeyError as error:
            return cast(
                ToolResult[TimelineContextOutput],
                failed_result(call_id, "UNKNOWN_TIMELINE_TARGET", str(error)),
            )

        result = self._dependencies.timeline_context.execute(
            TimelineContextRequest(
                runtime.video_id,
                chunks,
                shots,
                transcript,
                runtime.capability_context(call_id, "context"),
                ranges=request.ranges,
                chunk_ids=selected_ids,
                adjacent_chunks=0,
            )
        )
        if result.status is CapabilityStatus.FAILED:
            return cast(ToolResult[TimelineContextOutput], capability_failure(call_id, result))
        assert result.data is not None
        new_ids: list[str] = []
        reused_ids: list[str] = []
        excerpts: list[SubtitleExcerpt] = []
        for segment in result.data.transcript_segments:
            evidence_id = stable_id(
                "evidence",
                (
                    runtime.video_id,
                    EvidenceModality.TRANSCRIPT,
                    segment.segment_id,
                    segment.time_range,
                    segment.normalized_text,
                ),
            )
            evidence = EvidenceItem(
                evidence_id,
                runtime.video_id,
                segment.time_range,
                EvidenceModality.TRANSCRIPT,
                (segment.segment_id,),
                text=segment.raw_text,
                confidence=segment.confidence,
            )
            (new_ids if runtime.evidence.add(evidence) else reused_ids).append(evidence_id)
            excerpts.append(
                SubtitleExcerpt(
                    segment.segment_id,
                    segment.time_range,
                    segment.raw_text,
                    evidence_id,
                )
            )
        context_id = stable_id(
            "context",
            (
                runtime.video_id,
                tuple(chunk.chunk_id for chunk in result.data.chunks),
                result.data.resolved_ranges,
            ),
        )
        anchor_evidence_ids = tuple(
            runtime.candidates.get(candidate_id).evidence_id
            for candidate_id in request.candidate_ids
        )
        reused_ids.extend(anchor_evidence_ids)
        evidence_ids = tuple(
            dict.fromkeys(
                (
                    *anchor_evidence_ids,
                    *result.data.source_evidence_ids,
                    *(item.evidence_id for item in excerpts),
                )
            )
        )
        runtime.contexts.put(
            ContextWindowState(
                context_id,
                result.data.resolved_ranges,
                tuple(chunk.chunk_id for chunk in result.data.chunks),
                evidence_ids,
            )
        )
        gained = bool(new_ids)
        runtime.record_information_gain(gained)
        runtime.record_usage(result.usage)
        warnings = result.warnings
        if truncated:
            warnings += ("Adjacent context was limited by max_duration_ms.",)
        output = TimelineContextOutput(
            context_id,
            result.data.requested_ranges,
            result.data.resolved_ranges,
            tuple(chunk.chunk_id for chunk in result.data.chunks),
            tuple(shot.shot_id for shot in result.data.shots),
            tuple(excerpts),
            evidence_ids,
        )
        status = (
            ToolStatus.SUCCESS
            if result.status is CapabilityStatus.SUCCESS
            else ToolStatus.PARTIAL
        )
        return ToolResult(
            SCHEMA_VERSION,
            call_id,
            status,
            output,
            EvidenceDelta(tuple(dict.fromkeys(new_ids)), tuple(dict.fromkeys(reused_ids))),
            ToolProgress(
                new_evidence_count=len(set(new_ids)),
                no_information_gain=not gained,
            ),
            warnings,
            usage=result.usage,
        )

    @staticmethod
    def _select_chunks(
        request: ResolveTimelineContextInput,
        runtime: ToolRuntimeContext,
        manifest: ChunkManifest,
    ) -> tuple[tuple[str, ...], bool]:
        chunks = manifest.chunks
        indexes = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
        anchor_ids = set(request.chunk_ids)
        anchor_ids.update(runtime.candidates.get(item).chunk_id for item in request.candidate_ids)
        anchor_indexes = {indexes[item] for item in anchor_ids}
        anchor_indexes.update(
            index
            for index, chunk in enumerate(chunks)
            if any(chunk.time_range.overlaps(item) for item in request.ranges)
        )
        if not anchor_indexes:
            raise KeyError("no chunks overlap the requested timeline targets")
        selected = set(anchor_indexes)
        candidates: list[int] = []
        for index in sorted(anchor_indexes):
            if request.direction in {ContextDirection.BEFORE, ContextDirection.BOTH}:
                candidates.extend(range(max(0, index - request.adjacent_chunks), index))
            if request.direction in {ContextDirection.AFTER, ContextDirection.BOTH}:
                candidates.extend(
                    range(index + 1, min(len(chunks), index + request.adjacent_chunks + 1))
                )
        duration = sum(chunks[index].observation_range.duration_ms for index in selected)
        truncated = False
        ordered_candidates = sorted(
            set(candidates),
            key=lambda item: min(abs(item - anchor) for anchor in anchor_indexes),
        )
        for index in ordered_candidates:
            item_duration = chunks[index].observation_range.duration_ms
            if duration + item_duration > request.max_duration_ms:
                truncated = True
                continue
            selected.add(index)
            duration += item_duration
        return tuple(chunks[index].chunk_id for index in sorted(selected)), truncated
