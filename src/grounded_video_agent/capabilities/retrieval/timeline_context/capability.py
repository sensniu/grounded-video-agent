from __future__ import annotations

from time import perf_counter

from grounded_video_agent.capabilities._support import make_provenance
from grounded_video_agent.capabilities.retrieval.timeline_context.contracts import (
    TimelineContextRequest,
)
from grounded_video_agent.domain import (
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    TimelineContext,
    TimeRange,
)


class TimelineContextCapability:
    VERSION = "2.0.0"

    def execute(self, request: TimelineContextRequest) -> CapabilityResult[TimelineContext]:
        started = perf_counter()
        requested_ranges = list(request.ranges)
        evidence_ids: tuple[str, ...] = ()
        if request.retrieval is not None:
            requested_ranges.extend(hit.item.time_range for hit in request.retrieval.hits)
            evidence_ids = tuple(hit.item.evidence_id for hit in request.retrieval.hits)
        requested_ranges.extend(
            chunk.time_range
            for chunk in request.chunks.chunks
            if chunk.chunk_id in request.chunk_ids
        )
        ordered_requested = self._merge_ranges(tuple(sorted(set(requested_ranges))))
        selected_indexes = {
            index
            for index, chunk in enumerate(request.chunks.chunks)
            if any(chunk.time_range.overlaps(item) for item in ordered_requested)
        }
        expanded_indexes = {
            nearby
            for index in selected_indexes
            for nearby in range(
                max(0, index - request.adjacent_chunks),
                min(len(request.chunks.chunks), index + request.adjacent_chunks + 1),
            )
        }
        selected_chunks = tuple(request.chunks.chunks[index] for index in sorted(expanded_indexes))
        resolved_ranges = self._merge_ranges(
            tuple(chunk.observation_range for chunk in selected_chunks)
        )
        shots = tuple(
            shot
            for shot in request.shots.shots
            if any(shot.time_range.overlaps(item) for item in resolved_ranges)
        )
        segments = tuple(
            segment
            for segment in request.transcript.segments
            if any(segment.time_range.overlaps(item) for item in resolved_ranges)
        )
        result = TimelineContext(
            video_id=request.video_id,
            requested_ranges=ordered_requested,
            resolved_ranges=resolved_ranges,
            chunks=selected_chunks,
            shots=shots,
            transcript_segments=segments,
            source_evidence_ids=evidence_ids,
        )
        provenance = make_provenance(
            "timeline-context",
            self.VERSION,
            request,
            video_id=request.video_id,
            source_artifact_ids=(
                request.chunks.ref.artifact.artifact_id,
                request.shots.ref.artifact.artifact_id,
                request.transcript.ref.artifact.artifact_id,
            ),
        )
        status = CapabilityStatus.SUCCESS if selected_chunks else CapabilityStatus.PARTIAL
        warnings = () if selected_chunks else ("No chunks overlap the requested context.",)
        return CapabilityResult(
            status=status,
            data=result,
            warnings=warnings,
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=len(ordered_requested),
                output_items=len(selected_chunks),
                processed_duration_ms=sum(item.duration_ms for item in resolved_ranges),
            ),
            provenance=provenance,
        )

    @staticmethod
    def _merge_ranges(ranges: tuple[TimeRange, ...]) -> tuple[TimeRange, ...]:
        merged: list[TimeRange] = []
        for item in ranges:
            if not merged or merged[-1].end_ms < item.start_ms:
                merged.append(item)
                continue
            previous = merged[-1]
            merged[-1] = TimeRange(previous.start_ms, max(previous.end_ms, item.end_ms))
        return tuple(merged)
