from __future__ import annotations

from typing import cast

from grounded_video_agent.agent.tools._support import (
    SCHEMA_VERSION,
    failed_result,
    start_tool,
)
from grounded_video_agent.agent.tools.contracts import (
    EvidenceDelta,
    FrameObservation,
    InspectVisualContentInput,
    ToolProgress,
    ToolResult,
    ToolStatus,
    VisualInspectionOutput,
    VisualObservationOutput,
)
from grounded_video_agent.agent.tools.media_operations import VisualOperation
from grounded_video_agent.agent.tools.resolution import (
    ResolvedTarget,
    resolve_targets,
    sampling_ranges,
)
from grounded_video_agent.agent.tools.runtime import ToolRuntimeContext, stable_id
from grounded_video_agent.domain import CapabilityStatus, EvidenceItem, EvidenceModality
from grounded_video_agent.workspace.catalog import CatalogError


class InspectVisualContentTool:
    name = "inspect_visual_content"
    description = (
        "Sample frames from selected transcript candidates or timeline windows and ask the "
        "configured local visual model a focused question."
    )
    input_type = InspectVisualContentInput

    def __init__(self, operation: VisualOperation, *, enabled: bool = True) -> None:
        self._operation = operation
        self.enabled = enabled

    def execute(
        self,
        request: InspectVisualContentInput,
        runtime: ToolRuntimeContext,
    ) -> ToolResult[VisualInspectionOutput]:
        call_id, early = start_tool(runtime, self.name)
        if early is not None:
            return cast(ToolResult[VisualInspectionOutput], early)
        assert call_id is not None
        try:
            targets = resolve_targets(
                runtime,
                candidate_ids=request.candidate_ids,
                context_window_ids=request.context_window_ids,
                chunk_ids=request.chunk_ids,
                ranges=request.ranges,
            )
        except (CatalogError, KeyError) as error:
            return cast(
                ToolResult[VisualInspectionOutput],
                failed_result(call_id, "VISUAL_TARGET_RESOLUTION_FAILED", str(error)),
            )
        question = request.question
        if request.focus:
            question += " Focus on: " + ", ".join(request.focus)
        run = self._operation.inspect(runtime, call_id, targets, question, request.detail)
        if run.status is CapabilityStatus.FAILED:
            assert run.error is not None
            return cast(
                ToolResult[VisualInspectionOutput],
                failed_result(
                    call_id,
                    run.error.code,
                    run.error.message,
                    retryable=run.error.retryable,
                    usage=run.usage,
                ),
            )
        frames = run.frames.frames if run.frames is not None else ()
        frame_by_id = {frame.frame_id: frame for frame in frames}
        observations: list[VisualObservationOutput] = []
        new_ids: list[str] = []
        reused_ids: list[str] = []
        if run.descriptions is not None:
            for item in run.descriptions.descriptions:
                target_id = self._target_id(item.time_range, targets)
                timestamps = tuple(
                    frame_by_id[frame_id].timestamp_ms
                    for frame_id in item.frame_ids
                    if frame_id in frame_by_id
                )
                evidence_id = stable_id(
                    "evidence",
                    (
                        runtime.video_id,
                        EvidenceModality.VLM_OBSERVATION,
                        item.time_range,
                        timestamps,
                        item.text,
                        question,
                    ),
                )
                artifacts = tuple(
                    frame_by_id[frame_id].image
                    for frame_id in item.frame_ids
                    if frame_id in frame_by_id
                )
                evidence = EvidenceItem(
                    evidence_id,
                    runtime.video_id,
                    item.time_range,
                    EvidenceModality.VLM_OBSERVATION,
                    tuple(dict.fromkeys((target_id, *item.frame_ids))),
                    text=item.text,
                    artifacts=artifacts,
                    confidence=item.confidence,
                )
                (new_ids if runtime.evidence.add(evidence) else reused_ids).append(evidence_id)
                observations.append(
                    VisualObservationOutput(
                        evidence_id,
                        target_id,
                        item.time_range,
                        item.text,
                        item.frame_ids,
                        timestamps,
                        item.tags,
                        item.confidence,
                    )
                )
        covered = runtime.coverage.add(sampling_ranges(targets)) if frames else ()
        gained = bool(new_ids or covered)
        runtime.record_information_gain(gained)
        runtime.record_usage(run.usage)
        output = VisualInspectionOutput(
            sampling_ranges(targets),
            tuple(FrameObservation(frame.frame_id, frame.timestamp_ms) for frame in frames),
            tuple(observations),
            run.reused_frames,
            run.reused_analysis,
        )
        return ToolResult(
            SCHEMA_VERSION,
            call_id,
            ToolStatus.SUCCESS if run.status is CapabilityStatus.SUCCESS else ToolStatus.PARTIAL,
            output,
            EvidenceDelta(tuple(dict.fromkeys(new_ids)), tuple(dict.fromkeys(reused_ids))),
            ToolProgress(
                new_evidence_count=len(set(new_ids)),
                newly_covered_ranges=covered,
                cache_hit=run.reused_analysis,
                no_information_gain=not gained,
            ),
            run.warnings,
            usage=run.usage,
        )

    @staticmethod
    def _target_id(time_range: object, targets: tuple[ResolvedTarget, ...]) -> str:
        return next(
            (target.target_id for target in targets if target.time_range == time_range),
            "visual_target",
        )
