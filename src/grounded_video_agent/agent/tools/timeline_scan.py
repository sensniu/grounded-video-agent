from __future__ import annotations

from typing import cast

from grounded_video_agent.agent.tools._support import SCHEMA_VERSION, failed_result, start_tool
from grounded_video_agent.agent.tools.contracts import (
    EvidenceDelta,
    ScanVideoTimelineInput,
    TimelineScanCandidate,
    TimelineScanOutput,
    ToolProgress,
    ToolResult,
    ToolStatus,
)
from grounded_video_agent.agent.tools.media_operations import VisualOperation
from grounded_video_agent.agent.tools.resolution import ResolvedTarget, load_shots
from grounded_video_agent.agent.tools.runtime import (
    ContextWindowState,
    ToolRuntimeContext,
    stable_id,
)
from grounded_video_agent.domain import CapabilityStatus, EvidenceItem, EvidenceModality, TimeRange
from grounded_video_agent.pipelines.preprocessing.keys import MEDIA_INSPECTION_KEY
from grounded_video_agent.workspace.catalog import CatalogError, MediaInspectionDocument


class ScanVideoTimelineTool:
    name = "scan_video_timeline"
    description = (
        "Coarsely inspect previously unseen timeline regions. Use when transcript search is "
        "unavailable or exhausted, or when a question requires broad video coverage."
    )
    input_type = ScanVideoTimelineInput

    def __init__(self, operation: VisualOperation, *, enabled: bool = True) -> None:
        self._operation = operation
        self.enabled = enabled

    def execute(
        self,
        request: ScanVideoTimelineInput,
        runtime: ToolRuntimeContext,
    ) -> ToolResult[TimelineScanOutput]:
        call_id, early = start_tool(runtime, self.name)
        if early is not None:
            return cast(ToolResult[TimelineScanOutput], early)
        assert call_id is not None
        try:
            inspection = runtime.catalog.load_document(
                runtime.video_id,
                MEDIA_INSPECTION_KEY,
                MediaInspectionDocument,
            )
            shots = load_shots(runtime)
        except CatalogError as error:
            return cast(
                ToolResult[TimelineScanOutput],
                failed_result(call_id, f"CATALOG_{error.code.value.upper()}", str(error)),
            )
        duration_ms = inspection.media_probe.container.duration_ms
        if duration_ms is None or duration_ms <= 0:
            return cast(
                ToolResult[TimelineScanOutput],
                failed_result(call_id, "UNKNOWN_VIDEO_DURATION", "Video duration is unavailable."),
            )
        total = TimeRange(0, duration_ms)
        unseen = runtime.coverage.unseen(total)
        available = tuple(
            shot for shot in shots.shots if any(shot.time_range.overlaps(item) for item in unseen)
        )
        selected = self._evenly_select(available, request.max_windows)
        if not selected:
            output = TimelineScanOutput((), runtime.coverage.ranges, (), 1.0, True)
            runtime.record_information_gain(False)
            return ToolResult(
                SCHEMA_VERSION,
                call_id,
                ToolStatus.SUCCESS,
                output,
                progress=ToolProgress(exhausted=True, no_information_gain=True),
            )
        targets = tuple(
            ResolvedTarget(shot.shot_id, shot.time_range) for shot in selected
        )
        run = self._operation.inspect(
            runtime,
            call_id,
            targets,
            request.question,
            request.detail,
        )
        if run.status is CapabilityStatus.FAILED:
            assert run.error is not None
            return cast(
                ToolResult[TimelineScanOutput],
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
        candidates: list[TimelineScanCandidate] = []
        new_ids: list[str] = []
        reused_ids: list[str] = []
        if run.descriptions is not None:
            for item in run.descriptions.descriptions:
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
                        request.question,
                    ),
                )
                evidence = EvidenceItem(
                    evidence_id,
                    runtime.video_id,
                    item.time_range,
                    EvidenceModality.VLM_OBSERVATION,
                    tuple(
                        dict.fromkeys(
                            (
                                self._target_id(item.time_range, targets),
                                *item.frame_ids,
                            )
                        )
                    ),
                    text=item.text,
                    artifacts=tuple(
                        frame_by_id[frame_id].image
                        for frame_id in item.frame_ids
                        if frame_id in frame_by_id
                    ),
                    confidence=item.confidence,
                )
                (new_ids if runtime.evidence.add(evidence) else reused_ids).append(evidence_id)
                context_id = stable_id(
                    "context", (runtime.video_id, item.time_range, evidence_id)
                )
                runtime.contexts.put(
                    ContextWindowState(context_id, (item.time_range,), (), (evidence_id,))
                )
                candidates.append(
                    TimelineScanCandidate(
                        context_id,
                        item.time_range,
                        item.text,
                        (evidence_id,),
                        timestamps,
                        item.confidence,
                    )
                )
        selected_ranges = tuple(shot.time_range for shot in selected)
        newly_covered = runtime.coverage.add(selected_ranges) if frames else ()
        unseen_after = runtime.coverage.unseen(total)
        covered_ms = duration_ms - sum(item.duration_ms for item in unseen_after)
        coverage_ratio = max(0.0, min(1.0, covered_ms / duration_ms))
        exhausted = not unseen_after
        gained = bool(new_ids or newly_covered)
        runtime.record_information_gain(gained)
        runtime.record_usage(run.usage)
        output = TimelineScanOutput(
            tuple(candidates),
            runtime.coverage.ranges,
            unseen_after,
            coverage_ratio,
            exhausted,
        )
        return ToolResult(
            SCHEMA_VERSION,
            call_id,
            ToolStatus.SUCCESS if run.status is CapabilityStatus.SUCCESS else ToolStatus.PARTIAL,
            output,
            EvidenceDelta(tuple(dict.fromkeys(new_ids)), tuple(dict.fromkeys(reused_ids))),
            ToolProgress(
                new_candidate_count=len(candidates),
                new_evidence_count=len(set(new_ids)),
                newly_covered_ranges=newly_covered,
                cache_hit=run.reused_analysis,
                exhausted=exhausted,
                no_information_gain=not gained,
            ),
            run.warnings,
            usage=run.usage,
        )

    @staticmethod
    def _evenly_select(items: tuple, count: int) -> tuple:
        if len(items) <= count:
            return items
        if count == 1:
            return (items[len(items) // 2],)
        indexes = {round(index * (len(items) - 1) / (count - 1)) for index in range(count)}
        return tuple(items[index] for index in sorted(indexes))

    @staticmethod
    def _target_id(time_range: object, targets: tuple[ResolvedTarget, ...]) -> str:
        return next(
            (target.target_id for target in targets if target.time_range == time_range),
            "scan_target",
        )
