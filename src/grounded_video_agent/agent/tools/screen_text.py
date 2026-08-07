from __future__ import annotations

from typing import cast

from grounded_video_agent.agent.tools._support import SCHEMA_VERSION, failed_result, start_tool
from grounded_video_agent.agent.tools.contracts import (
    EvidenceDelta,
    ReadScreenTextInput,
    ScreenTextOutput,
    ScreenTextSpan,
    ToolProgress,
    ToolResult,
    ToolStatus,
)
from grounded_video_agent.agent.tools.media_operations import OCROperation
from grounded_video_agent.agent.tools.resolution import resolve_targets, sampling_ranges
from grounded_video_agent.agent.tools.runtime import ToolRuntimeContext, stable_id
from grounded_video_agent.domain import (
    CapabilityStatus,
    EvidenceItem,
    EvidenceModality,
    OCRObservation,
    OCRSpan,
    TimeRange,
)
from grounded_video_agent.workspace.catalog import CatalogError


class ReadScreenTextTool:
    name = "read_screen_text"
    description = (
        "Run OCR on existing sampled frames or on selected timeline targets and return "
        "time-aligned on-screen text evidence."
    )
    input_type = ReadScreenTextInput

    def __init__(self, operation: OCROperation, *, enabled: bool = True) -> None:
        self._operation = operation
        self.enabled = enabled

    def execute(
        self,
        request: ReadScreenTextInput,
        runtime: ToolRuntimeContext,
    ) -> ToolResult[ScreenTextOutput]:
        call_id, early = start_tool(runtime, self.name)
        if early is not None:
            return cast(ToolResult[ScreenTextOutput], early)
        assert call_id is not None
        try:
            targets = (
                resolve_targets(
                    runtime,
                    candidate_ids=request.candidate_ids,
                    context_window_ids=request.context_window_ids,
                    chunk_ids=request.chunk_ids,
                    ranges=request.ranges,
                )
                if any(
                    (
                        request.candidate_ids,
                        request.context_window_ids,
                        request.chunk_ids,
                        request.ranges,
                    )
                )
                else ()
            )
        except (CatalogError, KeyError) as error:
            return cast(
                ToolResult[ScreenTextOutput],
                failed_result(call_id, "OCR_TARGET_RESOLUTION_FAILED", str(error)),
            )
        run = self._operation.read(
            runtime,
            call_id,
            targets,
            request.detail,
            frame_ids=request.frame_ids,
            language=request.language,
            min_confidence=request.min_confidence,
        )
        if run.status is CapabilityStatus.FAILED:
            assert run.error is not None
            return cast(
                ToolResult[ScreenTextOutput],
                failed_result(
                    call_id,
                    run.error.code,
                    run.error.message,
                    retryable=run.error.retryable,
                    usage=run.usage,
                ),
            )
        assert run.frames is not None
        frame_by_id = {frame.frame_id: frame for frame in run.frames.frames}
        spans: list[ScreenTextSpan] = []
        new_ids: list[str] = []
        reused_ids: list[str] = []
        if run.ocr is not None:
            observation_by_id = {
                item.observation_id: item for item in run.ocr.observations
            }
            source_spans = run.ocr.spans or tuple(
                self._observation_span(item) for item in run.ocr.observations
            )
            for item in source_spans:
                frame_ids = tuple(
                    dict.fromkeys(
                        observation_by_id[observation_id].frame_id
                        for observation_id in item.observation_ids
                        if observation_id in observation_by_id
                    )
                )
                evidence_id = stable_id(
                    "evidence",
                    (
                        runtime.video_id,
                        EvidenceModality.OCR,
                        item.time_range,
                        item.text,
                        tuple(
                            frame_by_id[frame_id].timestamp_ms
                            for frame_id in frame_ids
                            if frame_id in frame_by_id
                        ),
                    ),
                )
                evidence = EvidenceItem(
                    evidence_id,
                    runtime.video_id,
                    item.time_range,
                    EvidenceModality.OCR,
                    frame_ids or item.observation_ids,
                    text=item.text,
                    artifacts=tuple(
                        frame_by_id[frame_id].image
                        for frame_id in frame_ids
                        if frame_id in frame_by_id
                    ),
                    confidence=item.confidence,
                )
                (new_ids if runtime.evidence.add(evidence) else reused_ids).append(evidence_id)
                spans.append(
                    ScreenTextSpan(
                        evidence_id,
                        item.text,
                        item.time_range,
                        frame_ids,
                        item.confidence,
                    )
                )
        inspected = (
            run.frames.requested_ranges
            if run.frames is not None
            else sampling_ranges(targets)
        )
        covered = runtime.coverage.add(inspected) if run.frames.frames else ()
        gained = bool(new_ids or covered)
        runtime.record_information_gain(gained)
        runtime.record_usage(run.usage)
        output = ScreenTextOutput(
            inspected,
            tuple(spans),
            tuple(frame_by_id),
            run.reused_frames,
            run.reused_ocr,
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
                cache_hit=run.reused_ocr,
                no_information_gain=not gained,
            ),
            run.warnings,
            usage=run.usage,
        )

    @staticmethod
    def _observation_span(item: OCRObservation) -> OCRSpan:
        return OCRSpan(
            f"single_{item.observation_id}",
            item.video_id,
            TimeRange(item.timestamp_ms, item.timestamp_ms + 1),
            item.normalized_text,
            (item.observation_id,),
            item.confidence,
        )
