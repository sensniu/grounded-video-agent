from __future__ import annotations

import json
from typing import Any, cast

import pytest

from grounded_video_agent.agent import (
    AgentProgressEvent,
    AgentRequest,
    AgentStatus,
    ProgressPhase,
    ProgressStatus,
    VideoAgent,
)
from grounded_video_agent.agent.orchestration import AgentDependencies
from grounded_video_agent.agent.reasoning import AgentReasoningService
from grounded_video_agent.agent.tools import (
    DeliveryState,
    EvidenceClipDelivery,
    EvidenceDelta,
    ExportEvidenceClipInput,
    ExportEvidenceClipOutput,
    GetVideoMetadataInput,
    SearchVideoTranscriptInput,
    ToolProgress,
    ToolResult,
    ToolRuntimeContext,
    ToolStatus,
    TranscriptCandidate,
    TranscriptSearchOutput,
    VideoMetadataOutput,
    VideoToolSuite,
)
from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    EvidenceItem,
    EvidenceModality,
    TimeRange,
)
from grounded_video_agent.infrastructure.llm import (
    LLMFinishReason,
    LLMModelInfo,
    LLMOutputFormat,
    LLMRequest,
    LLMResponse,
    LLMUsage,
)
from grounded_video_agent.pipelines import (
    PipelineReadiness,
    PipelineStatus,
    PreprocessingRequest,
    PreprocessingResult,
)
from grounded_video_agent.workspace.catalog import ArtifactCatalog


class _ScriptedLLM:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = outputs
        self.requests: list[LLMRequest] = []

    def get_model_info(self) -> LLMModelInfo:
        return LLMModelInfo("fake", "scripted", True)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        output = self.outputs.pop(0)
        content = json.dumps(output)
        return LLMResponse(
            operation_id=request.operation_id,
            response_id=f"response-{len(self.requests)}",
            provider="fake",
            model="scripted",
            output_format=LLMOutputFormat.JSON_OBJECT,
            content=content,
            json_object=output,
            finish_reason=LLMFinishReason.STOP,
            usage=LLMUsage(50, 20, 70),
            latency_ms=1,
        )


class _FakePipeline:
    def run(self, request: PreprocessingRequest | str) -> PreprocessingResult:
        return PreprocessingResult(
            "preprocess-test",
            PipelineStatus.READY,
            "video-1",
            1,
            PipelineReadiness(
                media_ready=True,
                shots_ready=True,
                transcript_ready=True,
                timeline_ready=True,
                sparse_search_ready=True,
            ),
            (),
        )


class _MetadataTool:
    name = "get_video_metadata"
    description = "Read video metadata."
    input_type = GetVideoMetadataInput
    enabled = True

    def execute(
        self,
        request: GetVideoMetadataInput,
        runtime: ToolRuntimeContext,
    ) -> ToolResult[VideoMetadataOutput]:
        call_id = runtime.start_call(self.name)
        return ToolResult(
            "1",
            call_id,
            ToolStatus.SUCCESS,
            VideoMetadataOutput(
                "video-1",
                "video.mp4",
                10_000,
                ("mp4",),
                1920,
                1080,
                25.0,
                "valid",
                True,
                "proceed",
                True,
                False,
                True,
                True,
                True,
                False,
            ),
        )


class _SearchTool:
    name = "search_video_transcript"
    description = "Search transcript chunks."
    input_type = SearchVideoTranscriptInput
    enabled = True

    def execute(
        self,
        request: SearchVideoTranscriptInput,
        runtime: ToolRuntimeContext,
    ) -> ToolResult[TranscriptSearchOutput]:
        call_id = runtime.start_call(self.name)
        time_range = TimeRange(1_000, 5_000)
        evidence = EvidenceItem(
            "evidence-1",
            runtime.video_id,
            time_range,
            EvidenceModality.TRANSCRIPT,
            ("chunk-1",),
            text="The person opens the door and enters the room.",
        )
        runtime.evidence.add(evidence)
        runtime.coverage.add((time_range,))
        runtime.record_information_gain(True)
        candidate = TranscriptCandidate(
            "candidate-1",
            "chunk-1",
            evidence.text or "",
            time_range,
            time_range,
            ("shot-1",),
            evidence.evidence_id,
            (("bm25", 1.0),),
            (request.query,),
            False,
        )
        return ToolResult(
            "1",
            call_id,
            ToolStatus.SUCCESS,
            TranscriptSearchOutput(request.query, "sparse", (candidate,), (), True),
            EvidenceDelta((evidence.evidence_id,), ()),
            ToolProgress(new_candidate_count=1, new_evidence_count=1, exhausted=True),
        )


class _ExportTool:
    name = "export_evidence_clip"
    description = "Export a verified evidence clip."
    input_type = ExportEvidenceClipInput
    enabled = True
    runtime_guarded = True

    def is_available(self, runtime: ToolRuntimeContext) -> bool:
        return runtime.delivery_policy.evidence_clip_requested and bool(
            runtime.delivery_policy.verified_evidence_ids
        )

    def execute(
        self,
        request: ExportEvidenceClipInput,
        runtime: ToolRuntimeContext,
    ) -> ToolResult[ExportEvidenceClipOutput]:
        call_id = runtime.start_call(self.name)
        assert runtime.delivery_policy.permits(request.evidence_ids)
        time_range = TimeRange(500, 5_500)
        artifact = ArtifactRef(
            "artifact-clip-1",
            ArtifactKind.VIDEO_CLIP,
            "artifacts/video-1/evidence.mp4",
        )
        runtime.deliveries.put(
            DeliveryState("delivery-1", "entry-1", artifact, request.evidence_ids)
        )
        delivery = EvidenceClipDelivery(
            "delivery-1",
            artifact.artifact_id,
            "entry-1",
            "evidence.mp4",
            time_range,
            time_range,
            time_range.duration_ms,
            True,
            request.evidence_ids,
            1_024,
        )
        return ToolResult(
            "1",
            call_id,
            ToolStatus.SUCCESS,
            ExportEvidenceClipOutput("export-1", (delivery,), total_duration_ms=5_000),
            EvidenceDelta(reused_evidence_ids=request.evidence_ids),
        )


def _decision(
    action: str,
    *,
    tool_name: str | None = None,
    tool_arguments: dict[str, Any] | None = None,
    evidence_ids: list[str] | None = None,
    final_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "intent": "local_event",
        "current_goal": "Answer what the person does",
        "missing_information": [] if action != "call_tool" else ["Relevant segment"],
        "action": action,
        "tool_name": tool_name,
        "tool_arguments": tool_arguments or {},
        "supporting_evidence_ids": evidence_ids or [],
        "expected_information_gain": (
            "Find a transcript segment" if action == "call_tool" else None
        ),
        "final_reason": final_reason,
    }


@pytest.mark.asyncio
async def test_agent_runs_search_verify_and_authorized_delivery() -> None:
    llm = _ScriptedLLM(
        [
            _decision(
                "call_tool",
                tool_name="search_video_transcript",
                tool_arguments={"query": "opens the door", "top_k": 3},
            ),
            _decision("answer", evidence_ids=["evidence-1"]),
            {
                "answer": "这个人打开门并进入房间。",
                "claims": [
                    {
                        "text": "这个人打开门并进入房间。",
                        "evidence_ids": ["evidence-1"],
                        "confidence": 0.9,
                        "uncertainties": [],
                    }
                ],
                "limitations": [],
            },
        ]
    )
    catalog = cast(ArtifactCatalog, object())
    dependencies = AgentDependencies(
        pipeline=_FakePipeline(),
        catalog=catalog,
        tools=VideoToolSuite((_MetadataTool(), _SearchTool(), _ExportTool())),
        reasoning=AgentReasoningService(llm),
    )
    agent = VideoAgent(dependencies)

    progress_events: list[AgentProgressEvent] = []
    result = await agent.ainvoke(
        AgentRequest(
            "video.mp4",
            "这个人进入房间前做了什么？",
            request_id="agent-test-1",
            evidence_clip_requested=True,
        ),
        progress=progress_events.append,
    )

    assert result.status is AgentStatus.SUCCESS
    assert result.answer == "这个人打开门并进入房间。"
    assert result.claims[0].evidence_ids == ("evidence-1",)
    assert result.citations[0].time_range == TimeRange(1_000, 5_000)
    assert result.attachments[0].attachment_id == "delivery-1"
    assert result.usage.llm_calls == 3
    assert result.usage.tool_calls == 3
    assert not llm.outputs
    assert progress_events[0].phase is ProgressPhase.INITIALIZING
    assert progress_events[-1].phase is ProgressPhase.COMPLETE
    assert progress_events[-1].status is ProgressStatus.COMPLETED
    assert any(
        event.phase is ProgressPhase.TOOL and event.status is ProgressStatus.STARTED
        for event in progress_events
    )


@pytest.mark.asyncio
async def test_agent_rejects_duplicate_tool_call_and_allows_abstention() -> None:
    repeated = _decision(
        "call_tool",
        tool_name="search_video_transcript",
        tool_arguments={"query": "opens the door", "top_k": 3},
    )
    llm = _ScriptedLLM(
        [
            repeated,
            repeated,
            _decision("abstain", final_reason="No non-duplicate action remains."),
        ]
    )
    dependencies = AgentDependencies(
        pipeline=_FakePipeline(),
        catalog=cast(ArtifactCatalog, object()),
        tools=VideoToolSuite((_MetadataTool(), _SearchTool())),
        reasoning=AgentReasoningService(llm),
    )

    result = await VideoAgent(dependencies).ainvoke(
        AgentRequest("video.mp4", "What happened?", request_id="agent-test-2")
    )

    assert result.status is AgentStatus.ABSTAINED
    assert result.answer == "No non-duplicate action remains."
    assert result.usage.tool_calls == 2
