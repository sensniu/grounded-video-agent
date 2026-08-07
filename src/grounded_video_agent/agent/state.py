"""Checkpointable state carried by the LangGraph orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from grounded_video_agent.agent.contracts import (
    AgentAttachment,
    AgentCitation,
    AgentClaim,
    AgentDecision,
    AgentRequest,
    AgentResult,
    AgentStatus,
    AnswerDraft,
    QuestionIntent,
)
from grounded_video_agent.agent.tools import ToolRuntimeSnapshot
from grounded_video_agent.domain import (
    CapabilityUsage,
    EvidenceBundle,
    EvidenceVerificationReport,
)
from grounded_video_agent.pipelines import PreprocessingResult


@dataclass(frozen=True, slots=True)
class AgentToolEvent:
    tool_name: str
    arguments: dict[str, Any]
    call_id: str
    status: str
    result: dict[str, Any]
    new_evidence_ids: tuple[str, ...] = ()
    reused_evidence_ids: tuple[str, ...] = ()
    no_information_gain: bool = False
    error_code: str | None = None
    error_message: str | None = None
    usage: CapabilityUsage = CapabilityUsage()


class AgentState(TypedDict):
    request: AgentRequest
    run_id: str
    phase: str
    route: str
    video_id: str | None
    preprocessing: PreprocessingResult | None
    metadata: dict[str, Any] | None
    intent: QuestionIntent | None
    decision: AgentDecision | None
    planner_feedback: tuple[str, ...]
    tool_events: tuple[AgentToolEvent, ...]
    runtime_snapshot: ToolRuntimeSnapshot | None
    evidence_bundle: EvidenceBundle | None
    draft: AnswerDraft | None
    verification: EvidenceVerificationReport | None
    claims: tuple[AgentClaim, ...]
    citations: tuple[AgentCitation, ...]
    attachments: tuple[AgentAttachment, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    iterations: int
    replans: int
    remediations: int
    invalid_decisions: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    capability_model_calls: int
    status: AgentStatus | None
    answer: str | None
    abstain_reason: str | None
    result: AgentResult | None


def initial_agent_state(request: AgentRequest) -> AgentState:
    return AgentState(
        request=request,
        run_id=request.request_id,
        phase="created",
        route="bootstrap",
        video_id=None,
        preprocessing=None,
        metadata=None,
        intent=None,
        decision=None,
        planner_feedback=(),
        tool_events=(),
        runtime_snapshot=None,
        evidence_bundle=None,
        draft=None,
        verification=None,
        claims=(),
        citations=(),
        attachments=(),
        warnings=(),
        errors=(),
        iterations=0,
        replans=0,
        remediations=0,
        invalid_decisions=0,
        llm_calls=0,
        input_tokens=0,
        output_tokens=0,
        capability_model_calls=0,
        status=None,
        answer=None,
        abstain_reason=None,
        result=None,
    )
