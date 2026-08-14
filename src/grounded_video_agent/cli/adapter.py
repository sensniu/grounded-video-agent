from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from grounded_video_agent.agent import AgentLimits, AgentRequest, AgentResult, ProgressSink


class AgentInvoker(Protocol):
    def invoke(
        self,
        request: AgentRequest,
        *,
        progress: ProgressSink | None = None,
    ) -> AgentResult: ...


@dataclass(frozen=True, slots=True)
class AnalyzeOptions:
    filename: str
    question: str
    response_language: str = "zh-CN"
    evidence_clip_requested: bool = False
    force_refresh: bool = False
    request_id: str | None = None
    max_iterations: int | None = None
    max_tool_calls: int | None = None
    max_llm_calls: int | None = None
    max_total_tokens: int | None = None


def invoke_agent(
    agent: AgentInvoker,
    options: AnalyzeOptions,
    *,
    progress: ProgressSink | None = None,
) -> AgentResult:
    default_limits = AgentLimits()
    limits = AgentLimits(
        max_iterations=options.max_iterations or default_limits.max_iterations,
        max_tool_calls=options.max_tool_calls or default_limits.max_tool_calls,
        max_llm_calls=options.max_llm_calls or default_limits.max_llm_calls,
        max_total_tokens=options.max_total_tokens or default_limits.max_total_tokens,
        max_replans=default_limits.max_replans,
        max_remediations=default_limits.max_remediations,
        min_global_coverage=default_limits.min_global_coverage,
    )
    if options.request_id is None:
        request = AgentRequest(
            filename=options.filename,
            question=options.question,
            response_language=options.response_language,
            evidence_clip_requested=options.evidence_clip_requested,
            force_refresh=options.force_refresh,
            limits=limits,
        )
    else:
        request = AgentRequest(
            filename=options.filename,
            question=options.question,
            request_id=options.request_id,
            response_language=options.response_language,
            evidence_clip_requested=options.evidence_clip_requested,
            force_refresh=options.force_refresh,
            limits=limits,
        )
    return agent.invoke(request, progress=progress)
