from __future__ import annotations

from dataclasses import replace
from typing import Any

from grounded_video_agent.agent.contracts import (
    AgentAction,
    AgentAttachment,
    AgentError,
    AgentResult,
    AgentStatus,
    AgentUsage,
    QuestionIntent,
)
from grounded_video_agent.agent.reasoning import (
    AgentReasoningError,
    build_answer_context,
    build_planning_context,
)
from grounded_video_agent.agent.state import AgentState, AgentToolEvent
from grounded_video_agent.agent.tools import (
    DeliveryPolicy,
    ExportEvidenceClipOutput,
    ToolResult,
    ToolRuntimeContext,
    ToolStatus,
)
from grounded_video_agent.domain import (
    EvidenceModality,
    EvidenceVerificationStatus,
    ResourceLimits,
)
from grounded_video_agent.infrastructure.llm import LLMResponse
from grounded_video_agent.observability import emit_trace
from grounded_video_agent.pipelines import (
    PipelineStatus,
    PreprocessingRequest,
)

from .dependencies import AgentDependencies

_VISUAL_MODALITIES = {
    EvidenceModality.FRAME,
    EvidenceModality.VISUAL_DESCRIPTION,
    EvidenceModality.VLM_OBSERVATION,
}


class AgentNodes:
    def __init__(self, dependencies: AgentDependencies) -> None:
        self._dependencies = dependencies

    async def bootstrap(self, state: AgentState) -> dict[str, Any]:
        request = state["request"]
        preprocessing_request = PreprocessingRequest(
            request.filename,
            force_refresh=request.force_refresh,
            trace_id=request.trace_id or request.request_id,
        )
        emit_trace(
            "pipeline.started",
            {"request": preprocessing_request},
            operation_id=request.request_id,
            phase="preprocessing",
        )
        try:
            preprocessing = self._dependencies.pipeline.run(preprocessing_request)
        except Exception as error:
            emit_trace(
                "pipeline.failed",
                {"request": preprocessing_request, "error": error},
                operation_id=request.request_id,
                phase="preprocessing",
            )
            raise
        warnings = (*state["warnings"], *preprocessing.warnings)
        if preprocessing.status is PipelineStatus.FAILED or preprocessing.video_id is None:
            message = (
                preprocessing.error.message
                if preprocessing.error is not None
                else "Video preprocessing failed."
            )
            return {
                "phase": "bootstrap_failed",
                "route": "finalize",
                "preprocessing": preprocessing,
                "video_id": preprocessing.video_id,
                "warnings": warnings,
                "errors": (*state["errors"], message),
                "status": AgentStatus.FAILED,
            }
        runtime = ToolRuntimeContext(
            video_id=preprocessing.video_id,
            catalog=self._dependencies.catalog,
            trace_id=request.trace_id or request.request_id,
            limits=ResourceLimits(
                max_model_calls=request.limits.max_llm_calls,
                max_tokens=request.limits.max_total_tokens,
            ),
            max_tool_calls=request.limits.max_tool_calls,
            delivery_policy=DeliveryPolicy(
                evidence_clip_requested=request.evidence_clip_requested
            ),
        )
        self._dependencies.runtimes.put(state["run_id"], runtime)
        metadata_result = self._dependencies.tools.invoke(
            "get_video_metadata",
            {},
            runtime,
        )
        event = _tool_event("get_video_metadata", {}, metadata_result)
        metadata = (
            metadata_result.to_dict()["data"]
            if metadata_result.status is not ToolStatus.FAILED
            else None
        )
        if metadata_result.status is ToolStatus.FAILED:
            message = (
                metadata_result.error.message
                if metadata_result.error is not None
                else "Video metadata could not be loaded."
            )
            return {
                "phase": "bootstrap_failed",
                "route": "finalize",
                "video_id": preprocessing.video_id,
                "preprocessing": preprocessing,
                "metadata": None,
                "tool_events": (*state["tool_events"], event),
                "runtime_snapshot": runtime.snapshot(),
                "warnings": warnings,
                "errors": (*state["errors"], message),
                "capability_model_calls": (
                    state["capability_model_calls"] + metadata_result.usage.model_calls
                ),
                "status": AgentStatus.FAILED,
            }
        return {
            "phase": "ready",
            "route": "plan",
            "video_id": preprocessing.video_id,
            "preprocessing": preprocessing,
            "metadata": metadata,
            "tool_events": (*state["tool_events"], event),
            "runtime_snapshot": runtime.snapshot(),
            "warnings": (*warnings, *metadata_result.warnings),
            "capability_model_calls": (
                state["capability_model_calls"] + metadata_result.usage.model_calls
            ),
        }

    async def plan(self, state: AgentState) -> dict[str, Any]:
        limit_reason = _planning_limit_reason(state)
        if limit_reason is not None:
            return {
                "phase": "budget_exhausted",
                "route": "finalize",
                "status": AgentStatus.ABSTAINED,
                "abstain_reason": limit_reason,
            }
        runtime = self._runtime(state)
        specs = tuple(
            spec
            for spec in self._dependencies.tools.available_specs_for(runtime)
            if spec.name != "export_evidence_clip"
        )
        context = build_planning_context(state, specs)
        try:
            reasoning = await self._dependencies.reasoning.plan(
                context,
                operation_id=f"{state['run_id']}_plan_{state['iterations'] + 1:04d}",
                trace_id=state["request"].trace_id or state["request"].request_id,
            )
        except AgentReasoningError as error:
            return {
                "phase": "planning_failed",
                "route": "finalize",
                "errors": (*state["errors"], str(error)),
                "status": AgentStatus.FAILED,
            }
        usage = _reasoning_usage(reasoning.responses)
        budget_error = _post_call_budget_error(state, usage)
        if budget_error is not None:
            return {
                "phase": "budget_exhausted",
                "route": "finalize",
                "iterations": state["iterations"] + 1,
                "llm_calls": state["llm_calls"] + usage[0],
                "input_tokens": state["input_tokens"] + usage[1],
                "output_tokens": state["output_tokens"] + usage[2],
                "status": AgentStatus.ABSTAINED,
                "abstain_reason": budget_error,
            }
        return {
            "phase": "planned",
            "route": "guard",
            "decision": reasoning.data,
            "iterations": state["iterations"] + 1,
            "llm_calls": state["llm_calls"] + usage[0],
            "input_tokens": state["input_tokens"] + usage[1],
            "output_tokens": state["output_tokens"] + usage[2],
        }

    async def guard(self, state: AgentState) -> dict[str, Any]:
        decision = state["decision"]
        if decision is None:
            return self._invalid_decision(state, "Planner returned no decision.")
        if state["intent"] is not None and decision.intent is not state["intent"]:
            return self._invalid_decision(
                state,
                f"Intent cannot change from {state['intent'].value} to {decision.intent.value}.",
            )
        intent_update = decision.intent if state["intent"] is None else state["intent"]
        if decision.action is AgentAction.ABSTAIN:
            return {
                "phase": "abstained",
                "route": "finalize",
                "intent": intent_update,
                "status": AgentStatus.ABSTAINED,
                "abstain_reason": decision.final_reason,
            }
        if decision.action is AgentAction.ANSWER:
            return {
                "phase": "answer_requested",
                "route": "answer_gate",
                "intent": intent_update,
            }
        runtime = self._runtime(state)
        available = {
            spec.name for spec in self._dependencies.tools.available_specs_for(runtime)
        }
        tool_name = decision.tool_name or ""
        if tool_name == "export_evidence_clip":
            return self._invalid_decision(
                state,
                "Evidence clips are exported only by the verified delivery stage.",
            )
        if tool_name not in available:
            return self._invalid_decision(
                state,
                f"Tool is unavailable or unknown: {tool_name}.",
            )
        if runtime.call_count >= runtime.max_tool_calls:
            return self._invalid_decision(
                state,
                "Tool budget is exhausted; answer with current evidence or abstain.",
            )
        if any(
            event.tool_name == tool_name and event.arguments == decision.tool_arguments
            for event in state["tool_events"]
        ):
            return self._invalid_decision(
                state,
                "The exact same tool call has already been executed; choose a new query, "
                "target, or action.",
            )
        return {
            "phase": "tool_authorized",
            "route": "execute_tool",
            "intent": intent_update,
        }

    async def execute_tool(self, state: AgentState) -> dict[str, Any]:
        decision = state["decision"]
        if decision is None or decision.tool_name is None:
            return self._invalid_decision(state, "No tool call is available to execute.")
        runtime = self._runtime(state)
        result = self._dependencies.tools.invoke(
            decision.tool_name,
            decision.tool_arguments,
            runtime,
        )
        event = _tool_event(decision.tool_name, decision.tool_arguments, result)
        feedback: list[str] = []
        replans = state["replans"]
        if result.status is ToolStatus.FAILED:
            message = result.error.message if result.error is not None else "Tool call failed."
            feedback.append(f"{decision.tool_name} failed: {message}")
        elif result.progress.no_information_gain:
            feedback.append(
                f"{decision.tool_name} produced no new information; do not repeat the same call."
            )
        if runtime.requires_replan:
            feedback.append(
                "Two consecutive tool calls produced no information gain; substantially replan."
            )
            replans += 1
        return {
            "phase": "tool_observed",
            "route": "plan",
            "tool_events": (*state["tool_events"], event),
            "runtime_snapshot": runtime.snapshot(),
            "planner_feedback": (*state["planner_feedback"], *feedback),
            "warnings": (*state["warnings"], *result.warnings),
            "replans": replans,
            "capability_model_calls": (
                state["capability_model_calls"] + result.usage.model_calls
            ),
        }

    async def answer_gate(self, state: AgentState) -> dict[str, Any]:
        runtime = self._runtime(state)
        bundle = runtime.build_evidence_bundle(state["request"].question)
        missing = _preliminary_missing(state, bundle, runtime)
        if missing:
            return self._remediate_or_abstain(state, missing, bundle=bundle)
        return {
            "phase": "answer_allowed",
            "route": "draft_answer",
            "evidence_bundle": bundle,
            "runtime_snapshot": runtime.snapshot(),
        }

    async def draft_answer(self, state: AgentState) -> dict[str, Any]:
        try:
            reasoning = await self._dependencies.reasoning.draft_answer(
                build_answer_context(state),
                operation_id=f"{state['run_id']}_answer_{state['remediations'] + 1:04d}",
                trace_id=state["request"].trace_id or state["request"].request_id,
            )
        except AgentReasoningError as error:
            return {
                "phase": "answer_generation_failed",
                "route": "finalize",
                "errors": (*state["errors"], str(error)),
                "status": AgentStatus.FAILED,
            }
        usage = _reasoning_usage(reasoning.responses)
        budget_error = _post_call_budget_error(state, usage)
        if budget_error is not None:
            return {
                "phase": "budget_exhausted",
                "route": "finalize",
                "llm_calls": state["llm_calls"] + usage[0],
                "input_tokens": state["input_tokens"] + usage[1],
                "output_tokens": state["output_tokens"] + usage[2],
                "status": AgentStatus.ABSTAINED,
                "abstain_reason": budget_error,
            }
        return {
            "phase": "answer_drafted",
            "route": "verify",
            "draft": reasoning.data,
            "llm_calls": state["llm_calls"] + usage[0],
            "input_tokens": state["input_tokens"] + usage[1],
            "output_tokens": state["output_tokens"] + usage[2],
        }

    async def verify(self, state: AgentState) -> dict[str, Any]:
        draft = state["draft"]
        bundle = state["evidence_bundle"]
        video_id = state["video_id"]
        intent = state["intent"]
        if draft is None or bundle is None or video_id is None or intent is None:
            return {
                "phase": "verification_failed",
                "route": "finalize",
                "errors": (*state["errors"], "Verification inputs are incomplete."),
                "status": AgentStatus.FAILED,
            }
        runtime = self._runtime(state)
        outcome = self._dependencies.verifier.verify(
            draft,
            bundle,
            video_id=video_id,
            intent=intent,
            duration_ms=_duration_ms(state),
            min_global_coverage=state["request"].limits.min_global_coverage,
            coverage_ranges=runtime.coverage.ranges,
        )
        if outcome.report.status is not EvidenceVerificationStatus.SUFFICIENT:
            missing = outcome.report.missing_evidence or (
                "Evidence verification did not approve the answer.",
            )
            return self._remediate_or_abstain(
                state,
                missing,
                bundle=bundle,
                verification=outcome.report,
            )
        policy = replace(
            runtime.delivery_policy,
            verified_evidence_ids=frozenset(outcome.verified_evidence_ids),
        )
        runtime.delivery_policy = policy
        return {
            "phase": "verified",
            "route": "deliver",
            "verification": outcome.report,
            "claims": outcome.claims,
            "citations": outcome.citations,
            "runtime_snapshot": runtime.snapshot(),
            "answer": draft.answer,
        }

    async def deliver(self, state: AgentState) -> dict[str, Any]:
        if not state["request"].evidence_clip_requested:
            return {
                "phase": "delivered",
                "route": "finalize",
                "status": AgentStatus.SUCCESS,
            }
        runtime = self._runtime(state)
        verified_ids = tuple(sorted(runtime.delivery_policy.verified_evidence_ids))
        if not verified_ids:
            return {
                "phase": "delivered_without_clip",
                "route": "finalize",
                "warnings": (
                    *state["warnings"],
                    "No time-based verified evidence was available for clip export.",
                ),
                "status": AgentStatus.PARTIAL,
            }
        arguments: dict[str, Any] = {"evidence_ids": verified_ids}
        result = self._dependencies.tools.invoke(
            "export_evidence_clip",
            arguments,
            runtime,
        )
        event = _tool_event("export_evidence_clip", arguments, result)
        attachments: tuple[AgentAttachment, ...] = ()
        status = AgentStatus.SUCCESS
        warnings = (*state["warnings"], *result.warnings)
        if result.status is ToolStatus.FAILED or not isinstance(
            result.data, ExportEvidenceClipOutput
        ):
            status = AgentStatus.PARTIAL
            message = result.error.message if result.error is not None else "Clip export failed."
            warnings = (*warnings, message)
        else:
            attachments = tuple(
                AgentAttachment(
                    item.delivery_id,
                    item.artifact_id,
                    item.filename,
                    item.evidence_ids,
                    item.actual_range,
                    item.size_bytes,
                )
                for item in result.data.clips
            )
            if result.status is ToolStatus.PARTIAL:
                status = AgentStatus.PARTIAL
        return {
            "phase": "delivered",
            "route": "finalize",
            "tool_events": (*state["tool_events"], event),
            "runtime_snapshot": runtime.snapshot(),
            "attachments": attachments,
            "warnings": warnings,
            "capability_model_calls": (
                state["capability_model_calls"] + result.usage.model_calls
            ),
            "status": status,
        }

    async def finalize(self, state: AgentState) -> dict[str, Any]:
        status = state["status"] or AgentStatus.ABSTAINED
        answer = state["answer"]
        error = None
        if status is AgentStatus.FAILED:
            answer = None
            error = AgentError(
                "AGENT_FAILED",
                state["errors"][-1] if state["errors"] else "Agent execution failed.",
            )
        elif status is AgentStatus.ABSTAINED and answer is None:
            answer = state["abstain_reason"] or "Available evidence is insufficient to answer."
        snapshot = state["runtime_snapshot"]
        result = AgentResult(
            request_id=state["request"].request_id,
            status=status,
            video_id=state["video_id"],
            answer=answer,
            claims=state["claims"],
            citations=state["citations"],
            attachments=state["attachments"],
            verification=state["verification"],
            usage=AgentUsage(
                llm_calls=state["llm_calls"],
                input_tokens=state["input_tokens"],
                output_tokens=state["output_tokens"],
                tool_calls=snapshot.call_count if snapshot is not None else 0,
                capability_model_calls=state["capability_model_calls"],
            ),
            warnings=tuple(dict.fromkeys(state["warnings"])),
            error=error,
        )
        return {"phase": "finished", "route": "end", "answer": answer, "result": result}

    def _runtime(self, state: AgentState) -> ToolRuntimeContext:
        snapshot = state["runtime_snapshot"]
        if snapshot is None:
            raise RuntimeError("tool runtime has not been initialized")
        return self._dependencies.runtimes.resolve(
            state["run_id"],
            snapshot,
            self._dependencies.catalog,
        )

    def _invalid_decision(self, state: AgentState, message: str) -> dict[str, Any]:
        replans = state["replans"] + 1
        if replans >= state["request"].limits.max_replans:
            return {
                "phase": "invalid_decision_limit",
                "route": "finalize",
                "planner_feedback": (*state["planner_feedback"], message),
                "invalid_decisions": state["invalid_decisions"] + 1,
                "replans": replans,
                "status": AgentStatus.ABSTAINED,
                "abstain_reason": "The planner repeatedly produced invalid or unsafe actions.",
            }
        return {
            "phase": "decision_rejected",
            "route": "plan",
            "planner_feedback": (*state["planner_feedback"], message),
            "invalid_decisions": state["invalid_decisions"] + 1,
            "replans": replans,
        }

    def _remediate_or_abstain(
        self,
        state: AgentState,
        missing: tuple[str, ...] | list[str],
        *,
        bundle: object,
        verification: object | None = None,
    ) -> dict[str, Any]:
        feedback = tuple(str(item) for item in missing)
        if state["remediations"] >= state["request"].limits.max_remediations:
            updates: dict[str, Any] = {
                "phase": "evidence_insufficient",
                "route": "finalize",
                "status": AgentStatus.ABSTAINED,
                "abstain_reason": "Evidence remained insufficient after remediation: "
                + "; ".join(feedback),
                "planner_feedback": (*state["planner_feedback"], *feedback),
            }
            if verification is not None:
                updates["verification"] = verification
            return updates
        updates = {
            "phase": "remediation_requested",
            "route": "plan",
            "planner_feedback": (*state["planner_feedback"], *feedback),
            "remediations": state["remediations"] + 1,
            "draft": None,
        }
        if hasattr(bundle, "bundle_id"):
            updates["evidence_bundle"] = bundle
        if verification is not None:
            updates["verification"] = verification
        return updates


def _tool_event(
    tool_name: str,
    arguments: dict[str, Any],
    result: ToolResult[Any],
) -> AgentToolEvent:
    return AgentToolEvent(
        tool_name=tool_name,
        arguments=arguments,
        call_id=result.call_id,
        status=result.status.value,
        result=result.to_dict(),
        new_evidence_ids=result.evidence_delta.new_evidence_ids,
        reused_evidence_ids=result.evidence_delta.reused_evidence_ids,
        no_information_gain=result.progress.no_information_gain,
        error_code=result.error.code if result.error is not None else None,
        error_message=result.error.message if result.error is not None else None,
        usage=result.usage,
    )


def _reasoning_usage(responses: tuple[LLMResponse, ...]) -> tuple[int, int, int]:
    return (
        sum(item.usage.model_calls for item in responses),
        sum(item.usage.input_tokens for item in responses),
        sum(item.usage.output_tokens for item in responses),
    )


def _planning_limit_reason(state: AgentState) -> str | None:
    limits = state["request"].limits
    if state["iterations"] >= limits.max_iterations:
        return "Maximum planning iterations were reached."
    if state["llm_calls"] >= limits.max_llm_calls:
        return "Maximum LLM calls were reached."
    if state["input_tokens"] + state["output_tokens"] >= limits.max_total_tokens:
        return "LLM token budget was exhausted."
    return None


def _post_call_budget_error(
    state: AgentState,
    usage: tuple[int, int, int],
) -> str | None:
    limits = state["request"].limits
    if state["llm_calls"] + usage[0] > limits.max_llm_calls:
        return "Maximum LLM calls were exceeded while repairing structured output."
    if state["input_tokens"] + state["output_tokens"] + usage[1] + usage[2] > (
        limits.max_total_tokens
    ):
        return "LLM token budget was exhausted."
    return None


def _preliminary_missing(
    state: AgentState,
    bundle: object,
    runtime: ToolRuntimeContext,
) -> tuple[str, ...]:
    decision = state["decision"]
    intent = state["intent"]
    if decision is None or intent is None or not hasattr(bundle, "items"):
        return ("Answer decision or evidence bundle is missing.",)
    items = tuple(bundle.items)
    known = {item.evidence_id: item for item in items}
    if intent is QuestionIntent.METADATA:
        return ()
    missing: list[str] = []
    if not decision.supporting_evidence_ids:
        missing.append("The answer decision selected no evidence.")
    unknown = tuple(
        item for item in decision.supporting_evidence_ids if item not in known
    )
    if unknown:
        missing.append(f"Unknown evidence IDs were selected: {', '.join(unknown)}.")
    selected = tuple(
        known[item] for item in decision.supporting_evidence_ids if item in known
    )
    modalities = {item.modality for item in selected}
    if intent is QuestionIntent.VISUAL and not modalities.intersection(_VISUAL_MODALITIES):
        missing.append("Visual evidence is required before answering this question.")
    if intent is QuestionIntent.SCREEN_TEXT and EvidenceModality.OCR not in modalities:
        missing.append("OCR evidence is required before answering this question.")
    if intent is QuestionIntent.CAUSAL and len(selected) < 2:
        missing.append("The causal answer requires corroborating evidence.")
    if intent in {QuestionIntent.GLOBAL, QuestionIntent.COUNT}:
        duration = _duration_ms(state)
        covered = sum(item.duration_ms for item in runtime.coverage.ranges)
        ratio = covered / duration if duration is not None and duration > 0 else 0.0
        if ratio < state["request"].limits.min_global_coverage:
            missing.append(
                f"Timeline coverage {ratio:.2f} is insufficient for a global or count answer."
            )
    return tuple(missing)


def _duration_ms(state: AgentState) -> int | None:
    metadata = state["metadata"]
    if metadata is None:
        return None
    value = metadata.get("duration_ms")
    return value if isinstance(value, int) and not isinstance(value, bool) else None
