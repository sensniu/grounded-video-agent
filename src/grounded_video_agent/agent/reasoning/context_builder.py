"""Compact, path-free contexts sent to the reasoning model."""

from __future__ import annotations

import json
from typing import Any

from grounded_video_agent.agent.state import AgentState
from grounded_video_agent.agent.tools import ToolSpec
from grounded_video_agent.capabilities._support import json_value

from .contracts import AnswerContext, PlanningContext


def build_planning_context(
    state: AgentState,
    tool_specs: tuple[ToolSpec, ...],
) -> PlanningContext:
    snapshot = state["runtime_snapshot"]
    evidence = () if snapshot is None else snapshot.evidence
    events = state["tool_events"][-6:]
    payload: dict[str, Any] = {
        "question": state["request"].question,
        "response_language": state["request"].response_language,
        "video": state["metadata"],
        "known_intent": state["intent"].value if state["intent"] is not None else None,
        "available_tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
            }
            for spec in tool_specs
        ],
        "evidence": [_evidence_summary(item) for item in evidence[-24:]],
        "recent_tool_results": [
            {
                "tool_name": event.tool_name,
                "call_id": event.call_id,
                "status": event.status,
                "new_evidence_ids": event.new_evidence_ids,
                "no_information_gain": event.no_information_gain,
                "error_code": event.error_code,
                "result": _bounded_json(event.result, 6_000),
            }
            for event in events
        ],
        "planner_feedback": state["planner_feedback"][-6:],
        "progress": {
            "iterations": state["iterations"],
            "tool_calls": snapshot.call_count if snapshot is not None else 0,
            "llm_calls": state["llm_calls"],
            "covered_ranges": (
                json_value(snapshot.coverage_ranges) if snapshot is not None else []
            ),
            "consecutive_no_information_gain": (
                snapshot.consecutive_no_information_gain if snapshot is not None else 0
            ),
        },
    }
    return PlanningContext(payload)


def build_answer_context(state: AgentState) -> AnswerContext:
    bundle = state["evidence_bundle"]
    decision = state["decision"]
    selected_ids = (
        set(decision.supporting_evidence_ids) if decision is not None else set()
    )
    evidence = (
        []
        if bundle is None
        else [
            _evidence_summary(item)
            for item in bundle.items
            if not selected_ids or item.evidence_id in selected_ids
        ]
    )
    return AnswerContext(
        {
            "question": state["request"].question,
            "response_language": state["request"].response_language,
            "intent": state["intent"].value if state["intent"] is not None else None,
            "video": state["metadata"],
            "evidence": evidence,
            "required_evidence_ids": (
                decision.supporting_evidence_ids if decision is not None else ()
            ),
        }
    )


def _evidence_summary(item: object) -> dict[str, Any]:
    value = json_value(item)
    assert isinstance(value, dict)
    value.pop("artifacts", None)
    text = value.get("text")
    if isinstance(text, str) and len(text) > 1_500:
        value["text"] = text[:1_497] + "..."
    return value


def _bounded_json(value: object, max_characters: int) -> object:
    normalized = json_value(value)
    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= max_characters:
        return normalized
    return {"truncated_json": encoded[: max_characters - 3] + "..."}
