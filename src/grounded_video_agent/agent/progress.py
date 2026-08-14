from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import Any

from grounded_video_agent.agent.contracts import AgentDecision, AgentRequest, AgentResult
from grounded_video_agent.agent.state import AgentState, AgentToolEvent


class ProgressPhase(StrEnum):
    INITIALIZING = "initializing"
    PREPROCESSING = "preprocessing"
    PLANNING = "planning"
    TOOL = "tool"
    EVIDENCE = "evidence"
    ANSWERING = "answering"
    VERIFYING = "verifying"
    DELIVERY = "delivery"
    COMPLETE = "complete"


class ProgressStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    INFO = "info"
    WARNING = "warning"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProgressCounters:
    iteration: int
    max_iterations: int
    llm_calls: int
    max_llm_calls: int
    tool_calls: int
    max_tool_calls: int
    input_tokens: int
    output_tokens: int
    max_total_tokens: int
    evidence_count: int = 0
    coverage_ratio: float | None = None
    capability_model_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class AgentProgressEvent:
    sequence: int
    run_id: str
    elapsed_ms: int
    phase: ProgressPhase
    status: ProgressStatus
    message: str
    counters: ProgressCounters
    tool_name: str | None = None
    call_id: str | None = None
    details: tuple[tuple[str, str], ...] = ()


ProgressSink = Callable[[AgentProgressEvent], None]


class AgentProgressTracker:
    """Translate graph node updates into bounded, presentation-neutral progress events."""

    _BUDGET_THRESHOLDS = (0.5, 0.75, 0.9)

    def __init__(self, request: AgentRequest, sink: ProgressSink) -> None:
        self._request = request
        self._sink = sink
        self._started = perf_counter()
        self._sequence = 0
        self._reported_budget_thresholds: set[float] = set()

    def start(self, state: AgentState) -> None:
        self._emit(
            state,
            ProgressPhase.INITIALIZING,
            ProgressStatus.STARTED,
            "开始分析视频。",
        )
        self._emit(
            state,
            ProgressPhase.PREPROCESSING,
            ProgressStatus.STARTED,
            "正在注册、检查并预处理媒体。",
        )

    def graph_update(
        self,
        node: str,
        update: Mapping[str, Any],
        state: AgentState,
    ) -> None:
        handlers = {
            "bootstrap": self._bootstrap,
            "plan": self._plan,
            "guard": self._guard,
            "execute_tool": self._execute_tool,
            "answer_gate": self._answer_gate,
            "draft_answer": self._draft_answer,
            "verify": self._verify,
            "deliver": self._deliver,
            "finalize": self._finalize,
        }
        handler = handlers.get(node)
        if handler is not None:
            handler(update, state)
        self._budget_warnings(state)

    def failed(self, state: AgentState, error: BaseException) -> None:
        self._emit(
            state,
            ProgressPhase.COMPLETE,
            ProgressStatus.FAILED,
            f"分析异常终止：{error}",
        )

    def _bootstrap(self, update: Mapping[str, Any], state: AgentState) -> None:
        if update.get("route") == "plan":
            details: list[tuple[str, str]] = []
            preprocessing = state.get("preprocessing")
            if preprocessing is not None:
                cache_hits = sum(
                    report.status.value == "cache_hit" for report in preprocessing.stages
                )
                details.extend(
                    (
                        ("pipeline_status", preprocessing.status.value),
                        ("stages", str(len(preprocessing.stages))),
                        ("cache_hits", str(cache_hits)),
                        ("transcript_ready", str(preprocessing.readiness.transcript_ready)),
                    )
                )
            self._emit(
                state,
                ProgressPhase.PREPROCESSING,
                ProgressStatus.COMPLETED,
                "媒体预处理完成。",
                details=tuple(details),
            )
            if preprocessing is not None and preprocessing.warnings:
                shown = "；".join(_truncate(item, 180) for item in preprocessing.warnings[:2])
                remaining = len(preprocessing.warnings) - 2
                suffix = f"；另有 {remaining} 条警告" if remaining > 0 else ""
                self._emit(
                    state,
                    ProgressPhase.PREPROCESSING,
                    ProgressStatus.WARNING,
                    f"预处理警告：{shown}{suffix}",
                )
            self._planning_started(state)
        else:
            self._emit(
                state,
                ProgressPhase.PREPROCESSING,
                ProgressStatus.FAILED,
                "媒体预处理未能完成。",
            )

    def _plan(self, update: Mapping[str, Any], state: AgentState) -> None:
        decision = state.get("decision")
        if update.get("route") == "guard" and isinstance(decision, AgentDecision):
            details: tuple[tuple[str, str], ...] = (
                ("intent", decision.intent.value),
                ("action", decision.action.value),
            )
            if decision.current_goal:
                details = (*details, ("goal", _truncate(decision.current_goal, 160)))
            self._emit(
                state,
                ProgressPhase.PLANNING,
                ProgressStatus.COMPLETED,
                f"第 {state['iterations']} 轮规划完成：{decision.action.value}。",
                tool_name=decision.tool_name,
                details=details,
            )
        else:
            reason = state.get("abstain_reason") or "规划停止。"
            self._emit(
                state,
                ProgressPhase.PLANNING,
                ProgressStatus.WARNING,
                str(reason),
            )

    def _guard(self, update: Mapping[str, Any], state: AgentState) -> None:
        route = update.get("route")
        decision = state.get("decision")
        if route == "execute_tool" and isinstance(decision, AgentDecision):
            self._emit(
                state,
                ProgressPhase.TOOL,
                ProgressStatus.STARTED,
                "正在执行 Agent 选择的工具。",
                tool_name=decision.tool_name,
                details=_argument_summary(decision.tool_arguments),
            )
        elif route == "answer_gate":
            self._emit(
                state,
                ProgressPhase.EVIDENCE,
                ProgressStatus.STARTED,
                "正在检查证据是否足以回答问题。",
            )
        elif route == "plan":
            feedback = state.get("planner_feedback", ())
            self._emit(
                state,
                ProgressPhase.PLANNING,
                ProgressStatus.WARNING,
                _truncate(str(feedback[-1]), 240) if feedback else "规划动作被拒绝，重新规划。",
            )
            self._planning_started(state)

    def _execute_tool(self, update: Mapping[str, Any], state: AgentState) -> None:
        event = _last_tool_event(state)
        if event is not None:
            status = (
                ProgressStatus.FAILED if event.status == "failed" else ProgressStatus.COMPLETED
            )
            message = "工具执行失败。" if status is ProgressStatus.FAILED else "工具执行完成。"
            details = (
                ("status", event.status),
                ("new_evidence", str(len(event.new_evidence_ids))),
                ("reused_evidence", str(len(event.reused_evidence_ids))),
                ("no_information_gain", str(event.no_information_gain)),
            )
            self._emit(
                state,
                ProgressPhase.TOOL,
                status,
                message,
                tool_name=event.tool_name,
                call_id=event.call_id,
                details=details,
            )
        self._planning_started(state)

    def _answer_gate(self, update: Mapping[str, Any], state: AgentState) -> None:
        if update.get("route") == "draft_answer":
            self._emit(
                state,
                ProgressPhase.EVIDENCE,
                ProgressStatus.COMPLETED,
                "证据门禁通过。",
            )
            self._emit(
                state,
                ProgressPhase.ANSWERING,
                ProgressStatus.STARTED,
                "正在生成证据约束的回答。",
            )
        elif update.get("route") == "plan":
            self._emit(
                state,
                ProgressPhase.EVIDENCE,
                ProgressStatus.WARNING,
                "证据仍不充分，进入补充检索。",
            )
            self._planning_started(state)

    def _draft_answer(self, update: Mapping[str, Any], state: AgentState) -> None:
        if update.get("route") == "verify":
            self._emit(
                state,
                ProgressPhase.ANSWERING,
                ProgressStatus.COMPLETED,
                "回答草稿生成完成。",
            )
            self._emit(
                state,
                ProgressPhase.VERIFYING,
                ProgressStatus.STARTED,
                "正在验证声明与证据引用。",
            )
        else:
            self._emit(
                state,
                ProgressPhase.ANSWERING,
                ProgressStatus.WARNING,
                str(state.get("abstain_reason") or "回答生成未完成。"),
            )

    def _verify(self, update: Mapping[str, Any], state: AgentState) -> None:
        route = update.get("route")
        if route == "deliver":
            self._emit(
                state,
                ProgressPhase.VERIFYING,
                ProgressStatus.COMPLETED,
                "证据验证通过。",
            )
            self._emit(
                state,
                ProgressPhase.DELIVERY,
                ProgressStatus.STARTED,
                "正在整理最终回答与附件。",
            )
        elif route == "plan":
            self._emit(
                state,
                ProgressPhase.VERIFYING,
                ProgressStatus.WARNING,
                "证据验证未通过，进入修复规划。",
            )
            self._planning_started(state)

    def _deliver(self, update: Mapping[str, Any], state: AgentState) -> None:
        self._emit(
            state,
            ProgressPhase.DELIVERY,
            ProgressStatus.COMPLETED,
            "回答交付准备完成。",
        )

    def _finalize(self, update: Mapping[str, Any], state: AgentState) -> None:
        result = state.get("result")
        if isinstance(result, AgentResult):
            status = (
                ProgressStatus.FAILED
                if result.status.value == "failed"
                else ProgressStatus.COMPLETED
            )
            self._emit(
                state,
                ProgressPhase.COMPLETE,
                status,
                f"分析结束：{result.status.value}。",
            )

    def _planning_started(self, state: AgentState) -> None:
        self._emit(
            state,
            ProgressPhase.PLANNING,
            ProgressStatus.STARTED,
            f"正在进行第 {state['iterations'] + 1} 轮规划。",
        )

    def _budget_warnings(self, state: AgentState) -> None:
        counters = _counters(state, self._request)
        ratio = counters.total_tokens / counters.max_total_tokens
        for threshold in self._BUDGET_THRESHOLDS:
            if ratio >= threshold and threshold not in self._reported_budget_thresholds:
                self._reported_budget_thresholds.add(threshold)
                self._emit(
                    state,
                    ProgressPhase.PLANNING,
                    ProgressStatus.WARNING,
                    f"LLM token 预算已使用 {ratio:.0%}。",
                )

    def _emit(
        self,
        state: AgentState,
        phase: ProgressPhase,
        status: ProgressStatus,
        message: str,
        *,
        tool_name: str | None = None,
        call_id: str | None = None,
        details: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self._sequence += 1
        event = AgentProgressEvent(
            self._sequence,
            self._request.request_id,
            round((perf_counter() - self._started) * 1_000),
            phase,
            status,
            message,
            _counters(state, self._request),
            tool_name,
            call_id,
            details,
        )
        try:
            self._sink(event)
        except Exception:
            # Progress is observational and must never change the Agent outcome.
            return


def _counters(state: AgentState, request: AgentRequest) -> ProgressCounters:
    snapshot = state.get("runtime_snapshot")
    evidence_count = len(snapshot.evidence) if snapshot is not None else 0
    tool_calls = snapshot.call_count if snapshot is not None else 0
    duration_ms = None
    metadata = state.get("metadata")
    if metadata is not None:
        value = metadata.get("duration_ms")
        duration_ms = value if isinstance(value, int) and not isinstance(value, bool) else None
    coverage_ratio = None
    if snapshot is not None and duration_ms is not None and duration_ms > 0:
        covered_ms = sum(item.duration_ms for item in snapshot.coverage_ranges)
        coverage_ratio = max(0.0, min(1.0, covered_ms / duration_ms))
    return ProgressCounters(
        iteration=state["iterations"],
        max_iterations=request.limits.max_iterations,
        llm_calls=state["llm_calls"],
        max_llm_calls=request.limits.max_llm_calls,
        tool_calls=tool_calls,
        max_tool_calls=request.limits.max_tool_calls,
        input_tokens=state["input_tokens"],
        output_tokens=state["output_tokens"],
        max_total_tokens=request.limits.max_total_tokens,
        evidence_count=evidence_count,
        coverage_ratio=coverage_ratio,
        capability_model_calls=state["capability_model_calls"],
    )


def _last_tool_event(state: AgentState) -> AgentToolEvent | None:
    events = state.get("tool_events", ())
    return events[-1] if events else None


def _argument_summary(arguments: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    safe: list[tuple[str, str]] = []
    for key, value in list(arguments.items())[:6]:
        if key.lower() in {"api_key", "token", "authorization"}:
            continue
        safe.append((key, _truncate(str(value), 120)))
    return tuple(safe)


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."
