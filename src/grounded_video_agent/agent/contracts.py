"""Public Agent contracts and LLM-facing structured-output schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grounded_video_agent.capabilities._support import json_value
from grounded_video_agent.domain import EvidenceVerificationReport, TimeRange


class AgentStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    ABSTAINED = "abstained"
    FAILED = "failed"


class QuestionIntent(StrEnum):
    METADATA = "metadata"
    LOCAL_EVENT = "local_event"
    VISUAL = "visual"
    SCREEN_TEXT = "screen_text"
    GLOBAL = "global"
    COUNT = "count"
    CAUSAL = "causal"
    OTHER = "other"


class AgentAction(StrEnum):
    CALL_TOOL = "call_tool"
    ANSWER = "answer"
    ABSTAIN = "abstain"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AgentDecision(_StrictModel):
    intent: QuestionIntent
    current_goal: str = Field(min_length=1)
    missing_information: tuple[str, ...] = ()
    action: AgentAction
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    supporting_evidence_ids: tuple[str, ...] = ()
    expected_information_gain: str | None = None
    final_reason: str | None = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> Self:
        _require_unique_texts(self.missing_information, "missing_information")
        _require_unique_texts(self.supporting_evidence_ids, "supporting_evidence_ids")
        if self.action is AgentAction.CALL_TOOL:
            if self.tool_name is None or not self.tool_name.strip():
                raise ValueError("call_tool decisions require tool_name")
            if self.final_reason is not None:
                raise ValueError("call_tool decisions must not contain final_reason")
        else:
            if self.tool_name is not None or self.tool_arguments:
                raise ValueError("answer and abstain decisions must not contain tool calls")
        if self.action is AgentAction.ANSWER:
            if (
                self.intent is not QuestionIntent.METADATA
                and not self.supporting_evidence_ids
            ):
                raise ValueError("non-metadata answers require supporting_evidence_ids")
        if self.action is AgentAction.ABSTAIN:
            if self.final_reason is None or not self.final_reason.strip():
                raise ValueError("abstain decisions require final_reason")
        for field_name in ("expected_information_gain", "final_reason"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        return self


class AnswerClaimDraft(_StrictModel):
    text: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    confidence: float | None = Field(default=None, ge=0, le=1)
    uncertainties: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_collections(self) -> Self:
        _require_unique_texts(self.evidence_ids, "evidence_ids")
        _require_unique_texts(self.uncertainties, "uncertainties")
        return self


class AnswerDraft(_StrictModel):
    answer: str = Field(min_length=1)
    claims: tuple[AnswerClaimDraft, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_limitations(self) -> Self:
        _require_unique_texts(self.limitations, "limitations")
        return self


@dataclass(frozen=True, slots=True)
class AgentLimits:
    max_iterations: int = 50
    max_tool_calls: int = 100
    max_llm_calls: int = 60
    max_total_tokens: int = 6_000_000
    max_replans: int = 3
    max_remediations: int = 2
    min_global_coverage: float = 0.7

    def __post_init__(self) -> None:
        for name in (
            "max_iterations",
            "max_tool_calls",
            "max_llm_calls",
            "max_total_tokens",
            "max_replans",
            "max_remediations",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.min_global_coverage <= 1:
            raise ValueError("min_global_coverage must be between zero and one")


@dataclass(frozen=True, slots=True)
class AgentRequest:
    filename: str
    question: str
    request_id: str = field(default_factory=lambda: f"agent_{uuid4().hex}")
    response_language: str = "zh-CN"
    evidence_clip_requested: bool = False
    force_refresh: bool = False
    limits: AgentLimits = field(default_factory=AgentLimits)
    trace_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("filename", "question", "request_id", "response_language"):
            value = getattr(self, name)
            if not value or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.trace_id is not None and not self.trace_id.strip():
            raise ValueError("trace_id must not be empty")


@dataclass(frozen=True, slots=True)
class AgentCitation:
    evidence_id: str
    time_range: TimeRange
    modality: str
    excerpt: str | None = None


@dataclass(frozen=True, slots=True)
class AgentClaim:
    claim_id: str
    text: str
    evidence_ids: tuple[str, ...]
    confidence: float | None = None
    uncertainties: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentAttachment:
    attachment_id: str
    artifact_id: str
    filename: str
    evidence_ids: tuple[str, ...]
    time_range: TimeRange
    size_bytes: int


@dataclass(frozen=True, slots=True)
class AgentUsage:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    capability_model_calls: int = 0

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")


@dataclass(frozen=True, slots=True)
class AgentError:
    code: str
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.message.strip():
            raise ValueError("agent error code and message must not be empty")


@dataclass(frozen=True, slots=True)
class AgentResult:
    request_id: str
    status: AgentStatus
    video_id: str | None
    answer: str | None
    claims: tuple[AgentClaim, ...] = ()
    citations: tuple[AgentCitation, ...] = ()
    attachments: tuple[AgentAttachment, ...] = ()
    verification: EvidenceVerificationReport | None = None
    usage: AgentUsage = field(default_factory=AgentUsage)
    warnings: tuple[str, ...] = ()
    error: AgentError | None = None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if self.status is AgentStatus.FAILED:
            if self.error is None or self.answer is not None:
                raise ValueError("failed results require an error and no answer")
        elif self.error is not None:
            raise ValueError("only failed results may contain an error")
        if self.status is AgentStatus.SUCCESS and not self.answer:
            raise ValueError("successful results require an answer")
        if any(not warning.strip() for warning in self.warnings):
            raise ValueError("warnings must not contain empty values")

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json_value(self))

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)


def _require_unique_texts(values: tuple[str, ...], field_name: str) -> None:
    if any(not value.strip() for value in values):
        raise ValueError(f"{field_name} must not contain empty values")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must be unique")
