"""Structured planning and answer generation over the generic LLM backend."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from grounded_video_agent.agent.contracts import AgentDecision, AnswerDraft
from grounded_video_agent.infrastructure.llm import (
    LLMBackend,
    LLMBackendError,
    LLMMessage,
    LLMOutputFormat,
    LLMRequest,
    LLMResponse,
    LLMRole,
    StructuredOutputSpec,
)
from grounded_video_agent.observability import emit_trace

from .contracts import (
    AgentReasoningError,
    AnswerContext,
    PlanningContext,
    ReasoningResult,
)
from .prompts import answer_messages, planning_messages

TModel = TypeVar("TModel", bound=BaseModel)


class AgentReasoningService:
    def __init__(self, backend: LLMBackend, *, schema_retries: int = 1) -> None:
        if schema_retries < 0:
            raise ValueError("schema_retries must be non-negative")
        self._backend = backend
        self._schema_retries = schema_retries

    async def plan(
        self,
        context: PlanningContext,
        *,
        operation_id: str,
        trace_id: str | None,
    ) -> ReasoningResult[AgentDecision]:
        return await self._generate(
            AgentDecision,
            planning_messages(context),
            operation_id=operation_id,
            trace_id=trace_id,
            schema_name="agent_decision",
            example={
                "intent": "local_event",
                "current_goal": "Find the relevant transcript segment",
                "missing_information": ["Relevant time range"],
                "action": "call_tool",
                "tool_name": "search_video_transcript",
                "tool_arguments": {"query": "person opens door", "top_k": 5},
                "supporting_evidence_ids": [],
                "expected_information_gain": "A candidate segment and exact time range",
                "final_reason": None,
            },
            max_output_tokens=12_000,
        )

    async def draft_answer(
        self,
        context: AnswerContext,
        *,
        operation_id: str,
        trace_id: str | None,
    ) -> ReasoningResult[AnswerDraft]:
        return await self._generate(
            AnswerDraft,
            answer_messages(context),
            operation_id=operation_id,
            trace_id=trace_id,
            schema_name="grounded_answer",
            example={
                "answer": "The person opens the door and enters the room.",
                "claims": [
                    {
                        "text": "The person opens the door.",
                        "evidence_ids": ["evidence_example"],
                        "confidence": 0.9,
                        "uncertainties": [],
                    }
                ],
                "limitations": [],
            },
            max_output_tokens=64_000,
        )

    async def _generate(
        self,
        model_type: type[TModel],
        messages: tuple[LLMMessage, ...],
        *,
        operation_id: str,
        trace_id: str | None,
        schema_name: str,
        example: dict[str, Any],
        max_output_tokens: int,
    ) -> ReasoningResult[TModel]:
        responses: list[LLMResponse] = []
        current_messages = messages
        for attempt in range(self._schema_retries + 1):
            request = LLMRequest(
                operation_id=f"{operation_id}_{attempt + 1}",
                messages=current_messages,
                output_format=LLMOutputFormat.JSON_OBJECT,
                structured_output=StructuredOutputSpec(
                    schema_name,
                    model_type.model_json_schema(),
                    example,
                ),
                max_output_tokens=max_output_tokens,
                temperature=0,
                trace_id=trace_id,
            )
            emit_trace(
                "llm.request",
                {"request": request, "attempt": attempt + 1},
                operation_id=request.operation_id,
                phase="reasoning",
            )
            try:
                response = await self._backend.complete(request)
            except LLMBackendError as error:
                emit_trace(
                    "llm.error",
                    {
                        "code": error.code,
                        "message": str(error),
                        "retryable": error.retryable,
                        "provider": error.provider,
                        "status_code": error.status_code,
                        "provider_request_id": error.request_id,
                        "suggested_action": error.suggested_action,
                    },
                    operation_id=request.operation_id,
                    phase="reasoning",
                )
                raise AgentReasoningError(
                    error.code.value,
                    str(error),
                    retryable=error.retryable,
                ) from error
            except Exception as error:
                emit_trace(
                    "llm.error",
                    {"error": error, "retryable": False},
                    operation_id=request.operation_id,
                    phase="reasoning",
                )
                raise
            emit_trace(
                "llm.response",
                {"response": response},
                operation_id=request.operation_id,
                phase="reasoning",
            )
            responses.append(response)
            try:
                return ReasoningResult(
                    model_type.model_validate(response.json_object),
                    tuple(responses),
                )
            except ValidationError as error:
                emit_trace(
                    "llm.validation_error",
                    {
                        "schema_name": schema_name,
                        "summary": _validation_summary(error),
                        "errors": error.errors(include_url=False),
                    },
                    operation_id=request.operation_id,
                    phase="reasoning",
                )
                if attempt >= self._schema_retries:
                    raise AgentReasoningError(
                        "INVALID_STRUCTURED_OUTPUT",
                        "The model response did not match the required schema",
                        retryable=False,
                    ) from error
                current_messages = (
                    *messages,
                    LLMMessage(LLMRole.ASSISTANT, response.content),
                    LLMMessage(
                        LLMRole.USER,
                        "The previous JSON did not match the required schema. Correct it and "
                        "return only the corrected JSON object. Validation summary: "
                        + _validation_summary(error),
                    ),
                )
        raise AssertionError("structured generation loop exited unexpectedly")


def _validation_summary(error: ValidationError) -> str:
    parts = []
    for item in error.errors(include_url=False)[:8]:
        location = ".".join(str(value) for value in item["loc"])
        parts.append(f"{location}: {item['msg']}")
    return "; ".join(parts)
