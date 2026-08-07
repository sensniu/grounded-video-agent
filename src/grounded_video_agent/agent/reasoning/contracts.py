from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from grounded_video_agent.infrastructure.llm import LLMResponse

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PlanningContext:
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AnswerContext:
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReasoningResult(Generic[T]):
    data: T
    responses: tuple[LLMResponse, ...]


class AgentReasoningError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        if not code.strip() or not message.strip():
            raise ValueError("reasoning error code and message must not be empty")
        super().__init__(message)
        self.code = code
        self.retryable = retryable
