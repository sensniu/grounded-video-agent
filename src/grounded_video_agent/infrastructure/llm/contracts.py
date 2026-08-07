"""Provider-neutral contracts for text language-model calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class LLMRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LLMOutputFormat(StrEnum):
    TEXT = "text"
    JSON_OBJECT = "json_object"


class LLMFinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    INSUFFICIENT_RESOURCES = "insufficient_resources"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: LLMRole
    content: str

    def __post_init__(self) -> None:
        _require_text(self.content, "content")


@dataclass(frozen=True, slots=True)
class StructuredOutputSpec:
    """Shape guidance for JSON output; domain decoding remains the caller's job."""

    name: str
    json_schema: dict[str, Any]
    example: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        if not self.json_schema:
            raise ValueError("json_schema must not be empty")


@dataclass(frozen=True, slots=True)
class LLMRequest:
    operation_id: str
    messages: tuple[LLMMessage, ...]
    output_format: LLMOutputFormat = LLMOutputFormat.TEXT
    structured_output: StructuredOutputSpec | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    stop: tuple[str, ...] = ()
    trace_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.operation_id, "operation_id")
        if not self.messages:
            raise ValueError("messages must not be empty")
        if self.trace_id is not None:
            _require_text(self.trace_id, "trace_id")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between zero and two")
        if len(self.stop) > 16:
            raise ValueError("stop must contain at most 16 sequences")
        if any(not value.strip() for value in self.stop):
            raise ValueError("stop sequences must not be empty")
        if len(set(self.stop)) != len(self.stop):
            raise ValueError("stop sequences must be unique")
        if self.output_format is LLMOutputFormat.JSON_OBJECT:
            if self.structured_output is None:
                raise ValueError("JSON output requires structured_output")
        elif self.structured_output is not None:
            raise ValueError("structured_output is only valid for JSON output")


@dataclass(frozen=True, slots=True)
class LLMModelInfo:
    provider: str
    model: str
    supports_json_object: bool
    supports_streaming: bool = False

    def __post_init__(self) -> None:
        _require_text(self.provider, "provider")
        _require_text(self.model, "model")


@dataclass(frozen=True, slots=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    model_calls: int = 1

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.model_calls <= 0:
            raise ValueError("model_calls must be positive")
        if self.total_tokens and self.total_tokens < self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens cannot be less than input plus output tokens")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")


@dataclass(frozen=True, slots=True)
class LLMResponse:
    operation_id: str
    response_id: str
    provider: str
    model: str
    output_format: LLMOutputFormat
    content: str
    finish_reason: LLMFinishReason
    usage: LLMUsage
    latency_ms: int
    json_object: dict[str, Any] | None = None
    provider_request_id: str | None = None
    attempt_count: int = 1
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in ("operation_id", "response_id", "provider", "model", "content"):
            _require_text(getattr(self, field_name), field_name)
        if self.output_format is LLMOutputFormat.JSON_OBJECT:
            if self.json_object is None:
                raise ValueError("JSON responses require json_object")
        elif self.json_object is not None:
            raise ValueError("text responses must not contain json_object")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if self.attempt_count <= 0:
            raise ValueError("attempt_count must be positive")
        if self.usage.model_calls != self.attempt_count:
            raise ValueError("usage.model_calls must equal attempt_count")
        if self.provider_request_id is not None:
            _require_text(self.provider_request_id, "provider_request_id")
        if any(not warning.strip() for warning in self.warnings):
            raise ValueError("warnings must not contain empty values")


def _require_text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
