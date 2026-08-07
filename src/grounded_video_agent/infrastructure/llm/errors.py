"""Normalized language-model backend errors."""

from __future__ import annotations

from enum import StrEnum


class LLMErrorCode(StrEnum):
    CONFIGURATION_ERROR = "configuration_error"
    AUTHENTICATION_FAILED = "authentication_failed"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    CONNECTION_FAILED = "connection_failed"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INVALID_REQUEST = "invalid_request"
    INVALID_RESPONSE = "invalid_response"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    OUTPUT_TRUNCATED = "output_truncated"
    EMPTY_RESPONSE = "empty_response"
    INVALID_JSON = "invalid_json"
    CONTENT_FILTERED = "content_filtered"
    INSUFFICIENT_PROVIDER_RESOURCES = "insufficient_provider_resources"
    UNSUPPORTED_RESPONSE = "unsupported_response"


class LLMBackendError(RuntimeError):
    """A safe, provider-neutral failure raised by an LLM transport adapter."""

    def __init__(
        self,
        code: LLMErrorCode,
        message: str,
        *,
        retryable: bool,
        provider: str,
        status_code: int | None = None,
        request_id: str | None = None,
        suggested_action: str | None = None,
    ) -> None:
        if not message.strip() or not provider.strip():
            raise ValueError("message and provider must not be empty")
        if status_code is not None and status_code <= 0:
            raise ValueError("status_code must be positive")
        if request_id is not None and not request_id.strip():
            raise ValueError("request_id must not be empty")
        if suggested_action is not None and not suggested_action.strip():
            raise ValueError("suggested_action must not be empty")
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.provider = provider
        self.status_code = status_code
        self.request_id = request_id
        self.suggested_action = suggested_action
