"""Async DeepSeek adapter for its OpenAI-compatible chat-completions API."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, cast

import httpx

from grounded_video_agent.infrastructure.llm.config import DeepSeekBackendConfig
from grounded_video_agent.infrastructure.llm.contracts import (
    LLMFinishReason,
    LLMModelInfo,
    LLMOutputFormat,
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMUsage,
    StructuredOutputSpec,
)
from grounded_video_agent.infrastructure.llm.errors import LLMBackendError, LLMErrorCode

_PROVIDER = "deepseek"
_TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


@dataclass(slots=True)
class _UsageAccumulator:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0

    def add(self, usage: LLMUsage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        self.cached_input_tokens += usage.cached_input_tokens

    def freeze(self, *, model_calls: int) -> LLMUsage:
        return LLMUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=max(self.total_tokens, self.input_tokens + self.output_tokens),
            cached_input_tokens=min(self.cached_input_tokens, self.input_tokens),
            model_calls=model_calls,
        )


class DeepSeekLLMBackend:
    def __init__(
        self,
        config: DeepSeekBackendConfig | None = None,
        *,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config or DeepSeekBackendConfig()
        resolved_api_key = api_key if api_key is not None else os.getenv(self._config.api_key_env)
        if resolved_api_key is None or not resolved_api_key.strip():
            raise LLMBackendError(
                LLMErrorCode.CONFIGURATION_ERROR,
                f"DeepSeek API key is missing; set {self._config.api_key_env}",
                retryable=False,
                provider=_PROVIDER,
                suggested_action=f"Set the {self._config.api_key_env} environment variable.",
            )
        self._api_key = resolved_api_key.strip()
        self._transport = transport

    def get_model_info(self) -> LLMModelInfo:
        return LLMModelInfo(
            provider=_PROVIDER,
            model=self._config.model,
            supports_json_object=True,
            supports_streaming=False,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload = self._build_payload(request)
        timeout = httpx.Timeout(
            self._config.completion_timeout_seconds,
            connect=self._config.connect_timeout_seconds,
        )
        attempts = self._config.transient_retries + 1
        started = perf_counter()
        usage = _UsageAccumulator()
        last_error: LLMBackendError | None = None
        async with httpx.AsyncClient(
            base_url=self._config.base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
            transport=self._transport,
        ) as client:
            for attempt in range(1, attempts + 1):
                try:
                    response = await client.post("chat/completions", json=payload)
                    provider_request_id = _request_id(response)
                    if response.status_code < 200 or response.status_code >= 300:
                        raise self._http_error(response, provider_request_id)
                    response_payload = self._decode_response(response, provider_request_id)
                    usage.add(_parse_usage(response_payload))
                    return self._parse_completion(
                        request,
                        response_payload,
                        provider_request_id=provider_request_id,
                        attempt_count=attempt,
                        usage=usage.freeze(model_calls=attempt),
                        latency_ms=round((perf_counter() - started) * 1_000),
                    )
                except httpx.TimeoutException as error:
                    last_error = LLMBackendError(
                        LLMErrorCode.TIMEOUT,
                        "DeepSeek request timed out",
                        retryable=True,
                        provider=_PROVIDER,
                        suggested_action="Retry the request or increase the completion timeout.",
                    )
                    last_error.__cause__ = error
                except httpx.RequestError as error:
                    last_error = LLMBackendError(
                        LLMErrorCode.CONNECTION_FAILED,
                        "DeepSeek connection failed",
                        retryable=True,
                        provider=_PROVIDER,
                        suggested_action="Check network connectivity and retry.",
                    )
                    last_error.__cause__ = error
                except LLMBackendError as error:
                    last_error = error
                if last_error is None or not last_error.retryable or attempt >= attempts:
                    if last_error is None:
                        raise AssertionError("DeepSeek request failed without an error")
                    raise last_error
                await self._backoff(attempt)
        raise AssertionError("DeepSeek retry loop exited unexpectedly")

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        messages = [
            {"role": message.role.value, "content": message.content}
            for message in request.messages
        ]
        if request.output_format is LLMOutputFormat.JSON_OBJECT:
            structured_output = request.structured_output
            if structured_output is None:
                raise AssertionError("validated JSON request has no structured output spec")
            messages = _with_json_instruction(messages, structured_output)
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": request.max_output_tokens
            or self._config.default_max_output_tokens,
            "temperature": (
                request.temperature
                if request.temperature is not None
                else self._config.default_temperature
            ),
            "stream": False,
        }
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.output_format is LLMOutputFormat.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _decode_response(
        self,
        response: httpx.Response,
        provider_request_id: str | None,
    ) -> dict[str, Any]:
        try:
            payload = response.json()
        except json.JSONDecodeError as error:
            raise LLMBackendError(
                LLMErrorCode.INVALID_RESPONSE,
                "DeepSeek returned a non-JSON API response",
                retryable=True,
                provider=_PROVIDER,
                status_code=response.status_code,
                request_id=provider_request_id,
            ) from error
        if not isinstance(payload, dict):
            raise LLMBackendError(
                LLMErrorCode.INVALID_RESPONSE,
                "DeepSeek returned a non-object API response",
                retryable=True,
                provider=_PROVIDER,
                status_code=response.status_code,
                request_id=provider_request_id,
            )
        return cast(dict[str, Any], payload)

    def _parse_completion(
        self,
        request: LLMRequest,
        payload: dict[str, Any],
        *,
        provider_request_id: str | None,
        attempt_count: int,
        usage: LLMUsage,
        latency_ms: int,
    ) -> LLMResponse:
        response_id = payload.get("id")
        choices = payload.get("choices")
        if not isinstance(response_id, str) or not response_id.strip():
            raise self._invalid_response("DeepSeek response is missing id", provider_request_id)
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise self._invalid_response(
                "DeepSeek response has no completion choice",
                provider_request_id,
            )
        choice = choices[0]
        provider_finish_reason = choice.get("finish_reason")
        finish_reason = _finish_reason(provider_finish_reason)
        if finish_reason is LLMFinishReason.LENGTH:
            raise LLMBackendError(
                LLMErrorCode.OUTPUT_TRUNCATED,
                "DeepSeek output was truncated by the token limit",
                retryable=False,
                provider=_PROVIDER,
                request_id=provider_request_id,
                suggested_action="Increase max_output_tokens or reduce the prompt size.",
            )
        if finish_reason is LLMFinishReason.CONTENT_FILTER:
            raise LLMBackendError(
                LLMErrorCode.CONTENT_FILTERED,
                "DeepSeek filtered the completion",
                retryable=False,
                provider=_PROVIDER,
                request_id=provider_request_id,
            )
        if finish_reason is LLMFinishReason.INSUFFICIENT_RESOURCES:
            raise LLMBackendError(
                LLMErrorCode.INSUFFICIENT_PROVIDER_RESOURCES,
                "DeepSeek reported insufficient inference resources",
                retryable=True,
                provider=_PROVIDER,
                request_id=provider_request_id,
                suggested_action="Retry later or select a smaller model.",
            )
        if finish_reason is LLMFinishReason.TOOL_CALLS:
            raise LLMBackendError(
                LLMErrorCode.UNSUPPORTED_RESPONSE,
                "Native tool calls are not supported by this LLM interface",
                retryable=False,
                provider=_PROVIDER,
                request_id=provider_request_id,
            )
        message = choice.get("message")
        if not isinstance(message, dict):
            raise self._invalid_response(
                "DeepSeek completion choice has no message",
                provider_request_id,
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMBackendError(
                LLMErrorCode.EMPTY_RESPONSE,
                "DeepSeek returned an empty completion",
                retryable=True,
                provider=_PROVIDER,
                request_id=provider_request_id,
            )
        parsed_json: dict[str, Any] | None = None
        if request.output_format is LLMOutputFormat.JSON_OBJECT:
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError as error:
                raise LLMBackendError(
                    LLMErrorCode.INVALID_JSON,
                    "DeepSeek returned invalid JSON content",
                    retryable=True,
                    provider=_PROVIDER,
                    request_id=provider_request_id,
                    suggested_action="Retry the structured-output request.",
                ) from error
            if not isinstance(decoded, dict):
                raise LLMBackendError(
                    LLMErrorCode.INVALID_JSON,
                    "DeepSeek JSON output must be an object",
                    retryable=True,
                    provider=_PROVIDER,
                    request_id=provider_request_id,
                )
            parsed_json = cast(dict[str, Any], decoded)
        model = payload.get("model")
        warnings: tuple[str, ...] = ()
        if not isinstance(model, str) or not model.strip():
            model = self._config.model
            warnings = ("DeepSeek response did not identify the served model.",)
        if finish_reason is LLMFinishReason.UNKNOWN:
            warnings += (f"Unrecognized finish reason: {provider_finish_reason!r}",)
        return LLMResponse(
            operation_id=request.operation_id,
            response_id=response_id,
            provider=_PROVIDER,
            model=model,
            output_format=request.output_format,
            content=content,
            json_object=parsed_json,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
            provider_request_id=provider_request_id,
            attempt_count=attempt_count,
            warnings=warnings,
        )

    def _http_error(
        self,
        response: httpx.Response,
        provider_request_id: str | None,
    ) -> LLMBackendError:
        status_code = response.status_code
        message = self._provider_error_message(response)
        lowered = message.lower()
        if status_code in {401, 403}:
            code = LLMErrorCode.AUTHENTICATION_FAILED
            retryable = False
            action = "Verify the DeepSeek API key and account permissions."
        elif status_code == 429:
            code = LLMErrorCode.RATE_LIMITED
            retryable = True
            action = "Retry after the provider rate-limit window."
        elif "context" in lowered and ("length" in lowered or "long" in lowered):
            code = LLMErrorCode.CONTEXT_LENGTH_EXCEEDED
            retryable = False
            action = "Reduce the prompt or evidence context."
        elif status_code == 408:
            code = LLMErrorCode.TIMEOUT
            retryable = True
            action = "Retry the request."
        elif status_code >= 500:
            code = LLMErrorCode.SERVICE_UNAVAILABLE
            retryable = True
            action = "Retry later."
        else:
            code = LLMErrorCode.INVALID_REQUEST
            retryable = status_code in _TRANSIENT_STATUS_CODES
            action = "Check the model name and request parameters."
        return LLMBackendError(
            code,
            message,
            retryable=retryable,
            provider=_PROVIDER,
            status_code=status_code,
            request_id=provider_request_id,
            suggested_action=action,
        )

    def _provider_error_message(self, response: httpx.Response) -> str:
        fallback = f"DeepSeek API returned HTTP {response.status_code}"
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return fallback
        if not isinstance(payload, Mapping):
            return fallback
        error = payload.get("error")
        if not isinstance(error, Mapping):
            return fallback
        message = error.get("message")
        if not isinstance(message, str) or not message.strip():
            return fallback
        sanitized = message.replace(self._api_key, "[REDACTED]")
        return sanitized.strip()[:500]

    @staticmethod
    def _invalid_response(
        message: str,
        provider_request_id: str | None,
    ) -> LLMBackendError:
        return LLMBackendError(
            LLMErrorCode.INVALID_RESPONSE,
            message,
            retryable=True,
            provider=_PROVIDER,
            request_id=provider_request_id,
        )

    async def _backoff(self, attempt: int) -> None:
        delay = self._config.retry_backoff_seconds * (2 ** (attempt - 1))
        if delay > 0:
            await asyncio.sleep(delay)


def _with_json_instruction(
    messages: list[dict[str, str]],
    spec: StructuredOutputSpec,
) -> list[dict[str, str]]:
    schema = json.dumps(spec.json_schema, ensure_ascii=False, separators=(",", ":"))
    instruction = (
        f"Return JSON only. For the {spec.name!r} output contract, the response must be one "
        f"JSON object matching this JSON Schema: {schema}."
    )
    if spec.example is not None:
        example = json.dumps(spec.example, ensure_ascii=False, separators=(",", ":"))
        instruction += f" Example JSON object: {example}."
    updated = [dict(message) for message in messages]
    if updated and updated[0]["role"] == LLMRole.SYSTEM.value:
        updated[0]["content"] = f"{updated[0]['content']}\n\n{instruction}"
    else:
        updated.insert(0, {"role": LLMRole.SYSTEM.value, "content": instruction})
    return updated


def _request_id(response: httpx.Response) -> str | None:
    for header in ("x-request-id", "x-ds-request-id", "request-id"):
        value = response.headers.get(header)
        if value and value.strip():
            return value.strip()
    return None


def _parse_usage(payload: dict[str, Any]) -> LLMUsage:
    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, Mapping):
        return LLMUsage()
    input_tokens = _non_negative_int(raw_usage.get("prompt_tokens"))
    output_tokens = _non_negative_int(raw_usage.get("completion_tokens"))
    total_tokens = max(
        _non_negative_int(raw_usage.get("total_tokens")),
        input_tokens + output_tokens,
    )
    cached_input_tokens = min(
        _non_negative_int(raw_usage.get("prompt_cache_hit_tokens")),
        input_tokens,
    )
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
    )


def _non_negative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _finish_reason(value: object) -> LLMFinishReason:
    mapping = {
        "stop": LLMFinishReason.STOP,
        "length": LLMFinishReason.LENGTH,
        "content_filter": LLMFinishReason.CONTENT_FILTER,
        "tool_calls": LLMFinishReason.TOOL_CALLS,
        "insufficient_system_resource": LLMFinishReason.INSUFFICIENT_RESOURCES,
    }
    if not isinstance(value, str):
        return LLMFinishReason.UNKNOWN
    return mapping.get(value, LLMFinishReason.UNKNOWN)
