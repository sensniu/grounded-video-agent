from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from grounded_video_agent.infrastructure.llm import (
    DeepSeekBackendConfig,
    DeepSeekLLMBackend,
    LLMBackendError,
    LLMErrorCode,
    LLMFinishReason,
    LLMMessage,
    LLMOutputFormat,
    LLMRequest,
    LLMRole,
    StructuredOutputSpec,
)
from grounded_video_agent.observability import JsonlTraceRecorder, trace_context

API_KEY = "test-deepseek-secret"


def _config(**overrides: object) -> DeepSeekBackendConfig:
    values: dict[str, object] = {
        "model": "deepseek-test",
        "base_url": "https://deepseek.test",
        "transient_retries": 0,
        "retry_backoff_seconds": 0,
    }
    values.update(overrides)
    return DeepSeekBackendConfig(**values)  # type: ignore[arg-type]


def _request(*, json_output: bool = False) -> LLMRequest:
    kwargs: dict[str, Any] = {}
    if json_output:
        kwargs.update(
            output_format=LLMOutputFormat.JSON_OBJECT,
            structured_output=StructuredOutputSpec(
                "decision",
                {
                    "type": "object",
                    "properties": {"action": {"type": "string"}},
                    "required": ["action"],
                },
                {"action": "answer"},
            ),
        )
    return LLMRequest(
        "operation-1",
        (
            LLMMessage(LLMRole.SYSTEM, "Use only supplied evidence."),
            LLMMessage(LLMRole.USER, "What happened?"),
        ),
        **kwargs,
    )


def _completion(
    content: str = "A person entered the room.",
    *,
    finish_reason: str = "stop",
) -> dict[str, object]:
    return {
        "id": "completion-1",
        "model": "deepseek-test",
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 25,
        },
    }


@pytest.mark.asyncio
async def test_deepseek_backend_maps_text_completion_and_usage() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            headers={"x-request-id": "provider-request-1"},
            json=_completion(),
        )

    backend = DeepSeekLLMBackend(
        _config(base_url="https://deepseek.test/v1"),
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
    )

    result = await backend.complete(_request())

    assert result.content == "A person entered the room."
    assert result.json_object is None
    assert result.finish_reason is LLMFinishReason.STOP
    assert result.provider_request_id == "provider-request-1"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 20
    assert result.usage.cached_input_tokens == 25
    assert result.attempt_count == 1
    assert captured[0].url.path == "/v1/chat/completions"
    assert captured[0].headers["authorization"] == f"Bearer {API_KEY}"
    payload = json.loads(captured[0].content)
    assert payload["model"] == "deepseek-test"
    assert payload["stream"] is False
    assert "response_format" not in payload


@pytest.mark.asyncio
async def test_deepseek_backend_requests_and_parses_json_object() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_completion('{"action":"search"}'))

    backend = DeepSeekLLMBackend(
        _config(),
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
    )

    result = await backend.complete(_request(json_output=True))

    assert result.json_object == {"action": "search"}
    assert captured[0]["response_format"] == {"type": "json_object"}
    messages = cast(list[dict[str, str]], captured[0]["messages"])
    assert "Return JSON only" in messages[0]["content"]
    assert '"required":["action"]' in messages[0]["content"]
    assert "Use only supplied evidence" in messages[0]["content"]


@pytest.mark.asyncio
async def test_deepseek_backend_retries_empty_structured_output_and_counts_usage() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json=_completion("   "))
        return httpx.Response(200, json=_completion('{"action":"answer"}'))

    backend = DeepSeekLLMBackend(
        _config(transient_retries=1),
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
    )

    result = await backend.complete(_request(json_output=True))

    assert result.json_object == {"action": "answer"}
    assert result.attempt_count == 2
    assert result.usage.model_calls == 2
    assert result.usage.total_tokens == 240


@pytest.mark.asyncio
async def test_deepseek_backend_retries_transient_http_status() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return httpx.Response(200, json=_completion())

    backend = DeepSeekLLMBackend(
        _config(transient_retries=1),
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
    )

    result = await backend.complete(_request())

    assert result.attempt_count == 2
    assert calls == 2


@pytest.mark.asyncio
async def test_deepseek_backend_maps_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    backend = DeepSeekLLMBackend(
        _config(),
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMBackendError) as caught:
        await backend.complete(_request())

    assert caught.value.code is LLMErrorCode.TIMEOUT
    assert caught.value.retryable is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "message", "expected_code", "retryable"),
    [
        (401, "invalid key", LLMErrorCode.AUTHENTICATION_FAILED, False),
        (429, "rate limit", LLMErrorCode.RATE_LIMITED, True),
        (400, "maximum context length exceeded", LLMErrorCode.CONTEXT_LENGTH_EXCEEDED, False),
        (500, "internal failure", LLMErrorCode.SERVICE_UNAVAILABLE, True),
    ],
)
async def test_deepseek_backend_normalizes_http_errors(
    status_code: int,
    message: str,
    expected_code: LLMErrorCode,
    retryable: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            headers={"x-request-id": "failed-request"},
            json={"error": {"message": message}},
        )

    backend = DeepSeekLLMBackend(
        _config(),
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMBackendError) as caught:
        await backend.complete(_request())

    assert caught.value.code is expected_code
    assert caught.value.retryable is retryable
    assert caught.value.status_code == status_code
    assert caught.value.request_id == "failed-request"


@pytest.mark.asyncio
async def test_deepseek_backend_rejects_truncated_and_invalid_json_outputs() -> None:
    responses = [
        _completion("partial", finish_reason="length"),
        _completion("not-json"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses.pop(0))

    backend = DeepSeekLLMBackend(
        _config(),
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMBackendError) as truncated:
        await backend.complete(_request())
    with pytest.raises(LLMBackendError) as invalid_json:
        await backend.complete(_request(json_output=True))

    assert truncated.value.code is LLMErrorCode.OUTPUT_TRUNCATED
    assert truncated.value.retryable is False
    assert invalid_json.value.code is LLMErrorCode.INVALID_JSON


@pytest.mark.asyncio
async def test_deepseek_trace_preserves_raw_truncated_response_before_error(
    tmp_path: Path,
) -> None:
    completion = _completion("partial", finish_reason="length")
    choices = cast(list[dict[str, Any]], completion["choices"])
    message = cast(dict[str, Any], choices[0]["message"])
    message["reasoning_content"] = "provider reasoning"
    usage = cast(dict[str, Any], completion["usage"])
    usage["completion_tokens_details"] = {"reasoning_tokens": 4_090}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-request-id": "truncated-request"},
            json=completion,
        )

    backend = DeepSeekLLMBackend(
        _config(),
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
    )
    recorder = JsonlTraceRecorder.create(tmp_path)
    with recorder, trace_context(recorder):
        with pytest.raises(LLMBackendError):
            await backend.complete(_request())

    raw = recorder.path.read_text(encoding="utf-8")
    assert API_KEY not in raw
    events = [json.loads(line) for line in raw.splitlines()]
    response = next(event for event in events if event["event_type"] == "provider.response")
    payload = response["payload"]["payload"]
    assert payload["choices"][0]["finish_reason"] == "length"
    assert payload["choices"][0]["message"]["reasoning_content"] == "provider reasoning"
    assert payload["usage"]["completion_tokens_details"]["reasoning_tokens"] == 4_090
    assert any(event["event_type"] == "provider.error" for event in events)


@pytest.mark.asyncio
async def test_deepseek_backend_redacts_api_key_from_provider_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": f"bad credential {API_KEY}"}},
        )

    backend = DeepSeekLLMBackend(
        _config(),
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMBackendError) as caught:
        await backend.complete(_request())

    assert API_KEY not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)


def test_deepseek_backend_requires_api_key_and_reports_model_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(LLMBackendError) as caught:
        DeepSeekLLMBackend(_config())

    assert caught.value.code is LLMErrorCode.CONFIGURATION_ERROR
    backend = DeepSeekLLMBackend(_config(), api_key=API_KEY)
    info = backend.get_model_info()
    assert info.provider == "deepseek"
    assert info.model == "deepseek-test"
    assert info.supports_json_object is True
