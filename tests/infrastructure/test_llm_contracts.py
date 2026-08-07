from __future__ import annotations

import pytest

from grounded_video_agent.infrastructure.llm import (
    DeepSeekBackendConfig,
    LLMMessage,
    LLMOutputFormat,
    LLMRequest,
    LLMRole,
    LLMUsage,
    StructuredOutputSpec,
)


def test_llm_request_accepts_text_and_structured_modes() -> None:
    message = LLMMessage(LLMRole.USER, "Summarize the evidence.")

    text_request = LLMRequest("text-operation", (message,))
    json_request = LLMRequest(
        "json-operation",
        (message,),
        output_format=LLMOutputFormat.JSON_OBJECT,
        structured_output=StructuredOutputSpec(
            "decision",
            {"type": "object", "required": ["action"]},
            {"action": "answer"},
        ),
        max_output_tokens=512,
        temperature=0.1,
        stop=("DONE",),
        trace_id="trace-1",
    )

    assert text_request.output_format is LLMOutputFormat.TEXT
    assert json_request.structured_output is not None
    assert json_request.structured_output.name == "decision"


def test_llm_request_rejects_mismatched_structured_output() -> None:
    message = LLMMessage(LLMRole.USER, "Decide.")
    spec = StructuredOutputSpec("decision", {"type": "object"})

    with pytest.raises(ValueError, match="requires structured_output"):
        LLMRequest(
            "operation",
            (message,),
            output_format=LLMOutputFormat.JSON_OBJECT,
        )
    with pytest.raises(ValueError, match="only valid for JSON"):
        LLMRequest("operation", (message,), structured_output=spec)


def test_llm_request_validates_limits_and_stop_sequences() -> None:
    message = LLMMessage(LLMRole.USER, "Answer.")

    with pytest.raises(ValueError, match="max_output_tokens"):
        LLMRequest("operation", (message,), max_output_tokens=0)
    with pytest.raises(ValueError, match="temperature"):
        LLMRequest("operation", (message,), temperature=2.1)
    with pytest.raises(ValueError, match="unique"):
        LLMRequest("operation", (message,), stop=("END", "END"))


def test_llm_usage_validates_token_accounting() -> None:
    usage = LLMUsage(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        cached_input_tokens=40,
    )

    assert usage.cached_input_tokens == 40
    with pytest.raises(ValueError, match="total_tokens"):
        LLMUsage(input_tokens=100, output_tokens=20, total_tokens=110)
    with pytest.raises(ValueError, match="cached_input_tokens"):
        LLMUsage(input_tokens=10, cached_input_tokens=11)


def test_deepseek_config_normalizes_and_validates_endpoint() -> None:
    config = DeepSeekBackendConfig(base_url="https://deepseek.example/v1/")

    assert config.base_url == "https://deepseek.example/v1"
    with pytest.raises(ValueError, match="without credentials"):
        DeepSeekBackendConfig(base_url="https://user:secret@deepseek.example")
    with pytest.raises(ValueError, match="timeouts"):
        DeepSeekBackendConfig(connect_timeout_seconds=0)
