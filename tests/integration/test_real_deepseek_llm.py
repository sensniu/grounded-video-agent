from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from grounded_video_agent.infrastructure.llm import (
    DeepSeekBackendConfig,
    DeepSeekLLMBackend,
    LLMMessage,
    LLMRequest,
    LLMRole,
)

load_dotenv()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_deepseek_text_completion() -> None:
    if os.getenv("RUN_DEEPSEEK_INTEGRATION") != "1":
        pytest.skip("set RUN_DEEPSEEK_INTEGRATION=1 to call the paid DeepSeek API")

    backend = DeepSeekLLMBackend(DeepSeekBackendConfig())
    response = await backend.complete(
        LLMRequest(
            "real-deepseek-smoke-test",
            (LLMMessage(LLMRole.USER, "Reply with the single word OK."),),
            max_output_tokens=16,
            temperature=0,
        )
    )

    assert response.content.strip()
    assert response.usage.model_calls >= 1
