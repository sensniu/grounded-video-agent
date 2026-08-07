"""Language-model backend protocol."""

from __future__ import annotations

from typing import Protocol

from grounded_video_agent.infrastructure.llm.contracts import (
    LLMModelInfo,
    LLMRequest,
    LLMResponse,
)


class LLMBackend(Protocol):
    def get_model_info(self) -> LLMModelInfo: ...

    async def complete(self, request: LLMRequest) -> LLMResponse: ...
