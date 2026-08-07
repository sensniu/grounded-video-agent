from __future__ import annotations

import json
import os
from typing import Any

import pytest
from dotenv import load_dotenv

from grounded_video_agent.agent import (
    AgentRequest,
    AgentStatus,
    build_local_video_agent,
)
from grounded_video_agent.infrastructure.llm import (
    LLMFinishReason,
    LLMModelInfo,
    LLMOutputFormat,
    LLMRequest,
    LLMResponse,
    LLMUsage,
)

load_dotenv()


class _EvidenceAwareFakeLLM:
    """Selects transcript search, then cites the first evidence item it receives."""

    def __init__(self) -> None:
        self.call_count = 0

    def get_model_info(self) -> LLMModelInfo:
        return LLMModelInfo("fake", "evidence-aware", True)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        context = _last_json_context(request)
        if "available_tools" in context:
            evidence = context.get("evidence")
            if isinstance(evidence, list) and evidence:
                evidence_id = str(evidence[0]["evidence_id"])
                output: dict[str, Any] = {
                    "intent": "local_event",
                    "current_goal": "Answer from the retrieved transcript",
                    "missing_information": [],
                    "action": "answer",
                    "tool_name": None,
                    "tool_arguments": {},
                    "supporting_evidence_ids": [evidence_id],
                    "expected_information_gain": None,
                    "final_reason": None,
                }
            elif _transcript_search_is_viable(context):
                output = {
                    "intent": "local_event",
                    "current_goal": "Retrieve a representative transcript segment",
                    "missing_information": ["Transcript evidence"],
                    "action": "call_tool",
                    "tool_name": "search_video_transcript",
                    "tool_arguments": {"query": "视频主要内容", "top_k": 5},
                    "supporting_evidence_ids": [],
                    "expected_information_gain": "A relevant subtitle segment",
                    "final_reason": None,
                }
            else:
                output = {
                    "intent": "local_event",
                    "current_goal": "Determine whether grounded evidence is available",
                    "missing_information": ["Transcript or visual evidence"],
                    "action": "abstain",
                    "tool_name": None,
                    "tool_arguments": {},
                    "supporting_evidence_ids": [],
                    "expected_information_gain": None,
                    "final_reason": "No usable transcript or configured visual tool is available.",
                }
        else:
            evidence = context.get("evidence")
            assert isinstance(evidence, list) and evidence
            first = evidence[0]
            evidence_id = str(first["evidence_id"])
            text = str(first.get("text") or "检索到了相关视频证据。")
            output = {
                "answer": text,
                "claims": [
                    {
                        "text": text,
                        "evidence_ids": [evidence_id],
                        "confidence": 0.8,
                        "uncertainties": [],
                    }
                ],
                "limitations": ["This is a deterministic Fake LLM integration test."],
            }
        content = json.dumps(output, ensure_ascii=False)
        return LLMResponse(
            request.operation_id,
            f"fake-{self.call_count}",
            "fake",
            "evidence-aware",
            LLMOutputFormat.JSON_OBJECT,
            content,
            LLMFinishReason.STOP,
            LLMUsage(10, 10, 20),
            1,
            json_object=output,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_video_agent_with_fake_llm() -> None:
    if os.getenv("RUN_VIDEO_AGENT_INTEGRATION") != "1":
        pytest.skip("set RUN_VIDEO_AGENT_INTEGRATION=1 to process the real sample video")

    agent = build_local_video_agent(_EvidenceAwareFakeLLM())
    result = await agent.ainvoke(
        AgentRequest(
            "1sTNqJVrqx8.mp4",
            "这个视频主要讲了什么？",
            request_id="real-video-agent-fake-llm",
        )
    )

    assert result.status in {AgentStatus.SUCCESS, AgentStatus.ABSTAINED}
    assert result.answer
    if result.status is AgentStatus.SUCCESS:
        assert result.citations
    else:
        assert not result.citations


def _last_json_context(request: LLMRequest) -> dict[str, Any]:
    content = request.messages[-1].content
    _, separator, encoded = content.partition("\n")
    if not separator:
        raise AssertionError("reasoning prompt did not contain a JSON context")
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise AssertionError("reasoning context must be a JSON object")
    return value


def _transcript_search_is_viable(context: dict[str, Any]) -> bool:
    video = context.get("video")
    if not isinstance(video, dict) or video.get("transcript_ready") is not True:
        return False
    recent = context.get("recent_tool_results")
    if not isinstance(recent, list):
        return True
    return not any(
        isinstance(item, dict) and item.get("tool_name") == "search_video_transcript"
        for item in recent
    )
