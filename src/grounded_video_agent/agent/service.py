from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from grounded_video_agent.agent.contracts import AgentRequest, AgentResult
from grounded_video_agent.agent.orchestration import AgentDependencies, build_agent_graph
from grounded_video_agent.agent.reasoning import AgentReasoningService
from grounded_video_agent.agent.state import initial_agent_state
from grounded_video_agent.agent.tools import build_video_tool_suite
from grounded_video_agent.agent.verification import EvidenceVerifier
from grounded_video_agent.infrastructure.embeddings import TextEmbeddingBackend
from grounded_video_agent.infrastructure.llm import LLMBackend
from grounded_video_agent.infrastructure.ocr import OCRBackend
from grounded_video_agent.infrastructure.visual_model import VisualModelBackend
from grounded_video_agent.pipelines import (
    PreprocessingConfig,
    build_local_preprocessing_pipeline,
)


class VideoAgent:
    def __init__(
        self,
        dependencies: AgentDependencies,
        *,
        checkpointer: Any = None,
    ) -> None:
        self._dependencies = dependencies
        self._checkpointer = checkpointer or InMemorySaver()
        self._graph = build_agent_graph(
            dependencies,
            checkpointer=self._checkpointer,
        )

    @property
    def dependencies(self) -> AgentDependencies:
        return self._dependencies

    async def ainvoke(self, request: AgentRequest) -> AgentResult:
        final_state = await self._graph.ainvoke(
            initial_agent_state(request),
            config={
                "configurable": {"thread_id": request.request_id},
                "recursion_limit": max(50, request.limits.max_iterations * 5 + 20),
            },
        )
        result = final_state.get("result")
        if not isinstance(result, AgentResult):
            raise RuntimeError("agent graph completed without an AgentResult")
        return result

    def invoke(self, request: AgentRequest) -> AgentResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(request))
        raise RuntimeError("VideoAgent.invoke cannot run inside an event loop; use ainvoke")

    @property
    def graph(self) -> Any:
        return self._graph


def build_local_video_agent(
    llm_backend: LLMBackend,
    *,
    input_root: str | Path = "analyzed_video",
    artifact_root: str | Path = "artifacts",
    catalog_root: str | Path | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    embedding_backend: TextEmbeddingBackend | None = None,
    visual_backend: VisualModelBackend | None = None,
    ocr_backend: OCRBackend | None = None,
    checkpointer: Any = None,
) -> VideoAgent:
    pipeline = build_local_preprocessing_pipeline(
        input_root=input_root,
        artifact_root=artifact_root,
        catalog_root=catalog_root,
        config=preprocessing_config,
        embedding_backend=embedding_backend,
    )
    dependencies = AgentDependencies(
        pipeline=pipeline,
        catalog=pipeline.catalog,
        tools=build_video_tool_suite(
            artifact_root=artifact_root,
            embedding_backend=embedding_backend,
            visual_backend=visual_backend,
            ocr_backend=ocr_backend,
        ),
        reasoning=AgentReasoningService(llm_backend),
        verifier=EvidenceVerifier(),
    )
    return VideoAgent(dependencies, checkpointer=checkpointer)
