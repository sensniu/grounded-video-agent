from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

from langgraph.checkpoint.memory import InMemorySaver

from grounded_video_agent.agent.contracts import AgentRequest, AgentResult
from grounded_video_agent.agent.orchestration import AgentDependencies, build_agent_graph
from grounded_video_agent.agent.progress import AgentProgressTracker, ProgressSink
from grounded_video_agent.agent.reasoning import AgentReasoningService
from grounded_video_agent.agent.state import AgentState, initial_agent_state
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

    async def ainvoke(
        self,
        request: AgentRequest,
        *,
        progress: ProgressSink | None = None,
    ) -> AgentResult:
        initial_state = initial_agent_state(request)
        config = {
            "configurable": {"thread_id": request.request_id},
            "recursion_limit": max(50, request.limits.max_iterations * 5 + 20),
        }
        if progress is None:
            final_state = await self._graph.ainvoke(initial_state, config=config)
        else:
            final_state = await self._invoke_with_progress(
                initial_state,
                config,
                AgentProgressTracker(request, progress),
            )
        result = final_state.get("result")
        if not isinstance(result, AgentResult):
            raise RuntimeError("agent graph completed without an AgentResult")
        return result

    async def _invoke_with_progress(
        self,
        initial_state: AgentState,
        config: dict[str, Any],
        tracker: AgentProgressTracker,
    ) -> AgentState:
        state = initial_state
        tracker.start(state)
        try:
            async for chunk in self._graph.astream(
                initial_state,
                config=config,
                stream_mode="updates",
            ):
                if not isinstance(chunk, dict):
                    continue
                for node, raw_update in chunk.items():
                    if not isinstance(node, str) or not isinstance(raw_update, dict):
                        continue
                    update = cast(dict[str, Any], raw_update)
                    state.update(update)  # type: ignore[typeddict-item]
                    tracker.graph_update(node, update, state)
        except Exception as error:
            tracker.failed(state, error)
            raise
        return state

    def invoke(
        self,
        request: AgentRequest,
        *,
        progress: ProgressSink | None = None,
    ) -> AgentResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(request, progress=progress))
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
