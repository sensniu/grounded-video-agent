from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Protocol

from grounded_video_agent.agent.reasoning import AgentReasoningService
from grounded_video_agent.agent.tools import (
    ToolRuntimeContext,
    ToolRuntimeSnapshot,
    VideoToolSuite,
)
from grounded_video_agent.agent.verification import EvidenceVerifier
from grounded_video_agent.pipelines import PreprocessingRequest, PreprocessingResult
from grounded_video_agent.workspace.catalog import ArtifactCatalog


class PreprocessingRunner(Protocol):
    def run(self, request: PreprocessingRequest | str) -> PreprocessingResult: ...


class AgentRuntimeRegistry:
    """Keeps ephemeral capability caches while snapshots remain the durable source of truth."""

    def __init__(self) -> None:
        self._items: dict[str, ToolRuntimeContext] = {}
        self._lock = RLock()

    def put(self, run_id: str, runtime: ToolRuntimeContext) -> None:
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        with self._lock:
            self._items[run_id] = runtime

    def resolve(
        self,
        run_id: str,
        snapshot: ToolRuntimeSnapshot,
        catalog: ArtifactCatalog,
    ) -> ToolRuntimeContext:
        with self._lock:
            runtime = self._items.get(run_id)
            if runtime is not None:
                if runtime.video_id != snapshot.video_id:
                    raise ValueError("runtime video_id does not match checkpoint snapshot")
                return runtime
            runtime = ToolRuntimeContext.from_snapshot(snapshot, catalog)
            self._items[run_id] = runtime
            return runtime

    def get(self, run_id: str) -> ToolRuntimeContext | None:
        with self._lock:
            return self._items.get(run_id)


@dataclass(frozen=True, slots=True)
class AgentDependencies:
    pipeline: PreprocessingRunner
    catalog: ArtifactCatalog
    tools: VideoToolSuite
    reasoning: AgentReasoningService
    verifier: EvidenceVerifier = field(default_factory=EvidenceVerifier)
    runtimes: AgentRuntimeRegistry = field(default_factory=AgentRuntimeRegistry)
