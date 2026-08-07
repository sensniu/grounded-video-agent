"""Framework-neutral registry for the six LLM-visible video tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

from pydantic import TypeAdapter, ValidationError

from grounded_video_agent.agent.tools._support import failed_result
from grounded_video_agent.agent.tools.contracts import ToolResult, ToolSpec
from grounded_video_agent.agent.tools.runtime import ToolRuntimeContext


class RegisteredVideoTool(Protocol):
    name: str
    description: str
    input_type: type[Any]
    enabled: bool

    def execute(self, request: Any, runtime: ToolRuntimeContext) -> ToolResult[Any]: ...


@dataclass(frozen=True, slots=True)
class VideoToolSuite:
    tools: tuple[RegisteredVideoTool, ...]

    def __post_init__(self) -> None:
        names = tuple(tool.name for tool in self.tools)
        if len(set(names)) != len(names):
            raise ValueError("tool names must be unique")

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return self._specs(self.tools)

    @property
    def available_specs(self) -> tuple[ToolSpec, ...]:
        """Definitions safe to register for the configured local backends."""

        return self._specs(
            tuple(
                tool
                for tool in self.tools
                if tool.enabled and not getattr(tool, "runtime_guarded", False)
            )
        )

    def available_specs_for(self, runtime: ToolRuntimeContext) -> tuple[ToolSpec, ...]:
        """Definitions enabled by both configuration and per-request policy."""

        available: list[RegisteredVideoTool] = []
        for tool in self.tools:
            if not tool.enabled:
                continue
            is_available = getattr(tool, "is_available", None)
            if is_available is None or is_available(runtime):
                available.append(tool)
        return self._specs(tuple(available))

    @staticmethod
    def _specs(tools: tuple[RegisteredVideoTool, ...]) -> tuple[ToolSpec, ...]:
        return tuple(
            ToolSpec(
                tool.name,
                tool.description,
                cast(dict[str, Any], TypeAdapter(tool.input_type).json_schema()),
            )
            for tool in tools
        )

    def invoke(
        self,
        name: str,
        arguments: object,
        runtime: ToolRuntimeContext,
    ) -> ToolResult[Any]:
        tool = next((item for item in self.tools if item.name == name), None)
        if tool is None:
            return failed_result(
                f"unknown_tool_{runtime.call_count + 1:04d}",
                "UNKNOWN_TOOL",
                f"Unknown video tool: {name}",
            )
        try:
            request = TypeAdapter(tool.input_type).validate_python(arguments)
        except ValidationError as error:
            return failed_result(
                f"{name}_invalid_input_{runtime.call_count + 1:04d}",
                "INVALID_TOOL_INPUT",
                str(error),
            )
        return tool.execute(request, runtime)
