from __future__ import annotations

from typing import Any

from grounded_video_agent.agent.tools.contracts import ToolError, ToolResult, ToolStatus
from grounded_video_agent.agent.tools.runtime import ToolBudgetExceeded, ToolRuntimeContext
from grounded_video_agent.domain import CapabilityResult, CapabilityStatus, CapabilityUsage
from grounded_video_agent.workspace.catalog import CatalogError

SCHEMA_VERSION = "1"


def start_tool(
    runtime: ToolRuntimeContext,
    name: str,
) -> tuple[str | None, ToolResult[Any] | None]:
    try:
        return runtime.start_call(name), None
    except ToolBudgetExceeded as error:
        return None, failed_result(
            f"{name}_budget",
            "TOOL_BUDGET_EXHAUSTED",
            str(error),
            retryable=False,
            suggested_action="Answer with current evidence or abstain.",
        )


def failed_result(
    call_id: str,
    code: str,
    message: str,
    *,
    retryable: bool = False,
    suggested_action: str | None = None,
    usage: CapabilityUsage | None = None,
) -> ToolResult[Any]:
    return ToolResult(
        SCHEMA_VERSION,
        call_id,
        ToolStatus.FAILED,
        None,
        error=ToolError(code, message, retryable, suggested_action),
        usage=usage or CapabilityUsage(),
    )


def catalog_failure(call_id: str, error: CatalogError) -> ToolResult[Any]:
    return failed_result(
        call_id,
        f"CATALOG_{error.code.value.upper()}",
        str(error),
        suggested_action="Run or repair the fixed preprocessing pipeline for this video.",
    )


def capability_failure(
    call_id: str,
    result: CapabilityResult[Any],
) -> ToolResult[Any]:
    assert result.status is CapabilityStatus.FAILED and result.error is not None
    return failed_result(
        call_id,
        result.error.code,
        result.error.message,
        retryable=result.error.retryable,
        suggested_action=result.error.suggested_action,
        usage=result.usage,
    )


def add_usage(*usages: CapabilityUsage) -> CapabilityUsage:
    names = CapabilityUsage.__dataclass_fields__
    return CapabilityUsage(
        **{name: sum(getattr(usage, name) for usage in usages) for name in names}
    )
