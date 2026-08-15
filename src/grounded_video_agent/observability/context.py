"""Task-local trace activation shared across Agent, Tool and provider boundaries."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .contracts import TraceSink

_CURRENT_TRACE_SINK: ContextVar[TraceSink | None] = ContextVar(
    "grounded_video_agent_trace_sink",
    default=None,
)
_CURRENT_TRACE_RUN_ID: ContextVar[str | None] = ContextVar(
    "grounded_video_agent_trace_run_id",
    default=None,
)


def current_trace_sink() -> TraceSink | None:
    return _CURRENT_TRACE_SINK.get()


def trace_is_active() -> bool:
    return current_trace_sink() is not None


def current_trace_run_id() -> str | None:
    return _CURRENT_TRACE_RUN_ID.get()


@contextmanager
def trace_context(sink: TraceSink | None) -> Iterator[None]:
    token = _CURRENT_TRACE_SINK.set(sink)
    try:
        yield
    finally:
        _CURRENT_TRACE_SINK.reset(token)


@contextmanager
def trace_run_context(run_id: str) -> Iterator[None]:
    token = _CURRENT_TRACE_RUN_ID.set(run_id)
    try:
        yield
    finally:
        _CURRENT_TRACE_RUN_ID.reset(token)


def emit_trace(
    event_type: str,
    payload: object,
    *,
    operation_id: str | None = None,
    phase: str | None = None,
) -> None:
    sink = current_trace_sink()
    if sink is not None:
        sink.emit(event_type, payload, operation_id=operation_id, phase=phase)
