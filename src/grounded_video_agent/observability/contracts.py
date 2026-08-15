"""Framework-neutral contracts for structured execution traces."""

from __future__ import annotations

from typing import Protocol


class TraceSink(Protocol):
    def emit(
        self,
        event_type: str,
        payload: object,
        *,
        operation_id: str | None = None,
        phase: str | None = None,
    ) -> None: ...

