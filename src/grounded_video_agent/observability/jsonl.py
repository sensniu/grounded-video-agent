"""One-file-per-run JSONL trace recorder."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import TextIO

from .context import current_trace_run_id
from .serialization import trace_json_value


class JsonlTraceRecorder:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path, stream: TextIO) -> None:
        self.path = path
        self._stream = stream
        self._started = perf_counter()
        self._sequence = 0
        self._lock = RLock()
        self._closed = False
        self._error: str | None = None

    @classmethod
    def create(
        cls,
        root: str | Path = "agent_traces",
        *,
        now: datetime | None = None,
    ) -> JsonlTraceRecorder:
        trace_root = Path(root).expanduser().resolve()
        trace_root.mkdir(parents=True, exist_ok=True)
        local_now = now or datetime.now().astimezone()
        stem = local_now.strftime("%Y%m%d_%H%M%S_%f")
        for suffix in range(1_000):
            filename = f"{stem}.jsonl" if suffix == 0 else f"{stem}_{suffix:03d}.jsonl"
            path = trace_root / filename
            try:
                stream = path.open("x", encoding="utf-8")
            except FileExistsError:
                continue
            return cls(path, stream)
        raise FileExistsError(f"could not allocate a unique trace filename under {trace_root}")

    @property
    def error(self) -> str | None:
        return self._error

    def emit(
        self,
        event_type: str,
        payload: object,
        *,
        operation_id: str | None = None,
        phase: str | None = None,
    ) -> None:
        if not event_type.strip():
            return
        with self._lock:
            if self._closed or self._error is not None:
                return
            try:
                self._sequence += 1
                run_id = current_trace_run_id()
                event = {
                    "schema_version": self.SCHEMA_VERSION,
                    "sequence": self._sequence,
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "elapsed_ms": round((perf_counter() - self._started) * 1_000),
                    "event_type": event_type,
                    "request_id": run_id,
                    "run_id": run_id,
                    "operation_id": operation_id,
                    "phase": phase,
                    "payload": trace_json_value(payload),
                }
                self._stream.write(
                    json.dumps(
                        event,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                self._stream.flush()
            except Exception as error:  # tracing must not terminate the Agent
                self._error = f"{type(error).__name__}: {error}"

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._stream.close()
            except Exception as error:  # pragma: no cover - platform dependent I/O failure
                if self._error is None:
                    self._error = f"{type(error).__name__}: {error}"
            self._closed = True

    def __enter__(self) -> JsonlTraceRecorder:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
