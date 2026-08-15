from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from grounded_video_agent.observability import (
    JsonlTraceRecorder,
    emit_trace,
    trace_context,
    trace_run_context,
)


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_jsonl_recorder_writes_ordered_events_and_redacts_credentials(tmp_path: Path) -> None:
    recorder = JsonlTraceRecorder.create(
        tmp_path,
        now=datetime(2026, 8, 16, 12, 34, 56, 123456, tzinfo=UTC),
    )

    with recorder, trace_context(recorder), trace_run_context("agent-test"):
        emit_trace(
            "llm.request",
            {
                "question": "What happened?",
                "DEEPSEEK_API_KEY": "secret-key",
                "headers": {"Authorization": "Bearer secret-key"},
                "max_tokens": 64_000,
            },
            operation_id="operation-1",
            phase="reasoning",
        )
        emit_trace("llm.response", {"content": "answer"}, operation_id="operation-1")

    assert re.fullmatch(r"20260816_123456_123456\.jsonl", recorder.path.name)
    events = _events(recorder.path)
    assert [event["sequence"] for event in events] == [1, 2]
    assert events[0]["request_id"] == "agent-test"
    assert events[0]["run_id"] == "agent-test"
    assert events[0]["operation_id"] == "operation-1"
    assert events[0]["phase"] == "reasoning"
    payload = events[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["DEEPSEEK_API_KEY"] == "[REDACTED]"
    assert payload["headers"] == {"Authorization": "[REDACTED]"}
    assert payload["max_tokens"] == 64_000
    assert "secret-key" not in recorder.path.read_text(encoding="utf-8")


def test_emit_trace_is_a_noop_without_an_active_recorder(tmp_path: Path) -> None:
    emit_trace("ignored", {"path": tmp_path})
    assert not tuple(tmp_path.iterdir())

