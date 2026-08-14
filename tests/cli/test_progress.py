from __future__ import annotations

from io import StringIO

from grounded_video_agent.agent import (
    AgentProgressEvent,
    ProgressCounters,
    ProgressPhase,
    ProgressStatus,
)
from grounded_video_agent.cli.progress import CLIProgressRenderer


def _event(*, status: ProgressStatus = ProgressStatus.COMPLETED) -> AgentProgressEvent:
    return AgentProgressEvent(
        sequence=1,
        run_id="run-1",
        elapsed_ms=65_000,
        phase=ProgressPhase.TOOL,
        status=status,
        message="工具执行完成。",
        counters=ProgressCounters(
            iteration=3,
            max_iterations=18,
            llm_calls=3,
            max_llm_calls=30,
            tool_calls=2,
            max_tool_calls=50,
            input_tokens=95_572,
            output_tokens=6_285,
            max_total_tokens=6_000_000,
            evidence_count=8,
            coverage_ratio=0.4,
        ),
        tool_name="scan_video_timeline",
        details=(("new_evidence", "8"),),
    )


def test_compact_progress_writes_bounded_milestone_to_stderr_stream() -> None:
    stream = StringIO()
    renderer = CLIProgressRenderer("compact", stream=stream, interactive=False)

    with renderer:
        renderer.emit(_event())

    output = stream.getvalue()
    assert "扫描视频时间线" in output
    assert "规划 3/18" in output
    assert "101.9k/6.0M" in output
    assert "new_evidence" not in output


def test_verbose_progress_includes_bounded_details() -> None:
    stream = StringIO()
    renderer = CLIProgressRenderer("verbose", stream=stream, interactive=False)

    with renderer:
        renderer.emit(_event(status=ProgressStatus.WARNING))

    assert "new_evidence=8" in stream.getvalue()


def test_off_progress_emits_nothing() -> None:
    stream = StringIO()
    renderer = CLIProgressRenderer("off", stream=stream, interactive=False)

    with renderer:
        renderer.emit(_event())

    assert stream.getvalue() == ""
