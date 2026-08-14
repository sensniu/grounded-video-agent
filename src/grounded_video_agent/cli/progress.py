from __future__ import annotations

import sys
from enum import StrEnum
from time import monotonic
from types import TracebackType
from typing import Any, TextIO

from grounded_video_agent.agent import (
    AgentProgressEvent,
    ProgressPhase,
    ProgressStatus,
)

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
except ImportError:  # pragma: no cover - README installs rich; plain output remains usable.
    Console = Live = Panel = Text = None  # type: ignore[assignment,misc]


class ProgressMode(StrEnum):
    AUTO = "auto"
    OFF = "off"
    COMPACT = "compact"
    VERBOSE = "verbose"


_PHASE_LABELS = {
    ProgressPhase.INITIALIZING: "初始化",
    ProgressPhase.PREPROCESSING: "媒体预处理",
    ProgressPhase.PLANNING: "Agent 规划",
    ProgressPhase.TOOL: "工具执行",
    ProgressPhase.EVIDENCE: "证据门禁",
    ProgressPhase.ANSWERING: "回答生成",
    ProgressPhase.VERIFYING: "证据验证",
    ProgressPhase.DELIVERY: "结果交付",
    ProgressPhase.COMPLETE: "完成",
}

_TOOL_LABELS = {
    "get_video_metadata": "读取媒体信息",
    "search_video_transcript": "检索字幕",
    "expand_timeline_context": "扩展时间线上下文",
    "inspect_visual_content": "分析指定画面",
    "read_screen_text": "识别画面文字",
    "scan_video_timeline": "扫描视频时间线",
    "export_evidence_clip": "导出证据片段",
}

_STATUS_ICONS = {
    ProgressStatus.STARTED: "…",
    ProgressStatus.INFO: "·",
    ProgressStatus.COMPLETED: "✓",
    ProgressStatus.WARNING: "!",
    ProgressStatus.FAILED: "✗",
}

_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧")


class _DynamicLiveView:
    def __init__(self, renderer: CLIProgressRenderer) -> None:
        self._renderer = renderer

    def __rich__(self) -> Any:
        return self._renderer._render_current_live()


class CLIProgressRenderer:
    def __init__(
        self,
        mode: str | ProgressMode = ProgressMode.AUTO,
        *,
        stream: TextIO | None = None,
        interactive: bool | None = None,
    ) -> None:
        self.mode = ProgressMode(mode)
        self._stream = stream or sys.stderr
        detected = bool(getattr(self._stream, "isatty", lambda: False)())
        self._interactive = detected if interactive is None else interactive
        self._live: Any = None
        self._last_event: AgentProgressEvent | None = None
        self._started_at: float | None = None
        self._recent_notice: str | None = None

    @property
    def enabled(self) -> bool:
        return self.mode is not ProgressMode.OFF

    def __enter__(self) -> CLIProgressRenderer:
        if self.enabled and self._interactive and Live is not None and Console is not None:
            console = Console(file=self._stream, force_terminal=True)
            self._live = Live(
                _DynamicLiveView(self),
                console=console,
                refresh_per_second=4,
                transient=False,
            )
            self._live.start()  # type: ignore[union-attr]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._live is not None:
            self._live.stop()  # type: ignore[union-attr]
            self._live = None

    def emit(self, event: AgentProgressEvent) -> None:
        if not self.enabled:
            return
        self._last_event = event
        if self._started_at is None:
            self._started_at = monotonic() - event.elapsed_ms / 1_000
        if event.status in {ProgressStatus.WARNING, ProgressStatus.FAILED}:
            self._recent_notice = event.message
        if self._live is not None:
            self._live.refresh()
            return
        if self.mode is ProgressMode.VERBOSE or _is_compact_milestone(event):
            print(_render_line(event, verbose=self.mode is ProgressMode.VERBOSE), file=self._stream)

    @staticmethod
    def _render_live_placeholder() -> Any:
        if Panel is None:
            return "正在启动视频分析…"
        return Panel("正在启动视频分析…", title="Grounded Video Agent")

    def _render_current_live(self) -> Any:
        event = self._last_event
        if event is None:
            return self._render_live_placeholder()
        if Panel is None or Text is None:
            return _render_line(event, verbose=self.mode is ProgressMode.VERBOSE)
        counters = event.counters
        elapsed_ms = event.elapsed_ms
        if self._started_at is not None:
            elapsed_ms = round((monotonic() - self._started_at) * 1_000)
        elapsed = _elapsed(elapsed_ms)
        phase = _PHASE_LABELS[event.phase]
        tool = _TOOL_LABELS.get(event.tool_name or "", event.tool_name or "-")
        activity = (
            _SPINNER_FRAMES[int(monotonic() * 8) % len(_SPINNER_FRAMES)]
            if event.status is ProgressStatus.STARTED
            else _STATUS_ICONS[event.status]
        )
        coverage = (
            f"{counters.coverage_ratio:.0%}"
            if counters.coverage_ratio is not None
            else "未知"
        )
        lines = [
            f"阶段  {activity} {phase}    用时 {elapsed}",
            f"当前  {event.message}",
            (
                f"进度  规划 {counters.iteration}/{counters.max_iterations} · "
                f"Tool {counters.tool_calls}/{counters.max_tool_calls} · "
                f"LLM {counters.llm_calls}/{counters.max_llm_calls}"
            ),
            (
                f"预算  {_format_tokens(counters.total_tokens)} / "
                f"{_format_tokens(counters.max_total_tokens)} tokens"
            ),
            f"证据  {counters.evidence_count} 条 · 时间线覆盖 {coverage}",
        ]
        if event.tool_name is not None:
            lines.insert(2, f"工具  {tool}")
        if self.mode is ProgressMode.VERBOSE and event.details:
            details = " · ".join(f"{key}={value}" for key, value in event.details)
            lines.append(f"详情  {details}")
        if self._recent_notice is not None:
            lines.append(f"提示  {self._recent_notice}")
        return Panel(Text("\n".join(lines)), title="Grounded Video Agent", border_style="cyan")


def _is_compact_milestone(event: AgentProgressEvent) -> bool:
    if event.status in {ProgressStatus.WARNING, ProgressStatus.FAILED}:
        return True
    if event.phase in {ProgressPhase.INITIALIZING, ProgressPhase.COMPLETE}:
        return True
    if event.phase is ProgressPhase.PLANNING:
        return event.status is ProgressStatus.COMPLETED
    if event.phase is ProgressPhase.TOOL:
        return True
    if event.phase is ProgressPhase.PREPROCESSING:
        return True
    return event.status is ProgressStatus.COMPLETED


def _render_line(event: AgentProgressEvent, *, verbose: bool) -> str:
    counters = event.counters
    icon = _STATUS_ICONS[event.status]
    phase = _PHASE_LABELS[event.phase]
    parts = [f"[{_elapsed(event.elapsed_ms)}] {icon} {phase}: {event.message}"]
    if event.tool_name is not None:
        parts.append(_TOOL_LABELS.get(event.tool_name, event.tool_name))
    parts.append(
        f"规划 {counters.iteration}/{counters.max_iterations} · "
        f"Tool {counters.tool_calls}/{counters.max_tool_calls} · "
        f"LLM {counters.llm_calls}/{counters.max_llm_calls} · "
        f"{_format_tokens(counters.total_tokens)}/{_format_tokens(counters.max_total_tokens)}"
    )
    if verbose and event.details:
        parts.append(" ".join(f"{key}={value}" for key, value in event.details))
    return " | ".join(parts)


def _elapsed(milliseconds: int) -> str:
    seconds = max(0, milliseconds // 1_000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)
