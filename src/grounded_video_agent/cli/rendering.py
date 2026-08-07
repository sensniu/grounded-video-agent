from __future__ import annotations

import json

from grounded_video_agent.agent import AgentResult

from .doctor import CheckStatus, DoctorReport


def render_agent_result(result: AgentResult, output_format: str) -> str:
    if output_format == "json":
        return result.to_json()
    lines = [f"状态: {result.status.value}", f"请求 ID: {result.request_id}"]
    if result.video_id:
        lines.append(f"视频 ID: {result.video_id}")
    if result.answer:
        lines.extend(("", "回答:", result.answer))
    if result.citations:
        lines.extend(("", "证据:"))
        for citation in result.citations:
            excerpt = f" — {citation.excerpt}" if citation.excerpt else ""
            lines.append(
                f"- [{_timestamp(citation.time_range.start_ms)}–"
                f"{_timestamp(citation.time_range.end_ms)}] "
                f"{citation.modality} ({citation.evidence_id}){excerpt}"
            )
    if result.attachments:
        lines.extend(("", "附件:"))
        for attachment in result.attachments:
            lines.append(f"- {attachment.filename} ({attachment.size_bytes} bytes)")
    if result.warnings:
        lines.extend(("", "警告:"))
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.error:
        lines.extend(("", f"错误 [{result.error.code}]: {result.error.message}"))
    lines.extend(
        (
            "",
            "用量: "
            f"LLM {result.usage.llm_calls} 次 / Tool {result.usage.tool_calls} 次 / "
            f"输入 {result.usage.input_tokens} tokens / 输出 {result.usage.output_tokens} tokens",
        )
    )
    return "\n".join(lines)


def render_doctor_report(report: DoctorReport, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    labels = {
        CheckStatus.OK: "OK",
        CheckStatus.WARNING: "WARN",
        CheckStatus.ERROR: "ERROR",
        CheckStatus.SKIPPED: "SKIP",
    }
    lines = [
        f"[{labels[check.status]:5}] {check.name}: {check.message}" for check in report.checks
    ]
    lines.append("")
    lines.append("环境检查通过。" if report.ok else "环境检查未通过，请处理 ERROR 项。")
    return "\n".join(lines)


def _timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
