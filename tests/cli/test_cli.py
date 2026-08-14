from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path

from grounded_video_agent.agent import (
    AgentLimits,
    AgentProgressEvent,
    AgentRequest,
    AgentResult,
    AgentStatus,
    ProgressCounters,
    ProgressPhase,
    ProgressSink,
    ProgressStatus,
)
from grounded_video_agent.cli.adapter import AgentInvoker
from grounded_video_agent.cli.config import CLIRuntimeSettings, OCRProvider, VLMProvider
from grounded_video_agent.cli.doctor import (
    CheckStatus,
    DoctorCheck,
    DoctorReport,
)
from grounded_video_agent.cli.main import run


class _FakeAgent:
    def __init__(
        self,
        result: AgentResult,
        progress_events: tuple[AgentProgressEvent, ...] = (),
    ) -> None:
        self.result = result
        self.progress_events = progress_events
        self.requests: list[AgentRequest] = []

    def invoke(
        self,
        request: AgentRequest,
        *,
        progress: ProgressSink | None = None,
    ) -> AgentResult:
        self.requests.append(request)
        if progress is not None:
            for event in self.progress_events:
                progress(event)
        return self.result


def _namespace(**values: object) -> Namespace:
    defaults: dict[str, object] = {
        "input_root": None,
        "artifact_root": None,
        "llm_model": None,
        "llm_base_url": None,
        "ocr": None,
        "vlm": None,
        "vlm_url": None,
        "vlm_model": None,
        "progress": "auto",
    }
    defaults.update(values)
    return Namespace(**defaults)


def _successful_result() -> AgentResult:
    return AgentResult(
        request_id="request-1",
        status=AgentStatus.SUCCESS,
        video_id="video-1",
        answer="视频展示了一个测试场景。",
    )


def test_settings_precedence_is_cli_then_environment_then_default(tmp_path: Path) -> None:
    settings = CLIRuntimeSettings.from_namespace(
        _namespace(input_root=str(tmp_path / "cli-input"), ocr="rapidocr"),
        {
            "GVA_INPUT_ROOT": str(tmp_path / "env-input"),
            "GVA_ARTIFACT_ROOT": str(tmp_path / "env-artifacts"),
            "GVA_DEEPSEEK_MODEL": "env-model",
            "GVA_VLM_BACKEND": "llama-cpp",
        },
    )

    assert settings.input_root == (tmp_path / "cli-input").resolve()
    assert settings.artifact_root == (tmp_path / "env-artifacts").resolve()
    assert settings.deepseek_model == "env-model"
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.ocr_provider is OCRProvider.RAPIDOCR
    assert settings.vlm_provider is VLMProvider.LLAMA_CPP


def test_settings_default_to_fastapi_visual_service(tmp_path: Path) -> None:
    settings = CLIRuntimeSettings.from_namespace(
        _namespace(),
        {
            "GVA_ARTIFACT_ROOT": str(tmp_path),
            "GVA_FASTAPI_VLM_BASE_URL": "http://visual-api.test:8081",
        },
    )

    assert settings.vlm_provider is VLMProvider.FASTAPI
    assert settings.selected_vlm_base_url == "http://visual-api.test:8081"


def test_agent_default_resource_limits_are_sized_for_long_video_analysis() -> None:
    limits = AgentLimits()

    assert limits.max_iterations == 50
    assert limits.max_tool_calls == 100
    assert limits.max_llm_calls == 60
    assert limits.max_total_tokens == 6_000_000


def test_progress_uses_stderr_without_corrupting_json_stdout(capsys: object) -> None:
    event = AgentProgressEvent(
        1,
        "request-1",
        1_000,
        ProgressPhase.INITIALIZING,
        ProgressStatus.STARTED,
        "开始分析视频。",
        ProgressCounters(0, 50, 0, 60, 0, 100, 0, 0, 6_000_000),
    )
    fake_agent = _FakeAgent(_successful_result(), (event,))

    exit_code = run(
        [
            "analyze",
            "video.mp4",
            "-q",
            "发生了什么？",
            "--format",
            "json",
            "--progress",
            "compact",
        ],
        environ={"DEEPSEEK_API_KEY": "test-key"},
        agent_factory=lambda settings: fake_agent,
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 0
    assert json.loads(captured.out)["status"] == "success"
    assert "初始化" in captured.err


def test_analyze_maps_cli_arguments_to_agent_request(capsys: object) -> None:
    fake_agent = _FakeAgent(_successful_result())
    captured_settings: list[CLIRuntimeSettings] = []

    def factory(settings: CLIRuntimeSettings) -> AgentInvoker:
        captured_settings.append(settings)
        return fake_agent

    exit_code = run(
        [
            "analyze",
            "video.mp4",
            "-q",
            "发生了什么？",
            "--request-id",
            "request-1",
            "--evidence-clip",
            "--max-tool-calls",
            "4",
            "--format",
            "json",
        ],
        environ={"DEEPSEEK_API_KEY": "test-key"},
        agent_factory=factory,
    )

    assert exit_code == 0
    assert len(captured_settings) == 1
    request = fake_agent.requests[0]
    assert request.filename == "video.mp4"
    assert request.question == "发生了什么？"
    assert request.request_id == "request-1"
    assert request.evidence_clip_requested is True
    assert request.limits.max_tool_calls == 4
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert json.loads(output)["status"] == "success"


def test_analyze_rejects_paths_before_building_agent(capsys: object) -> None:
    called = False

    def factory(settings: CLIRuntimeSettings) -> AgentInvoker:
        nonlocal called
        called = True
        raise AssertionError(settings)

    exit_code = run(
        ["analyze", "../video.mp4", "-q", "question"],
        environ={"DEEPSEEK_API_KEY": "test-key"},
        agent_factory=factory,
    )

    assert exit_code == 3
    assert called is False
    assert "plain filename" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_analyze_rejects_existing_output_before_building_agent(
    tmp_path: Path,
    capsys: object,
) -> None:
    output = tmp_path / "result.json"
    output.write_text("keep", encoding="utf-8")
    called = False

    def factory(settings: CLIRuntimeSettings) -> AgentInvoker:
        nonlocal called
        called = True
        raise AssertionError(settings)

    exit_code = run(
        [
            "analyze",
            "video.mp4",
            "-q",
            "question",
            "--output",
            str(output),
        ],
        environ={"DEEPSEEK_API_KEY": "test-key"},
        agent_factory=factory,
    )

    assert exit_code == 3
    assert called is False
    assert output.read_text(encoding="utf-8") == "keep"
    assert "already exists" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_doctor_does_not_build_agent_and_can_skip_network(capsys: object) -> None:
    received: list[bool] = []

    def doctor_runner(
        settings: CLIRuntimeSettings,
        environ: Mapping[str, str],
        check_network: bool,
    ) -> DoctorReport:
        assert settings.vlm_provider is VLMProvider.LLAMA_CPP
        assert environ["DEEPSEEK_API_KEY"] == "test-key"
        received.append(check_network)
        return DoctorReport((DoctorCheck("environment", CheckStatus.OK, "ready"),))

    def forbidden_factory(settings: CLIRuntimeSettings) -> AgentInvoker:
        raise AssertionError(settings)

    exit_code = run(
        ["doctor", "--vlm", "llama-cpp", "--no-network", "--format", "json"],
        environ={"DEEPSEEK_API_KEY": "test-key"},
        agent_factory=forbidden_factory,
        doctor_runner=doctor_runner,
    )

    assert exit_code == 0
    assert received == [False]
    assert json.loads(capsys.readouterr().out)["ok"] is True  # type: ignore[attr-defined]
