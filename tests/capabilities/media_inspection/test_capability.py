import json
from pathlib import Path

from grounded_video_agent.capabilities.media_inspection import (
    InspectionErrorCode,
    InspectionExecutionStatus,
    MediaInspectionCapability,
    NextAction,
)
from grounded_video_agent.capabilities.media_inspection.ffprobe import (
    FFprobeError,
    FFprobeErrorCode,
    RawProbeResult,
)

from .sample_payload import sample_ffprobe_payload


class FakeProbeRunner:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[Path] = []

    def probe(self, source_path: str | Path) -> RawProbeResult:
        self.calls.append(Path(source_path))
        return RawProbeResult(payload=self.payload, stderr="", duration_ms=1)


class FailingProbeRunner:
    def probe(self, source_path: str | Path) -> RawProbeResult:
        raise FFprobeError(
            FFprobeErrorCode.TIMEOUT,
            "probe timed out",
            retryable=True,
        )


def test_capability_registers_filename_and_returns_agent_ready_json(tmp_path: Path) -> None:
    (tmp_path / "sample.mp4").write_bytes(b"video")
    runner = FakeProbeRunner(sample_ffprobe_payload())
    capability = MediaInspectionCapability(input_root=tmp_path, probe_runner=runner)

    result = capability.inspect("sample.mp4")
    payload = result.to_dict()

    assert result.execution.status is InspectionExecutionStatus.SUCCEEDED
    assert result.video_context is not None
    assert result.next_action is NextAction.PROCEED
    assert result.video_context.basic_flags.has_embedded_subtitles
    assert runner.calls == [(tmp_path / "sample.mp4").resolve()]
    assert payload["registration"]["status"] == "succeeded"
    assert payload["video_context"]["validation"]["status"] == "valid"
    assert payload["video_context"]["basic_flags"]["has_audio"] is True
    assert json.loads(result.to_json())["schema_version"] == "1"


def test_registration_failure_does_not_call_probe_runner(tmp_path: Path) -> None:
    runner = FakeProbeRunner(sample_ffprobe_payload())
    capability = MediaInspectionCapability(input_root=tmp_path, probe_runner=runner)

    result = capability.inspect("../outside.mp4")

    assert result.execution.status is InspectionExecutionStatus.FAILED
    assert result.error is not None
    assert result.error.code is InspectionErrorCode.INVALID_FILENAME
    assert runner.calls == []


def test_probe_failure_is_returned_as_structured_error(tmp_path: Path) -> None:
    (tmp_path / "sample.mp4").write_bytes(b"video")
    capability = MediaInspectionCapability(
        input_root=tmp_path,
        probe_runner=FailingProbeRunner(),
    )

    result = capability.inspect("sample.mp4")

    assert result.execution.status is InspectionExecutionStatus.FAILED
    assert result.error is not None
    assert result.error.code is InspectionErrorCode.FFPROBE_TIMEOUT
    assert result.error.retryable
    assert result.next_action is NextAction.RETRY_INSPECTION
