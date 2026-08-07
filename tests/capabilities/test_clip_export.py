from __future__ import annotations

import subprocess
from pathlib import Path

from grounded_video_agent.capabilities.visual.clip_export import (
    ClipExportCapability,
    ClipExportRequest,
)
from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    CapabilityRequestContext,
    CapabilityStatus,
    TimeRange,
    VideoAsset,
)


def _asset(tmp_path: Path) -> VideoAsset:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    return VideoAsset(
        "video-clip-test",
        ArtifactRef("source", ArtifactKind.SOURCE_VIDEO, str(source)),
    )


def test_clip_export_validates_then_atomically_publishes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"clip")
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(
            command,
            0,
            '{"format":{"duration":"0.950"},"streams":[{"codec_type":"video"}]}',
            "",
        )

    monkeypatch.setattr(subprocess, "run", run)
    result = ClipExportCapability(tmp_path).execute(
        ClipExportRequest(
            _asset(tmp_path),
            TimeRange(1_000, 2_000),
            CapabilityRequestContext("export"),
        )
    )

    assert result.status is CapabilityStatus.SUCCESS
    assert result.data is not None
    assert result.data.actual_range == TimeRange(1_000, 1_950)
    assert not result.data.includes_audio
    assert Path(result.data.artifact.uri).read_bytes() == b"clip"
    assert any("contains no audio" in warning for warning in result.warnings)
    assert commands[0][-1].endswith(".partial.mp4")
    assert commands[1][0] == "ffprobe"
    assert not tuple(tmp_path.rglob("*.partial.mp4"))


def test_clip_export_removes_partial_output_after_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(command, 1, "", "encoding failed")

    monkeypatch.setattr(subprocess, "run", run)
    result = ClipExportCapability(tmp_path).execute(
        ClipExportRequest(
            _asset(tmp_path),
            TimeRange(0, 1_000),
            CapabilityRequestContext("failed-export"),
        )
    )

    assert result.status is CapabilityStatus.FAILED
    assert result.error is not None
    assert "encoding failed" in result.error.message
    assert not tuple(tmp_path.rglob("*.partial.mp4"))
    assert not tuple((tmp_path / "clips").rglob("clip_failed-export.mp4"))
