from pathlib import Path
from shutil import which

import pytest

from grounded_video_agent.capabilities.media_inspection import (
    InspectionExecutionStatus,
    MediaInspectionCapability,
    NextAction,
)


@pytest.mark.integration
def test_inspects_repository_sample_video() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_root = project_root / "analyzed_video"
    sample = input_root / "1sTNqJVrqx8.mp4"
    if which("ffprobe") is None or not sample.is_file():
        pytest.skip("local FFprobe or sample video is unavailable")

    result = MediaInspectionCapability(input_root=input_root).inspect(sample.name)

    assert result.execution.status is InspectionExecutionStatus.SUCCEEDED
    assert result.video_context is not None
    assert result.next_action is NextAction.PROCEED
    assert result.video_context.media_probe.container.duration_ms == 299_050
    assert result.video_context.primary_streams.video_stream_index == 0
    assert result.video_context.primary_streams.audio_stream_index == 1
    assert not result.video_context.basic_flags.has_embedded_subtitles
