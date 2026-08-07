from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which

import pytest

from grounded_video_agent.capabilities.audio.extraction import (
    AudioExtractionCapability,
    AudioExtractionRequest,
)
from grounded_video_agent.capabilities.temporal.shot_detection import (
    ShotDetectionCapability,
    ShotDetectionRequest,
)
from grounded_video_agent.capabilities.visual.clip_export import (
    ClipExportCapability,
    ClipExportRequest,
)
from grounded_video_agent.capabilities.visual.frame_sampling import (
    FrameSamplingCapability,
    FrameSamplingRequest,
)
from grounded_video_agent.domain import (
    ArtifactKind,
    ArtifactRef,
    CapabilityRequestContext,
    CapabilityStatus,
    FrameSamplingStrategy,
    TimeRange,
    VideoAsset,
)


@pytest.fixture
def generated_video(tmp_path: Path) -> VideoAsset:
    if which("ffmpeg") is None:
        pytest.skip("local FFmpeg is unavailable")
    path = tmp_path / "source.mp4"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x120:rate=10:duration=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=16000:duration=2",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    artifact = ArtifactRef(
        "generated-source",
        ArtifactKind.SOURCE_VIDEO,
        str(path),
        size_bytes=path.stat().st_size,
    )
    return VideoAsset("generated-video", artifact, path.name)


@pytest.mark.integration
def test_local_derivation_capabilities_produce_artifacts(
    generated_video: VideoAsset,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    source_range = TimeRange(0, 1_500)

    audio = AudioExtractionCapability(output_root).execute(
        AudioExtractionRequest(
            generated_video,
            source_range,
            1,
            CapabilityRequestContext("audio-integration"),
        )
    )
    clip = ClipExportCapability(output_root).execute(
        ClipExportRequest(
            generated_video,
            TimeRange(250, 1_250),
            CapabilityRequestContext("clip-integration"),
            include_audio=False,
        )
    )
    frames = FrameSamplingCapability(output_root).execute(
        FrameSamplingRequest(
            generated_video,
            (source_range,),
            FrameSamplingStrategy.UNIFORM,
            CapabilityRequestContext("frames-integration"),
            max_frames=3,
            deduplicate=False,
        )
    )
    shots = ShotDetectionCapability(output_root).execute(
        ShotDetectionRequest(
            generated_video,
            source_range,
            CapabilityRequestContext("shots-integration"),
        )
    )

    assert audio.status is CapabilityStatus.SUCCESS
    assert clip.status is CapabilityStatus.SUCCESS
    assert frames.status is CapabilityStatus.SUCCESS
    assert shots.status is CapabilityStatus.SUCCESS
    assert audio.data is not None and Path(audio.data.artifact.uri).is_file()
    assert clip.data is not None and Path(clip.data.artifact.uri).is_file()
    assert frames.data is not None and len(frames.data.frames) == 3
    assert all(Path(frame.image.uri).is_file() for frame in frames.data.frames)
    assert shots.data is not None and shots.data.shots
