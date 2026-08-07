from __future__ import annotations

import subprocess
from pathlib import Path
from time import perf_counter

from grounded_video_agent.capabilities._support import file_artifact, make_provenance
from grounded_video_agent.capabilities.audio.extraction.contracts import AudioExtractionRequest
from grounded_video_agent.domain import (
    ArtifactKind,
    AudioArtifact,
    CapabilityError,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    TimelineMapping,
    TimeRange,
)


class AudioExtractionCapability:
    VERSION = "1.0.0"

    def __init__(self, output_root: str | Path = "artifacts", *, ffmpeg: str = "ffmpeg") -> None:
        self._output_root = Path(output_root).resolve()
        self._ffmpeg = ffmpeg

    def execute(self, request: AudioExtractionRequest) -> CapabilityResult[AudioArtifact]:
        started = perf_counter()
        audio_id = f"audio_{request.context.operation_id}"
        output = self._output_root / "audio" / request.video_asset.video_id / f"{audio_id}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        provenance = make_provenance(
            "audio-extraction",
            self.VERSION,
            request,
            video_id=request.video_asset.video_id,
            source_artifact_ids=(request.video_asset.source.artifact_id,),
        )
        timeout = (request.context.limits.max_wall_time_ms or 120_000) / 1000
        command = [
            self._ffmpeg,
            "-v",
            "error",
            "-y",
            "-ss",
            f"{request.source_range.start_ms / 1000:.3f}",
            "-t",
            f"{request.source_range.duration_ms / 1000:.3f}",
            "-i",
            request.video_asset.source.uri,
            "-map",
            f"0:{request.stream_index}",
            "-vn",
            "-ac",
            str(request.channels),
            "-ar",
            str(request.sample_rate_hz),
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return self._failure(
                str(error),
                started,
                retryable=isinstance(error, subprocess.TimeoutExpired),
            )
        if result.returncode != 0 or not output.is_file():
            message = result.stderr.strip() or "FFmpeg audio extraction failed."
            return self._failure(message, started)

        artifact = file_artifact(
            output,
            artifact_id=f"{audio_id}_artifact",
            kind=ArtifactKind.AUDIO,
            provenance=provenance,
        )
        derived_range = TimeRange(0, request.source_range.duration_ms)
        audio = AudioArtifact(
            audio_id=audio_id,
            video_id=request.video_asset.video_id,
            artifact=artifact,
            source_range=request.source_range,
            timeline_mapping=TimelineMapping(
                request.video_asset.video_id,
                request.source_range,
                audio_id,
                derived_range,
            ),
            stream_index=request.stream_index,
            sample_rate_hz=request.sample_rate_hz,
            channels=request.channels,
        )
        return CapabilityResult(
            status=CapabilityStatus.SUCCESS,
            data=audio,
            artifacts=(artifact,),
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=1,
                output_items=1,
                processed_duration_ms=request.source_range.duration_ms,
            ),
            provenance=provenance,
        )

    @staticmethod
    def _failure(
        message: str,
        started: float,
        *,
        retryable: bool = False,
    ) -> CapabilityResult[AudioArtifact]:
        return CapabilityResult(
            status=CapabilityStatus.FAILED,
            data=None,
            error=CapabilityError("AUDIO_EXTRACTION_FAILED", message, "ffmpeg", retryable),
            usage=CapabilityUsage(wall_time_ms=round((perf_counter() - started) * 1000)),
        )
