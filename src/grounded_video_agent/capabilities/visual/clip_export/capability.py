from __future__ import annotations

import subprocess
from pathlib import Path
from time import perf_counter

from grounded_video_agent.capabilities._support import file_artifact, make_provenance
from grounded_video_agent.capabilities.visual.clip_export.contracts import ClipExportRequest
from grounded_video_agent.domain import (
    ArtifactKind,
    CapabilityError,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    TimelineMapping,
    TimeRange,
    VideoClipArtifact,
)


class ClipExportCapability:
    VERSION = "1.0.0"

    def __init__(self, output_root: str | Path = "artifacts", *, ffmpeg: str = "ffmpeg") -> None:
        self._output_root = Path(output_root).resolve()
        self._ffmpeg = ffmpeg

    def execute(self, request: ClipExportRequest) -> CapabilityResult[VideoClipArtifact]:
        started = perf_counter()
        clip_id = f"clip_{request.context.operation_id}"
        output = self._output_root / "clips" / request.video_asset.video_id / f"{clip_id}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        provenance = make_provenance(
            "clip-export",
            self.VERSION,
            request,
            video_id=request.video_asset.video_id,
            source_artifact_ids=(request.video_asset.source.artifact_id,),
        )
        command = [
            self._ffmpeg,
            "-v",
            "error",
            "-y",
            "-ss",
            f"{request.time_range.start_ms / 1000:.3f}",
            "-t",
            f"{request.time_range.duration_ms / 1000:.3f}",
            "-i",
            request.video_asset.source.uri,
            "-map",
            "0:v:0",
        ]
        if request.include_audio:
            command += ["-map", "0:a?"]
        else:
            command += ["-an"]
        if request.reencode:
            command += ["-c:v", "libx264"]
            if request.include_audio:
                command += ["-c:a", "aac"]
        else:
            command += ["-c", "copy"]
        command.append(str(output))
        timeout = (request.context.limits.max_wall_time_ms or 120_000) / 1000
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
            return self._failure(result.stderr.strip() or "FFmpeg clip export failed.", started)
        artifact = file_artifact(
            output,
            artifact_id=f"{clip_id}_artifact",
            kind=ArtifactKind.VIDEO_CLIP,
            provenance=provenance,
        )
        clip = VideoClipArtifact(
            clip_id=clip_id,
            video_id=request.video_asset.video_id,
            artifact=artifact,
            requested_range=request.time_range,
            actual_range=request.time_range,
            timeline_mapping=TimelineMapping(
                request.video_asset.video_id,
                request.time_range,
                clip_id,
                TimeRange(0, request.time_range.duration_ms),
            ),
            includes_audio=request.include_audio,
        )
        warnings = (
            ()
            if request.reencode
            else ("Stream-copy clip boundaries may align to keyframes.",)
        )
        return CapabilityResult(
            status=CapabilityStatus.SUCCESS,
            data=clip,
            artifacts=(artifact,),
            warnings=warnings,
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=1,
                output_items=1,
                processed_duration_ms=request.time_range.duration_ms,
            ),
            provenance=provenance,
        )

    @staticmethod
    def _failure(
        message: str,
        started: float,
        *,
        retryable: bool = False,
    ) -> CapabilityResult[VideoClipArtifact]:
        return CapabilityResult(
            status=CapabilityStatus.FAILED,
            data=None,
            error=CapabilityError("CLIP_EXPORT_FAILED", message, "ffmpeg", retryable),
            usage=CapabilityUsage(wall_time_ms=round((perf_counter() - started) * 1000)),
        )
