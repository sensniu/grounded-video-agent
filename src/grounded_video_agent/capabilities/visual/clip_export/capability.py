from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from time import perf_counter
from uuid import uuid4

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

    def __init__(
        self,
        output_root: str | Path = "artifacts",
        *,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
    ) -> None:
        self._output_root = Path(output_root).resolve()
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe

    def execute(self, request: ClipExportRequest) -> CapabilityResult[VideoClipArtifact]:
        started = perf_counter()
        clip_id = f"clip_{request.context.operation_id}"
        output = self._output_root / "clips" / request.video_asset.video_id / f"{clip_id}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output.with_name(
            f".{output.stem}.{uuid4().hex}.partial.mp4"
        )
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
        command.append(str(temporary_output))
        timeout = (request.context.limits.max_wall_time_ms or 120_000) / 1000
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if (
                result.returncode != 0
                or not temporary_output.is_file()
                or temporary_output.stat().st_size <= 0
            ):
                return self._failure(
                    result.stderr.strip() or "FFmpeg clip export failed.",
                    started,
                )
            actual_duration_ms, has_audio = self._probe_output(
                temporary_output,
                timeout,
            )
            os.replace(temporary_output, output)
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            return self._failure(
                str(error),
                started,
                retryable=isinstance(error, subprocess.TimeoutExpired),
            )
        finally:
            temporary_output.unlink(missing_ok=True)
        artifact = file_artifact(
            output,
            artifact_id=f"{clip_id}_artifact",
            kind=ArtifactKind.VIDEO_CLIP,
            provenance=provenance,
        )
        actual_range = TimeRange(
            request.time_range.start_ms,
            request.time_range.start_ms + actual_duration_ms,
        )
        clip = VideoClipArtifact(
            clip_id=clip_id,
            video_id=request.video_asset.video_id,
            artifact=artifact,
            requested_range=request.time_range,
            actual_range=actual_range,
            timeline_mapping=TimelineMapping(
                request.video_asset.video_id,
                actual_range,
                clip_id,
                TimeRange(0, actual_duration_ms),
            ),
            includes_audio=has_audio,
        )
        warnings: tuple[str, ...] = ()
        if not request.reencode:
            warnings += ("Stream-copy clip boundaries may align to keyframes.",)
        if request.include_audio and not has_audio:
            warnings += ("The exported clip contains no audio stream.",)
        if abs(actual_duration_ms - request.time_range.duration_ms) > 250:
            warnings += (
                "The measured output duration differs from the requested duration by more "
                "than 250 ms.",
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
                processed_duration_ms=actual_duration_ms,
            ),
            provenance=provenance,
        )

    def _probe_output(self, path: Path, timeout: float) -> tuple[int, bool]:
        result = subprocess.run(
            [
                self._ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError(result.stderr.strip() or "FFprobe clip validation failed.")
        try:
            payload = json.loads(result.stdout)
            duration_ms = round(float(payload["format"]["duration"]) * 1000)
            streams = payload.get("streams", ())
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid FFprobe clip validation output: {error}") from error
        if duration_ms <= 0:
            raise ValueError("Exported clip duration must be positive.")
        has_audio = any(item.get("codec_type") == "audio" for item in streams)
        return duration_ms, has_audio

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
