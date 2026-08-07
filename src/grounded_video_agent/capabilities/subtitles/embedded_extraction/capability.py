from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path
from time import perf_counter

from grounded_video_agent.capabilities._support import (
    file_artifact,
    make_provenance,
    manifest_ref,
    write_json,
)
from grounded_video_agent.capabilities.subtitles.embedded_extraction.contracts import (
    EmbeddedSubtitleExtractionRequest,
)
from grounded_video_agent.domain import (
    ArtifactKind,
    CapabilityError,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    ManifestKind,
    TimeRange,
    TranscriptManifest,
    TranscriptSegment,
    TranscriptSource,
)

_TIMING = re.compile(
    r"(?P<start>(?:\d{2}:)?\d{2}:\d{2}\.\d{3})\s+-->\s+"
    r"(?P<end>(?:\d{2}:)?\d{2}:\d{2}\.\d{3})"
)
_TAG = re.compile(r"<[^>]+>")


class EmbeddedSubtitleExtractionCapability:
    VERSION = "1.0.0"

    def __init__(self, output_root: str | Path = "artifacts", *, ffmpeg: str = "ffmpeg") -> None:
        self._output_root = Path(output_root).resolve()
        self._ffmpeg = ffmpeg

    def execute(
        self,
        request: EmbeddedSubtitleExtractionRequest,
    ) -> CapabilityResult[TranscriptManifest]:
        started = perf_counter()
        transcript_id = f"embedded_subtitle_{request.context.operation_id}"
        directory = self._output_root / "subtitles" / request.video_asset.video_id
        subtitle_path = directory / f"{transcript_id}.vtt"
        directory.mkdir(parents=True, exist_ok=True)
        provenance = make_provenance(
            "embedded-subtitle-extraction",
            self.VERSION,
            request,
            video_id=request.video_asset.video_id,
            source_artifact_ids=(request.video_asset.source.artifact_id,),
        )
        timeout = (request.context.limits.max_wall_time_ms or 60_000) / 1000
        command = [
            self._ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            request.video_asset.source.uri,
            "-map",
            f"0:{request.stream_index}",
            "-f",
            "webvtt",
            str(subtitle_path),
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
        if result.returncode != 0 or not subtitle_path.is_file():
            return self._failure(result.stderr.strip() or "Subtitle extraction failed.", started)

        segments = self._parse_vtt(
            subtitle_path,
            video_id=request.video_asset.video_id,
            operation_id=request.context.operation_id,
            stream_index=request.stream_index,
            language=request.language,
        )
        subtitle_artifact = file_artifact(
            subtitle_path,
            artifact_id=f"{transcript_id}_vtt",
            kind=ArtifactKind.TRANSCRIPT,
            provenance=provenance,
        )
        manifest_path = directory / f"{transcript_id}.manifest.json"
        ref = manifest_ref(
            manifest_path,
            manifest_id=transcript_id,
            kind=ManifestKind.TRANSCRIPT,
            video_id=request.video_asset.video_id,
            item_count=len(segments),
            provenance=provenance,
        )
        manifest = TranscriptManifest(
            ref=ref,
            video_id=request.video_asset.video_id,
            source=TranscriptSource.EMBEDDED_SUBTITLE,
            segments=segments,
            language=request.language,
        )
        write_json(manifest_path, manifest)
        status = CapabilityStatus.SUCCESS if segments else CapabilityStatus.PARTIAL
        warnings = () if segments else ("The subtitle stream contained no text cues.",)
        return CapabilityResult(
            status=status,
            data=manifest,
            artifacts=(subtitle_artifact, ref.artifact),
            warnings=warnings,
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=1,
                output_items=len(segments),
            ),
            provenance=provenance,
        )

    @staticmethod
    def _parse_vtt(
        path: Path,
        *,
        video_id: str,
        operation_id: str,
        stream_index: int,
        language: str | None,
    ) -> tuple[TranscriptSegment, ...]:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        segments: list[TranscriptSegment] = []
        index = 0
        while index < len(lines):
            match = _TIMING.search(lines[index])
            if match is None:
                index += 1
                continue
            start_ms = _vtt_timestamp_ms(match.group("start"))
            end_ms = _vtt_timestamp_ms(match.group("end"))
            index += 1
            text_lines: list[str] = []
            while index < len(lines) and lines[index].strip():
                text_lines.append(lines[index].strip())
                index += 1
            raw_text = " ".join(text_lines).strip()
            normalized = " ".join(html.unescape(_TAG.sub("", raw_text)).split())
            if normalized and end_ms > start_ms:
                segments.append(
                    TranscriptSegment(
                        segment_id=f"subtitle_{operation_id}_{len(segments):06d}",
                        video_id=video_id,
                        time_range=TimeRange(start_ms, end_ms),
                        raw_text=raw_text,
                        normalized_text=normalized,
                        source=TranscriptSource.EMBEDDED_SUBTITLE,
                        language=language,
                        source_stream_index=stream_index,
                    )
                )
        return tuple(sorted(segments, key=lambda segment: segment.time_range))

    @staticmethod
    def _failure(
        message: str,
        started: float,
        *,
        retryable: bool = False,
    ) -> CapabilityResult[TranscriptManifest]:
        return CapabilityResult(
            status=CapabilityStatus.FAILED,
            data=None,
            error=CapabilityError("SUBTITLE_EXTRACTION_FAILED", message, "ffmpeg", retryable),
            usage=CapabilityUsage(wall_time_ms=round((perf_counter() - started) * 1000)),
        )


def _vtt_timestamp_ms(value: str) -> int:
    parts = value.split(":")
    if len(parts) == 2:
        hours = "0"
        minutes, seconds = parts
    else:
        hours, minutes, seconds = parts
    whole_seconds, milliseconds = seconds.split(".")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(whole_seconds) * 1_000
        + int(milliseconds)
    )
