from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter

from grounded_video_agent.capabilities._support import make_provenance, manifest_ref, write_json
from grounded_video_agent.capabilities.temporal.chunking.contracts import TemporalChunkingRequest
from grounded_video_agent.domain import (
    CapabilityError,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    Chunk,
    ChunkBasis,
    ChunkManifest,
    ManifestKind,
    Shot,
    TimeRange,
    TranscriptSegment,
)

_SENTENCE_END = re.compile(r"[.!?。！？；;][\"'”’）)]?$")


class TemporalChunkingCapability:
    VERSION = "2.0.0"

    def __init__(self, output_root: str | Path = "artifacts") -> None:
        self._output_root = Path(output_root).resolve()

    def execute(self, request: TemporalChunkingRequest) -> CapabilityResult[ChunkManifest]:
        started = perf_counter()
        segments = self._source_segments(request)
        if segments:
            chunks, warnings = self._transcript_chunks(segments, request)
        else:
            chunks, warnings = self._shot_fallback_chunks(request)
        if not chunks:
            return CapabilityResult(
                status=CapabilityStatus.FAILED,
                data=None,
                error=CapabilityError(
                    "NO_CHUNK_SOURCE",
                    "Neither usable transcript segments nor shots are available.",
                    "chunking",
                ),
                usage=CapabilityUsage(wall_time_ms=round((perf_counter() - started) * 1000)),
            )

        source_ids: list[str] = []
        if request.shots is not None:
            source_ids.append(request.shots.ref.artifact.artifact_id)
        if request.transcript is not None:
            source_ids.append(request.transcript.ref.artifact.artifact_id)
        provenance = make_provenance(
            "temporal-chunking",
            self.VERSION,
            request,
            video_id=request.video_id,
            source_artifact_ids=tuple(source_ids),
        )
        manifest_id = f"chunks_{request.context.operation_id}"
        path = self._output_root / "manifests" / request.video_id / f"{manifest_id}.json"
        ref = manifest_ref(
            path,
            manifest_id=manifest_id,
            kind=ManifestKind.CHUNKS,
            video_id=request.video_id,
            item_count=len(chunks),
            provenance=provenance,
        )
        manifest = ChunkManifest(ref=ref, video_id=request.video_id, chunks=chunks)
        write_json(path, manifest)
        return CapabilityResult(
            status=CapabilityStatus.PARTIAL if warnings else CapabilityStatus.SUCCESS,
            data=manifest,
            artifacts=(ref.artifact,),
            warnings=warnings,
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=len(segments) or (len(request.shots.shots) if request.shots else 0),
                output_items=len(chunks),
                processed_duration_ms=request.source_range.duration_ms,
            ),
            provenance=provenance,
        )

    @staticmethod
    def _source_segments(request: TemporalChunkingRequest) -> tuple[TranscriptSegment, ...]:
        if request.transcript is None:
            return ()
        return tuple(
            segment
            for segment in request.transcript.segments
            if segment.time_range.overlaps(request.source_range)
        )

    def _transcript_chunks(
        self,
        segments: tuple[TranscriptSegment, ...],
        request: TemporalChunkingRequest,
    ) -> tuple[tuple[Chunk, ...], tuple[str, ...]]:
        groups: list[tuple[TranscriptSegment, ...]] = []
        current: list[TranscriptSegment] = []
        warnings: list[str] = []
        for segment in segments:
            if current and self._must_break_before(current, segment, request):
                groups.append(tuple(current))
                current = []
            current.append(segment)
            if self._prefer_break_after(current, request):
                groups.append(tuple(current))
                current = []
        if current:
            groups.append(tuple(current))

        chunks: list[Chunk] = []
        for index, group in enumerate(groups):
            exact_range = TimeRange(
                max(request.source_range.start_ms, group[0].time_range.start_ms),
                min(request.source_range.end_ms, group[-1].time_range.end_ms),
            )
            if exact_range.duration_ms > request.max_duration_ms:
                warnings.append(
                    f"Chunk {index} exceeds max_duration_ms because a subtitle segment "
                    "cannot be split safely."
                )
            text = " ".join(segment.normalized_text.strip() for segment in group).strip()
            shots = self._inspection_shots(exact_range, request)
            inspection_range = self._inspection_range(exact_range, shots, request)
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{request.context.operation_id}_{index:06d}",
                    video_id=request.video_id,
                    time_range=exact_range,
                    shot_ids=tuple(shot.shot_id for shot in shots),
                    transcript_segment_ids=tuple(segment.segment_id for segment in group),
                    inspection_range=inspection_range,
                    text=text,
                    basis=ChunkBasis.TRANSCRIPT,
                )
            )
        if request.shots is None:
            warnings.append("Shot alignment is unavailable; inspection ranges use subtitle time.")
        return tuple(chunks), tuple(dict.fromkeys(warnings))

    @staticmethod
    def _must_break_before(
        current: list[TranscriptSegment],
        segment: TranscriptSegment,
        request: TemporalChunkingRequest,
    ) -> bool:
        previous = current[-1]
        gap_ms = max(0, segment.time_range.start_ms - previous.time_range.end_ms)
        characters = sum(len(item.normalized_text) for item in current) + len(
            segment.normalized_text
        )
        duration_ms = segment.time_range.end_ms - current[0].time_range.start_ms
        return (
            gap_ms > request.max_silence_gap_ms
            or characters > request.max_characters
            or duration_ms > request.max_duration_ms
        )

    @staticmethod
    def _prefer_break_after(
        current: list[TranscriptSegment],
        request: TemporalChunkingRequest,
    ) -> bool:
        characters = sum(len(item.normalized_text) for item in current)
        duration_ms = current[-1].time_range.end_ms - current[0].time_range.start_ms
        reached_target = (
            characters >= request.target_characters
            or duration_ms >= request.target_duration_ms
        )
        return reached_target and _SENTENCE_END.search(current[-1].normalized_text) is not None

    @staticmethod
    def _inspection_shots(
        exact_range: TimeRange,
        request: TemporalChunkingRequest,
    ) -> tuple[Shot, ...]:
        if request.shots is None or not request.align_to_shots:
            return ()
        mandatory = tuple(
            shot for shot in request.shots.shots if shot.time_range.overlaps(exact_range)
        )
        if request.context_padding_ms == 0:
            return mandatory
        padded = TimeRange(
            max(request.source_range.start_ms, exact_range.start_ms - request.context_padding_ms),
            min(request.source_range.end_ms, exact_range.end_ms + request.context_padding_ms),
        )
        selected = list(mandatory)
        for shot in request.shots.shots:
            if shot in selected or not shot.time_range.overlaps(padded):
                continue
            candidate = (*selected, shot)
            start_ms = min(item.time_range.start_ms for item in candidate)
            end_ms = max(item.time_range.end_ms for item in candidate)
            if end_ms - start_ms <= request.max_inspection_duration_ms:
                selected.append(shot)
        return tuple(sorted(selected, key=lambda item: item.time_range))

    @staticmethod
    def _inspection_range(
        exact_range: TimeRange,
        shots: tuple[Shot, ...],
        request: TemporalChunkingRequest,
    ) -> TimeRange:
        if shots:
            return TimeRange(
                max(
                    request.source_range.start_ms,
                    min(exact_range.start_ms, shots[0].time_range.start_ms),
                ),
                min(
                    request.source_range.end_ms,
                    max(exact_range.end_ms, shots[-1].time_range.end_ms),
                ),
            )
        if request.context_padding_ms == 0:
            return exact_range
        padded_start = max(
            request.source_range.start_ms,
            exact_range.start_ms - request.context_padding_ms,
        )
        padded_end = min(
            request.source_range.end_ms,
            exact_range.end_ms + request.context_padding_ms,
        )
        if padded_end - padded_start > request.max_inspection_duration_ms:
            return exact_range
        return TimeRange(padded_start, padded_end)

    @staticmethod
    def _shot_fallback_chunks(
        request: TemporalChunkingRequest,
    ) -> tuple[tuple[Chunk, ...], tuple[str, ...]]:
        if request.shots is None:
            return (), ()
        chunks: list[Chunk] = []
        for shot in request.shots.shots:
            time_range = shot.time_range.intersection(request.source_range)
            if time_range is None:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{request.context.operation_id}_{len(chunks):06d}",
                    video_id=request.video_id,
                    time_range=time_range,
                    shot_ids=(shot.shot_id,),
                    inspection_range=time_range,
                    basis=ChunkBasis.SHOT_FALLBACK,
                )
            )
        warnings = (
            "No usable transcript is available; generated shot-only fallback chunks.",
        )
        return tuple(chunks), warnings
