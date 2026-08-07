from __future__ import annotations

from pathlib import Path
from time import perf_counter

from grounded_video_agent.capabilities._support import make_provenance, manifest_ref, write_json
from grounded_video_agent.capabilities.subtitles.speech_transcription.backend import (
    ASRSegment,
    FasterWhisperBackend,
    TranscriptionBackend,
)
from grounded_video_agent.capabilities.subtitles.speech_transcription.contracts import (
    SpeechTranscriptionRequest,
)
from grounded_video_agent.domain import (
    CapabilityError,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    ManifestKind,
    TimeRange,
    TranscriptManifest,
    TranscriptSegment,
    TranscriptSource,
    TranscriptWord,
)


class SpeechTranscriptionCapability:
    VERSION = "1.0.0"

    def __init__(
        self,
        output_root: str | Path = "artifacts",
        *,
        backend: TranscriptionBackend | None = None,
    ) -> None:
        self._output_root = Path(output_root).resolve()
        self._backend = backend or FasterWhisperBackend()

    def cache_identity(self) -> object:
        identity = getattr(self._backend, "cache_identity", None)
        if callable(identity):
            return identity()
        backend_type = type(self._backend)
        return {"backend": f"{backend_type.__module__}.{backend_type.__qualname__}"}

    def execute(self, request: SpeechTranscriptionRequest) -> CapabilityResult[TranscriptManifest]:
        started = perf_counter()
        try:
            transcript = self._backend.transcribe(
                request.audio.artifact.uri,
                language=request.language_hint,
                word_timestamps=request.word_timestamps,
            )
            converted = (
                self._segment(index, segment, request, transcript.language)
                for index, segment in enumerate(transcript.segments)
            )
            segments = tuple(
                sorted(
                    (segment for segment in converted if segment is not None),
                    key=lambda segment: segment.time_range,
                )
            )
        except Exception as error:
            return CapabilityResult(
                status=CapabilityStatus.FAILED,
                data=None,
                error=CapabilityError("TRANSCRIPTION_FAILED", str(error), "asr", True),
                usage=CapabilityUsage(wall_time_ms=round((perf_counter() - started) * 1000)),
            )

        provenance = make_provenance(
            "speech-transcription",
            self.VERSION,
            request,
            video_id=request.audio.video_id,
            source_artifact_ids=(request.audio.artifact.artifact_id,),
        )
        manifest_id = f"asr_{request.context.operation_id}"
        path = self._output_root / "transcripts" / request.audio.video_id / f"{manifest_id}.json"
        ref = manifest_ref(
            path,
            manifest_id=manifest_id,
            kind=ManifestKind.TRANSCRIPT,
            video_id=request.audio.video_id,
            item_count=len(segments),
            provenance=provenance,
        )
        manifest = TranscriptManifest(
            ref=ref,
            video_id=request.audio.video_id,
            source=TranscriptSource.ASR,
            segments=segments,
            language=transcript.language,
        )
        write_json(path, manifest)
        status = CapabilityStatus.SUCCESS if segments else CapabilityStatus.PARTIAL
        warnings = () if segments else ("ASR produced no speech segments.",)
        return CapabilityResult(
            status=status,
            data=manifest,
            artifacts=(ref.artifact,),
            warnings=warnings,
            usage=CapabilityUsage(
                wall_time_ms=round((perf_counter() - started) * 1000),
                input_items=1,
                output_items=len(segments),
                processed_duration_ms=request.audio.source_range.duration_ms,
                model_calls=1,
            ),
            provenance=provenance,
        )

    @staticmethod
    def _segment(
        index: int,
        segment: ASRSegment,
        request: SpeechTranscriptionRequest,
        language: str | None,
    ) -> TranscriptSegment | None:
        if not segment.text.strip() or segment.end_seconds <= segment.start_seconds:
            return None
        source_start = request.audio.source_range.start_ms
        source_end = request.audio.source_range.end_ms
        start_ms = max(source_start, source_start + round(segment.start_seconds * 1000))
        end_ms = min(source_end, source_start + round(segment.end_seconds * 1000))
        if start_ms >= end_ms:
            return None
        words: list[TranscriptWord] = []
        for word in segment.words:
            word_start = max(start_ms, source_start + round(word.start_seconds * 1000))
            word_end = min(end_ms, source_start + round(word.end_seconds * 1000))
            text = word.text.strip()
            if not text or word_start >= word_end:
                continue
            words.append(
                TranscriptWord(
                    text=text,
                    time_range=TimeRange(word_start, word_end),
                    confidence=word.probability,
                )
            )
        ordered_words = tuple(sorted(words, key=lambda word: word.time_range))
        confidence = None
        probabilities = [
            word.confidence for word in ordered_words if word.confidence is not None
        ]
        if probabilities:
            confidence = sum(probabilities) / len(probabilities)
        raw = segment.text.strip()
        return TranscriptSegment(
            segment_id=f"asr_{request.context.operation_id}_{index:06d}",
            video_id=request.audio.video_id,
            time_range=TimeRange(start_ms, end_ms),
            raw_text=raw,
            normalized_text=" ".join(raw.split()),
            source=TranscriptSource.ASR,
            language=language,
            confidence=confidence,
            words=ordered_words,
            source_stream_index=request.audio.stream_index,
        )
