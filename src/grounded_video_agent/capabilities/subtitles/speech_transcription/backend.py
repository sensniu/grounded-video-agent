from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ASRWord:
    text: str
    start_seconds: float
    end_seconds: float
    probability: float | None = None


@dataclass(frozen=True, slots=True)
class ASRSegment:
    text: str
    start_seconds: float
    end_seconds: float
    words: tuple[ASRWord, ...] = ()


@dataclass(frozen=True, slots=True)
class ASRTranscript:
    language: str | None
    segments: tuple[ASRSegment, ...]


class TranscriptionBackend(Protocol):
    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None,
        word_timestamps: bool,
    ) -> ASRTranscript: ...


class FasterWhisperBackend:
    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "auto",
        compute_type: str = "default",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: Any = None

    def cache_identity(self) -> dict[str, str]:
        return {
            "backend": "faster-whisper",
            "model_size": self._model_size,
            "device": self._device,
            "compute_type": self._compute_type,
        }

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None,
        word_timestamps: bool,
    ) -> ASRTranscript:
        if self._model is None:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        segments, info = self._model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=True,
        )
        converted: list[ASRSegment] = []
        for segment in segments:
            words = tuple(
                ASRWord(word.word, word.start, word.end, getattr(word, "probability", None))
                for word in (segment.words or ())
            )
            converted.append(ASRSegment(segment.text, segment.start, segment.end, words))
        return ASRTranscript(getattr(info, "language", language), tuple(converted))
