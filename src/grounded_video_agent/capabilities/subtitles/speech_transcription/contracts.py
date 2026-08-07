from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.domain import AudioArtifact, CapabilityRequestContext


@dataclass(frozen=True, slots=True)
class SpeechTranscriptionRequest:
    audio: AudioArtifact
    context: CapabilityRequestContext
    language_hint: str | None = None
    word_timestamps: bool = True

    def __post_init__(self) -> None:
        if self.language_hint is not None and not self.language_hint.strip():
            raise ValueError("language_hint must not be empty")
