from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from grounded_video_agent.domain import ResourceLimits


class SubtitlePolicy(StrEnum):
    AUTO = "auto"
    EMBEDDED_ONLY = "embedded_only"
    ASR_ONLY = "asr_only"


class DenseIndexPolicy(StrEnum):
    DISABLED = "disabled"
    OPTIONAL = "optional"
    REQUIRED = "required"


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_duration_ms: int = 15_000
    max_duration_ms: int = 30_000
    target_characters: int = 240
    max_characters: int = 480
    max_silence_gap_ms: int = 2_500
    context_padding_ms: int = 500
    max_inspection_duration_ms: int = 60_000
    align_to_shots: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.target_duration_ms <= self.max_duration_ms:
            raise ValueError("target duration must be positive and at most max duration")
        if not 0 < self.target_characters <= self.max_characters:
            raise ValueError("target characters must be positive and at most max characters")
        if self.max_silence_gap_ms < 0 or self.context_padding_ms < 0:
            raise ValueError("chunk gap and padding must be non-negative")
        if self.max_inspection_duration_ms < self.max_duration_ms:
            raise ValueError("max inspection duration must cover max chunk duration")


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    subtitle_policy: SubtitlePolicy = SubtitlePolicy.AUTO
    dense_index_policy: DenseIndexPolicy = DenseIndexPolicy.DISABLED
    language_hint: str | None = None
    audio_sample_rate_hz: int = 16_000
    audio_channels: int = 1
    shot_threshold: float = 27.0
    min_shot_duration_ms: int = 500
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    limits: ResourceLimits = field(default_factory=ResourceLimits)

    def __post_init__(self) -> None:
        if self.language_hint is not None and not self.language_hint.strip():
            raise ValueError("language_hint must not be empty")
        if self.audio_sample_rate_hz <= 0 or self.audio_channels <= 0:
            raise ValueError("audio output settings must be positive")
        if self.shot_threshold <= 0 or self.min_shot_duration_ms <= 0:
            raise ValueError("shot detection settings must be positive")
