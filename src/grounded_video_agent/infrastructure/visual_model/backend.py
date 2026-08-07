from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


class VisualModelBackendError(RuntimeError):
    """A normalized error raised by visual-model transport adapters."""


@dataclass(frozen=True, slots=True)
class VisualModelInfo:
    model_name: str
    model_version: str
    provider: str | None = None
    quantization: str | None = None
    context_length: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.model_name, "model_name")
        _require_text(self.model_version, "model_version")
        if self.provider is not None:
            _require_text(self.provider, "provider")
        if self.quantization is not None:
            _require_text(self.quantization, "quantization")
        if self.context_length is not None and self.context_length <= 0:
            raise ValueError("context_length must be positive")


@dataclass(frozen=True, slots=True)
class VisualModelFrame:
    frame_id: str
    uri: str
    timestamp_ms: int

    def __post_init__(self) -> None:
        _require_text(self.frame_id, "frame_id")
        _require_text(self.uri, "uri")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")


@dataclass(frozen=True, slots=True)
class VisualModelTarget:
    target_id: str
    start_ms: int
    end_ms: int
    frame_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.target_id, "target_id")
        if self.start_ms < 0 or self.start_ms >= self.end_ms:
            raise ValueError("target range must be non-negative and non-empty")
        _require_unique_texts(self.frame_ids, "frame_ids")
        if not self.frame_ids:
            raise ValueError("target requires at least one frame")


@dataclass(frozen=True, slots=True)
class VisualModelRequest:
    operation_id: str
    mode: str
    frames: tuple[VisualModelFrame, ...]
    targets: tuple[VisualModelTarget, ...]
    question: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.operation_id, "operation_id")
        _require_text(self.mode, "mode")
        _require_unique_texts((frame.frame_id for frame in self.frames), "frame_ids")
        _require_unique_texts((target.target_id for target in self.targets), "target_ids")
        if not self.targets:
            raise ValueError("visual model request requires targets")
        known_frames = {frame.frame_id for frame in self.frames}
        if any(not set(target.frame_ids).issubset(known_frames) for target in self.targets):
            raise ValueError("targets must reference submitted frames")
        if self.question is not None:
            _require_text(self.question, "question")


@dataclass(frozen=True, slots=True)
class VisualModelObservation:
    target_id: str
    text: str
    frame_ids: tuple[str, ...]
    tags: tuple[str, ...] = ()
    confidence: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.target_id, "target_id")
        _require_text(self.text, "text")
        _require_unique_texts(self.frame_ids, "frame_ids")
        _require_unique_texts(self.tags, "tags")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class VisualModelResponse:
    model: VisualModelInfo
    observations: tuple[VisualModelObservation, ...]
    warnings: tuple[str, ...] = ()
    model_calls: int = 1

    def __post_init__(self) -> None:
        _require_unique_texts(
            (observation.target_id for observation in self.observations),
            "observation target_ids",
        )
        if any(not warning.strip() for warning in self.warnings):
            raise ValueError("warnings must not contain empty values")
        if self.model_calls <= 0:
            raise ValueError("model_calls must be positive")


class VisualModelBackend(Protocol):
    def get_model_info(self) -> VisualModelInfo: ...

    def analyze(self, request: VisualModelRequest) -> VisualModelResponse: ...


def _require_text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_unique_texts(values: Iterable[object], field_name: str) -> None:
    items: tuple[object, ...] = tuple(values)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(set(items)) != len(items):
        raise ValueError(f"{field_name} must be unique")
