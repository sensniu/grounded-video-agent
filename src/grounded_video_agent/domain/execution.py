"""Execution contracts shared by deterministic capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeVar

from grounded_video_agent.domain._invariants import (
    require_non_negative_int,
    require_optional_positive_int,
    require_text,
)
from grounded_video_agent.domain.artifacts import ArtifactRef, Provenance

T = TypeVar("T")


class CapabilityStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CapabilityError:
    code: str
    message: str
    stage: str
    retryable: bool = False
    suggested_action: str | None = None

    def __post_init__(self) -> None:
        require_text(self.code, "code")
        require_text(self.message, "message")
        require_text(self.stage, "stage")
        if self.suggested_action is not None:
            require_text(self.suggested_action, "suggested_action")


@dataclass(frozen=True, slots=True)
class CapabilityUsage:
    wall_time_ms: int = 0
    input_items: int = 0
    output_items: int = 0
    processed_duration_ms: int = 0
    decoded_frames: int = 0
    returned_frames: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            require_non_negative_int(getattr(self, field_name), field_name)
        if self.returned_frames > self.decoded_frames:
            raise ValueError("returned_frames cannot exceed decoded_frames")


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    max_wall_time_ms: int | None = None
    max_media_duration_ms: int | None = None
    max_frames: int | None = None
    max_items: int | None = None
    max_model_calls: int | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            require_optional_positive_int(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class CapabilityRequestContext:
    operation_id: str
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    trace_id: str | None = None
    force_refresh: bool = False

    def __post_init__(self) -> None:
        require_text(self.operation_id, "operation_id")
        if self.trace_id is not None:
            require_text(self.trace_id, "trace_id")


@dataclass(frozen=True, slots=True)
class CapabilityResult(Generic[T]):
    status: CapabilityStatus
    data: T | None
    usage: CapabilityUsage = field(default_factory=CapabilityUsage)
    artifacts: tuple[ArtifactRef, ...] = ()
    warnings: tuple[str, ...] = ()
    error: CapabilityError | None = None
    provenance: Provenance | None = None
    cache_hit: bool = False

    def __post_init__(self) -> None:
        if self.status is CapabilityStatus.SUCCESS:
            if self.data is None or self.error is not None:
                raise ValueError("successful capability result requires data and no error")
        elif self.status is CapabilityStatus.PARTIAL:
            if self.data is None:
                raise ValueError("partial capability result requires data")
        elif self.data is not None or self.error is None:
            raise ValueError("failed capability result requires an error and no data")
        if any(not warning.strip() for warning in self.warnings):
            raise ValueError("warnings must not contain empty values")
        artifact_ids = tuple(artifact.artifact_id for artifact in self.artifacts)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("artifacts must be unique")
