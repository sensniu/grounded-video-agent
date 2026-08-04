"""Public input-independent results produced by media inspection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from grounded_video_agent.domain import MediaProbe, ValidationReport, VideoAsset
from grounded_video_agent.input import VideoRegistrationResult


class InspectionExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class InspectionStage(StrEnum):
    REGISTRATION = "registration"
    INPUT_CHECK = "input_check"
    PROBING = "probing"
    MAPPING = "mapping"
    VALIDATING = "validating"
    COMPLETED = "completed"


class InspectionErrorCode(StrEnum):
    INVALID_FILENAME = "invalid_filename"
    INPUT_ROOT_NOT_FOUND = "input_root_not_found"
    SOURCE_NOT_FOUND = "source_not_found"
    SOURCE_NOT_READABLE = "source_not_readable"
    SOURCE_NOT_A_FILE = "source_not_a_file"
    SYMLINK_NOT_ALLOWED = "symlink_not_allowed"
    PATH_NOT_ALLOWED = "path_not_allowed"
    SOURCE_CHANGED = "source_changed"
    FFPROBE_NOT_FOUND = "ffprobe_not_found"
    FFPROBE_TIMEOUT = "ffprobe_timeout"
    FFPROBE_FAILED = "ffprobe_failed"
    EMPTY_OUTPUT = "empty_ffprobe_output"
    INVALID_JSON = "invalid_ffprobe_json"
    MAPPING_FAILED = "mapping_failed"
    INTERNAL_ERROR = "internal_error"


class NextAction(StrEnum):
    PROCEED = "proceed"
    PROCEED_WITH_LIMITATIONS = "proceed_with_limitations"
    NORMALIZE_AND_REINSPECT = "normalize_and_reinspect"
    RETRY_INSPECTION = "retry_inspection"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class InspectionError:
    code: InspectionErrorCode
    message: str
    stage: InspectionStage
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class InspectionExecution:
    status: InspectionExecutionStatus
    stage: InspectionStage
    started_at: datetime
    finished_at: datetime
    duration_ms: int


@dataclass(frozen=True, slots=True)
class PrimaryStreams:
    video_stream_index: int | None
    audio_stream_index: int | None
    subtitle_stream_index: int | None


@dataclass(frozen=True, slots=True)
class BasicVideoFlags:
    has_video: bool
    has_audio: bool
    has_embedded_subtitles: bool
    has_multiple_video_streams: bool
    has_multiple_audio_streams: bool
    is_variable_frame_rate: bool
    has_rotation_metadata: bool


@dataclass(frozen=True, slots=True)
class VideoInspectionContext:
    """Complete basic media snapshot reused by downstream processing."""

    video_asset: VideoAsset
    media_probe: MediaProbe
    validation: ValidationReport
    primary_streams: PrimaryStreams
    basic_flags: BasicVideoFlags


@dataclass(frozen=True, slots=True)
class InspectionDiagnostics:
    ffprobe_stderr: str = ""
    raw_probe: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class VideoInspectionResult:
    schema_version: str
    inspection_id: str
    registration: VideoRegistrationResult
    execution: InspectionExecution
    next_action: NextAction
    video_context: VideoInspectionContext | None
    error: InspectionError | None
    diagnostics: InspectionDiagnostics

    def __post_init__(self) -> None:
        if self.execution.status is InspectionExecutionStatus.SUCCEEDED:
            if self.video_context is None or self.error is not None:
                raise ValueError("successful inspection requires video context only")
        elif self.video_context is not None or self.error is None:
            raise ValueError("failed inspection requires an error only")

    def to_dict(self) -> dict[str, Any]:
        from grounded_video_agent.capabilities.media_inspection.serializer import (
            inspection_result_to_dict,
        )

        return inspection_result_to_dict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        from grounded_video_agent.capabilities.media_inspection.serializer import (
            inspection_result_to_json,
        )

        return inspection_result_to_json(self, indent=indent)
