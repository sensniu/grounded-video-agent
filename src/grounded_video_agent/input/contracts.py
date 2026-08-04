"""Contracts returned by lightweight local video registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from grounded_video_agent.domain import VideoAsset


class RegistrationStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RegistrationErrorCode(StrEnum):
    INVALID_FILENAME = "invalid_filename"
    INPUT_ROOT_NOT_FOUND = "input_root_not_found"
    FILE_NOT_FOUND = "file_not_found"
    NOT_A_FILE = "not_a_file"
    SYMLINK_NOT_ALLOWED = "symlink_not_allowed"
    PATH_NOT_ALLOWED = "path_not_allowed"
    PERMISSION_DENIED = "permission_denied"
    FILE_CHANGED = "file_changed_during_registration"
    IO_ERROR = "io_error"


@dataclass(frozen=True, slots=True)
class RegistrationError:
    code: RegistrationErrorCode
    message: str


@dataclass(frozen=True, slots=True)
class RegisteredFileInfo:
    """Filesystem facts recorded without decoding the media file."""

    filename: str
    relative_uri: str
    resolved_path: str
    size_bytes: int
    modified_at_ns: int


@dataclass(frozen=True, slots=True)
class VideoRegistrationResult:
    """Result of turning a trusted filename into a stable video asset."""

    registration_id: str
    status: RegistrationStatus
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    video_asset: VideoAsset | None = None
    file_info: RegisteredFileInfo | None = None
    error: RegistrationError | None = None

    def __post_init__(self) -> None:
        if self.registered_at.tzinfo is None or self.registered_at.utcoffset() is None:
            raise ValueError("registered_at must be timezone-aware")
        if self.status is RegistrationStatus.SUCCEEDED:
            if self.video_asset is None or self.file_info is None or self.error is not None:
                raise ValueError("successful registration requires asset and file info only")
        elif self.video_asset is not None or self.file_info is not None or self.error is None:
            raise ValueError("failed registration requires an error only")
