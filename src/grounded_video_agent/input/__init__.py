"""Registration of external inputs into trusted domain objects."""

from grounded_video_agent.input.contracts import (
    RegisteredFileInfo,
    RegistrationError,
    RegistrationErrorCode,
    RegistrationStatus,
    VideoRegistrationResult,
)
from grounded_video_agent.input.video_registration import VideoRegistrar

__all__ = [
    "RegisteredFileInfo",
    "RegistrationError",
    "RegistrationErrorCode",
    "RegistrationStatus",
    "VideoRegistrar",
    "VideoRegistrationResult",
]
