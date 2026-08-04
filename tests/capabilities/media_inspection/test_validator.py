from grounded_video_agent.capabilities.media_inspection.mapper import map_ffprobe_payload
from grounded_video_agent.capabilities.media_inspection.validator import (
    MediaValidationPolicy,
    MediaValidator,
)
from grounded_video_agent.domain import ValidationStatus

from .sample_payload import sample_ffprobe_payload


def test_accepts_basic_video_with_audio() -> None:
    probe = map_ffprobe_payload("video-1", sample_ffprobe_payload())

    report = MediaValidator().validate(probe)

    assert report.status is ValidationStatus.VALID


def test_video_without_audio_is_processable_with_warning_by_default() -> None:
    probe = map_ffprobe_payload("video-1", sample_ffprobe_payload(include_audio=False))

    report = MediaValidator().validate(probe)

    assert report.status is ValidationStatus.VALID_WITH_WARNINGS
    assert report.issues[0].code == "NO_AUDIO_STREAM"


def test_required_audio_rejects_silent_video() -> None:
    probe = map_ffprobe_payload("video-1", sample_ffprobe_payload(include_audio=False))
    validator = MediaValidator(MediaValidationPolicy(require_audio=True))

    report = validator.validate(probe)

    assert report.status is ValidationStatus.INVALID


def test_resolution_limit_requests_normalization() -> None:
    probe = map_ffprobe_payload("video-1", sample_ffprobe_payload())
    validator = MediaValidator(MediaValidationPolicy(max_width=640))

    report = validator.validate(probe)

    assert report.status is ValidationStatus.REQUIRES_NORMALIZATION
