"""Independent basic-media validation rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grounded_video_agent.domain import (
    MediaProbe,
    RecoveryAction,
    ValidationIssue,
    ValidationSeverity,
)

if TYPE_CHECKING:
    from grounded_video_agent.capabilities.media_inspection.validator import (
        MediaValidationPolicy,
    )


def validate_media_probe(
    probe: MediaProbe,
    policy: MediaValidationPolicy,
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    playable_video_streams = tuple(
        stream for stream in probe.video_streams if not stream.is_attached_picture
    )
    primary_video = probe.primary_video_stream

    if not playable_video_streams:
        issues.append(
            ValidationIssue(
                code="NO_VIDEO_STREAM",
                severity=ValidationSeverity.FATAL,
                message="The file does not contain a playable video stream.",
                recovery_action=RecoveryAction.REJECT,
                scope="video",
            )
        )
        return tuple(issues)

    if primary_video is not None:
        if primary_video.codec_name is None:
            issues.append(
                ValidationIssue(
                    code="MISSING_VIDEO_CODEC",
                    severity=ValidationSeverity.FATAL,
                    message="The primary video stream has no recognized codec.",
                    recovery_action=RecoveryAction.REJECT,
                    scope="video_stream",
                    stream_index=primary_video.stream_index,
                )
            )
        elif (
            policy.supported_video_codecs is not None
            and primary_video.codec_name not in policy.supported_video_codecs
        ):
            issues.append(
                ValidationIssue(
                    code="UNSUPPORTED_VIDEO_CODEC",
                    severity=ValidationSeverity.FATAL,
                    message=f"Unsupported video codec: {primary_video.codec_name}.",
                    recovery_action=RecoveryAction.REJECT,
                    scope="video_stream",
                    stream_index=primary_video.stream_index,
                )
            )
        if primary_video.frame_rate is None:
            issues.append(
                ValidationIssue(
                    code="MISSING_FRAME_RATE",
                    severity=ValidationSeverity.WARNING,
                    message="The primary video stream has no usable nominal frame rate.",
                    scope="video_stream",
                    stream_index=primary_video.stream_index,
                )
            )
        if primary_video.is_variable_frame_rate:
            issues.append(
                ValidationIssue(
                    code="VARIABLE_FRAME_RATE",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        "Nominal and average frame rates differ; timestamp-based sampling "
                        "is required."
                    ),
                    scope="video_stream",
                    stream_index=primary_video.stream_index,
                )
            )
        if primary_video.rotation_degrees % 360 != 0:
            issues.append(
                ValidationIssue(
                    code="ROTATION_METADATA_PRESENT",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        "The primary video stream declares "
                        f"{primary_video.rotation_degrees} degrees "
                        "of rotation."
                    ),
                    scope="video_stream",
                    stream_index=primary_video.stream_index,
                )
            )
        if policy.max_width is not None and primary_video.width > policy.max_width:
            issues.append(
                ValidationIssue(
                    code="VIDEO_WIDTH_EXCEEDED",
                    severity=ValidationSeverity.ERROR,
                    message=f"Video width {primary_video.width} exceeds {policy.max_width}.",
                    recovery_action=RecoveryAction.NORMALIZE,
                    scope="video_stream",
                    stream_index=primary_video.stream_index,
                )
            )
        if policy.max_height is not None and primary_video.height > policy.max_height:
            issues.append(
                ValidationIssue(
                    code="VIDEO_HEIGHT_EXCEEDED",
                    severity=ValidationSeverity.ERROR,
                    message=f"Video height {primary_video.height} exceeds {policy.max_height}.",
                    recovery_action=RecoveryAction.NORMALIZE,
                    scope="video_stream",
                    stream_index=primary_video.stream_index,
                )
            )

    duration_ms = probe.container.duration_ms
    if duration_ms is None and primary_video is not None:
        duration_ms = primary_video.duration_ms
    if duration_ms is None or duration_ms <= 0:
        issues.append(
            ValidationIssue(
                code="MISSING_DURATION",
                severity=ValidationSeverity.FATAL,
                message="No positive media duration could be determined.",
                recovery_action=RecoveryAction.REJECT,
                scope="timeline",
            )
        )
    elif policy.max_duration_ms is not None and duration_ms > policy.max_duration_ms:
        issues.append(
            ValidationIssue(
                code="VIDEO_DURATION_EXCEEDED",
                severity=ValidationSeverity.FATAL,
                message=f"Video duration {duration_ms}ms exceeds {policy.max_duration_ms}ms.",
                recovery_action=RecoveryAction.REJECT,
                scope="timeline",
            )
        )

    if not probe.audio_streams:
        if policy.require_audio:
            issues.append(
                ValidationIssue(
                    code="NO_AUDIO_STREAM",
                    severity=ValidationSeverity.FATAL,
                    message="The file has no audio stream, but audio is required.",
                    recovery_action=RecoveryAction.REJECT,
                    scope="audio",
                )
            )
        elif policy.warn_when_audio_missing:
            issues.append(
                ValidationIssue(
                    code="NO_AUDIO_STREAM",
                    severity=ValidationSeverity.WARNING,
                    message="The file has no audio stream; audio and ASR stages must be skipped.",
                    scope="audio",
                )
            )

    if not probe.container.format_names:
        issues.append(
            ValidationIssue(
                code="UNKNOWN_CONTAINER_FORMAT",
                severity=ValidationSeverity.WARNING,
                message="FFprobe did not report a container format.",
                scope="container",
            )
        )
    return tuple(issues)
