import pytest

from grounded_video_agent.domain import (
    ProducerInfo,
    RecoveryAction,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    ValidationStatus,
)

VALIDATOR = ProducerInfo(name="media-validator", version="1.0.0")


def test_empty_validation_report_is_valid() -> None:
    report = ValidationReport(video_id="video-1", validator=VALIDATOR)

    assert report.status is ValidationStatus.VALID
    assert report.is_processable


@pytest.mark.parametrize(
    ("issue", "expected_status"),
    [
        (
            ValidationIssue(
                code="VARIABLE_FRAME_RATE",
                severity=ValidationSeverity.WARNING,
                message="Input uses a variable frame rate.",
            ),
            ValidationStatus.VALID_WITH_WARNINGS,
        ),
        (
            ValidationIssue(
                code="ROTATION_METADATA_PRESENT",
                severity=ValidationSeverity.ERROR,
                message="Rotation must be normalized.",
                recovery_action=RecoveryAction.NORMALIZE,
            ),
            ValidationStatus.REQUIRES_NORMALIZATION,
        ),
        (
            ValidationIssue(
                code="UNSUPPORTED_SUBTITLE",
                severity=ValidationSeverity.ERROR,
                message="The subtitle stream is not supported.",
                recovery_action=RecoveryAction.SKIP_STREAM,
            ),
            ValidationStatus.PARTIALLY_SUPPORTED,
        ),
        (
            ValidationIssue(
                code="UNDECODABLE_VIDEO",
                severity=ValidationSeverity.FATAL,
                message="The primary video stream cannot be decoded.",
                recovery_action=RecoveryAction.REJECT,
            ),
            ValidationStatus.INVALID,
        ),
    ],
)
def test_validation_status_is_derived_from_issues(
    issue: ValidationIssue,
    expected_status: ValidationStatus,
) -> None:
    report = ValidationReport(video_id="video-1", validator=VALIDATOR, issues=(issue,))

    assert report.status is expected_status


def test_invalid_report_is_not_processable() -> None:
    issue = ValidationIssue(
        code="CORRUPT_FILE",
        severity=ValidationSeverity.FATAL,
        message="The source file is corrupt.",
    )
    report = ValidationReport(video_id="video-1", validator=VALIDATOR, issues=(issue,))

    assert not report.is_processable
    assert report.errors == (issue,)


def test_fatal_issue_cannot_recommend_normalization() -> None:
    with pytest.raises(ValueError, match="fatal issues"):
        ValidationIssue(
            code="CORRUPT_FILE",
            severity=ValidationSeverity.FATAL,
            message="The source file is corrupt.",
            recovery_action=RecoveryAction.NORMALIZE,
        )
