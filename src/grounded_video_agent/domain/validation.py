"""Structured media validation outcomes and recovery guidance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from grounded_video_agent.domain.artifacts import ProducerInfo


def _require_text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


class ValidationSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class RecoveryAction(StrEnum):
    """The policy-level response recommended for a validation issue."""

    NONE = "none"
    NORMALIZE = "normalize"
    SKIP_STREAM = "skip_stream"
    REJECT = "reject"


class ValidationStatus(StrEnum):
    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    REQUIRES_NORMALIZATION = "requires_normalization"
    PARTIALLY_SUPPORTED = "partially_supported"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One stable, machine-actionable media validation finding."""

    code: str
    severity: ValidationSeverity
    message: str
    recovery_action: RecoveryAction = RecoveryAction.NONE
    scope: str = "media"
    stream_index: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.code, "code")
        _require_text(self.message, "message")
        _require_text(self.scope, "scope")
        if self.stream_index is not None and self.stream_index < 0:
            raise ValueError("stream_index must be non-negative")
        if self.severity is ValidationSeverity.FATAL and self.recovery_action not in {
            RecoveryAction.NONE,
            RecoveryAction.REJECT,
        }:
            raise ValueError("fatal issues cannot be normalized or skipped")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """A validator's findings; status is derived to prevent contradictory state."""

    video_id: str
    validator: ProducerInfo
    issues: tuple[ValidationIssue, ...] = ()
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        _require_text(self.video_id, "video_id")
        if self.checked_at.tzinfo is None or self.checked_at.utcoffset() is None:
            raise ValueError("checked_at must be timezone-aware")

    @property
    def status(self) -> ValidationStatus:
        if any(
            issue.severity is ValidationSeverity.FATAL
            or issue.recovery_action is RecoveryAction.REJECT
            for issue in self.issues
        ):
            return ValidationStatus.INVALID
        if any(issue.recovery_action is RecoveryAction.SKIP_STREAM for issue in self.issues):
            return ValidationStatus.PARTIALLY_SUPPORTED
        if any(issue.recovery_action is RecoveryAction.NORMALIZE for issue in self.issues):
            return ValidationStatus.REQUIRES_NORMALIZATION
        if any(issue.severity is ValidationSeverity.WARNING for issue in self.issues):
            return ValidationStatus.VALID_WITH_WARNINGS
        return ValidationStatus.VALID

    @property
    def is_processable(self) -> bool:
        return self.status is not ValidationStatus.INVALID

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue
            for issue in self.issues
            if issue.severity in {ValidationSeverity.ERROR, ValidationSeverity.FATAL}
        )
