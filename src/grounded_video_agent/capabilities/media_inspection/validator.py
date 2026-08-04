"""Policy-driven validation of normalized media probe facts."""

from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.capabilities.media_inspection.rules import validate_media_probe
from grounded_video_agent.domain import MediaProbe, ProducerInfo, ValidationReport


@dataclass(frozen=True, slots=True)
class MediaValidationPolicy:
    """Project-level limits applied after probing; no media is modified here."""

    require_audio: bool = False
    warn_when_audio_missing: bool = True
    max_duration_ms: int | None = None
    max_width: int | None = None
    max_height: int | None = None
    supported_video_codecs: frozenset[str] | None = None

    def __post_init__(self) -> None:
        for field_name in ("max_duration_ms", "max_width", "max_height"):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive when provided")
        if self.supported_video_codecs is not None and any(
            not codec.strip() for codec in self.supported_video_codecs
        ):
            raise ValueError("supported_video_codecs must not contain empty values")


class MediaValidator:
    def __init__(
        self,
        policy: MediaValidationPolicy | None = None,
        *,
        version: str = "1.0.0",
    ) -> None:
        self._policy = policy or MediaValidationPolicy()
        self._producer = ProducerInfo(name="media-inspection-validator", version=version)

    @property
    def policy(self) -> MediaValidationPolicy:
        return self._policy

    def validate(self, probe: MediaProbe) -> ValidationReport:
        return ValidationReport(
            video_id=probe.video_id,
            validator=self._producer,
            issues=validate_media_probe(probe, self._policy),
        )
