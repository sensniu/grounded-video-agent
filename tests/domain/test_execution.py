import pytest

from grounded_video_agent.domain import (
    CapabilityError,
    CapabilityResult,
    CapabilityStatus,
    CapabilityUsage,
    ResourceLimits,
)


def test_successful_capability_result_requires_data_without_error() -> None:
    result = CapabilityResult(status=CapabilityStatus.SUCCESS, data="manifest")

    assert result.data == "manifest"


def test_failed_capability_result_requires_error_without_data() -> None:
    error = CapabilityError(code="TIMEOUT", message="Timed out", stage="processing")
    result: CapabilityResult[str] = CapabilityResult(
        status=CapabilityStatus.FAILED,
        data=None,
        error=error,
    )

    assert result.error == error


def test_rejects_contradictory_capability_result() -> None:
    with pytest.raises(ValueError, match="requires data"):
        CapabilityResult(status=CapabilityStatus.SUCCESS, data=None)


def test_usage_rejects_more_returned_than_decoded_frames() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        CapabilityUsage(decoded_frames=1, returned_frames=2)


def test_resource_limits_are_positive_hard_caps() -> None:
    assert ResourceLimits(max_frames=16).max_frames == 16

    with pytest.raises(ValueError, match="positive"):
        ResourceLimits(max_frames=0)
