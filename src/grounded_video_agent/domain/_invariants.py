"""Small validation helpers shared by immutable domain records."""

from __future__ import annotations

import math
from collections.abc import Iterable

from grounded_video_agent.domain.artifacts import ManifestKind, ManifestRef


def require_text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def require_non_negative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def require_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def require_optional_positive_int(value: int | None, field_name: str) -> None:
    if value is not None:
        require_positive_int(value, field_name)


def require_probability(value: float | None, field_name: str = "confidence") -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a number between 0 and 1")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")


def require_finite_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")


def require_unique_texts(values: Iterable[str], field_name: str) -> None:
    collected = tuple(values)
    if any(not value or not value.strip() for value in collected):
        raise ValueError(f"{field_name} must not contain empty values")
    if len(set(collected)) != len(collected):
        raise ValueError(f"{field_name} must contain unique values")


def require_manifest(
    ref: ManifestRef,
    *,
    kind: ManifestKind,
    video_id: str,
    item_count: int,
) -> None:
    if ref.kind is not kind:
        raise ValueError(f"manifest must have kind {kind.value}")
    if ref.source_video_id != video_id:
        raise ValueError("manifest source_video_id must match video_id")
    if ref.item_count != item_count:
        raise ValueError("manifest item_count must match the contained records")
