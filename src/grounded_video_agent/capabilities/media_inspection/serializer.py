"""Stable JSON serialization for inspection snapshots."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from grounded_video_agent.capabilities.media_inspection.contracts import VideoInspectionResult


def inspection_result_to_dict(result: VideoInspectionResult) -> dict[str, Any]:
    payload = _json_value(result)
    if not isinstance(payload, dict):
        raise TypeError("serialized inspection result must be a dictionary")
    if result.video_context is not None:
        validation = payload["video_context"]["validation"]
        validation["status"] = result.video_context.validation.status.value
        validation["is_processable"] = result.video_context.validation.is_processable
    return payload


def inspection_result_to_json(
    result: VideoInspectionResult,
    *,
    indent: int | None = 2,
) -> str:
    return json.dumps(
        inspection_result_to_dict(result),
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set | frozenset):
        return [_json_value(item) for item in value]
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")
