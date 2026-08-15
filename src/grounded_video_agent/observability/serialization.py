"""Lossless-enough JSON conversion with recursive credential redaction."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "client_secret",
    "cookie",
    "password",
    "proxy_authorization",
    "refresh_token",
    "secret",
    "set_cookie",
}


def trace_json_value(value: Any, *, key: str | None = None) -> Any:
    """Convert trace payloads to JSON values without embedding binary media."""

    if key is not None and _is_sensitive_key(key):
        return "[REDACTED]"
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return trace_json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if isinstance(value, bytes | bytearray | memoryview):
        return {"type": "binary", "size_bytes": len(value)}
    if isinstance(value, BaseModel):
        return trace_json_value(value.model_dump(mode="python"))
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: trace_json_value(getattr(value, field.name), key=field.name)
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(item_key): trace_json_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, tuple | list | set | frozenset):
        return [trace_json_value(item) for item in value]
    return {"type": type(value).__name__, "repr": repr(value)}


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(
        ("_api_key", "_password", "_secret", "_access_token", "_refresh_token")
    )

