"""Safe subprocess wrapper around the local FFprobe executable."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol


class FFprobeErrorCode(StrEnum):
    EXECUTABLE_NOT_FOUND = "executable_not_found"
    TIMEOUT = "timeout"
    PROCESS_FAILED = "process_failed"
    EMPTY_OUTPUT = "empty_output"
    INVALID_JSON = "invalid_json"


class FFprobeError(RuntimeError):
    def __init__(
        self,
        code: FFprobeErrorCode,
        message: str,
        *,
        stderr: str = "",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stderr = stderr
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class RawProbeResult:
    payload: dict[str, Any]
    stderr: str
    duration_ms: int


class ProbeRunner(Protocol):
    def probe(self, source_path: str | Path) -> RawProbeResult: ...


class FFprobeRunner:
    """Run FFprobe without a shell and parse its JSON response."""

    def __init__(
        self,
        *,
        executable: str = "ffprobe",
        timeout_seconds: float = 30.0,
        stderr_limit: int = 4_000,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if stderr_limit < 0:
            raise ValueError("stderr_limit must be non-negative")
        self._executable = executable
        self._timeout_seconds = timeout_seconds
        self._stderr_limit = stderr_limit

    def probe(self, source_path: str | Path) -> RawProbeResult:
        command = [
            self._executable,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-print_format",
            "json",
            str(source_path),
        ]
        started = perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
                shell=False,
            )
        except FileNotFoundError as error:
            raise FFprobeError(
                FFprobeErrorCode.EXECUTABLE_NOT_FOUND,
                f"FFprobe executable was not found: {self._executable}",
            ) from error
        except subprocess.TimeoutExpired as error:
            stderr = self._truncate(error.stderr or "")
            raise FFprobeError(
                FFprobeErrorCode.TIMEOUT,
                f"FFprobe exceeded the {self._timeout_seconds:g}s timeout.",
                stderr=stderr,
                retryable=True,
            ) from error
        except OSError as error:
            raise FFprobeError(
                FFprobeErrorCode.PROCESS_FAILED,
                f"FFprobe could not be executed: {error}",
            ) from error

        duration_ms = round((perf_counter() - started) * 1000)
        stderr = self._truncate(completed.stderr)
        if completed.returncode != 0:
            raise FFprobeError(
                FFprobeErrorCode.PROCESS_FAILED,
                f"FFprobe exited with status {completed.returncode}.",
                stderr=stderr,
            )
        if not completed.stdout.strip():
            raise FFprobeError(
                FFprobeErrorCode.EMPTY_OUTPUT,
                "FFprobe returned an empty response.",
                stderr=stderr,
            )

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise FFprobeError(
                FFprobeErrorCode.INVALID_JSON,
                "FFprobe response is not valid JSON.",
                stderr=stderr,
            ) from error
        if not isinstance(payload, dict):
            raise FFprobeError(
                FFprobeErrorCode.INVALID_JSON,
                "FFprobe JSON root must be an object.",
                stderr=stderr,
            )
        return RawProbeResult(payload=payload, stderr=stderr, duration_ms=duration_ms)

    def _truncate(self, value: str | bytes) -> str:
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        if self._stderr_limit == 0:
            return ""
        return value[-self._stderr_limit :]
