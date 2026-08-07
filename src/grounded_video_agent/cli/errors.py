from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    RUNTIME_ERROR = 1
    INVALID_CONFIGURATION = 3
    INTERRUPTED = 130


class CLIError(RuntimeError):
    def __init__(self, message: str, *, exit_code: ExitCode) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class CLIConfigurationError(CLIError):
    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=ExitCode.INVALID_CONFIGURATION)
