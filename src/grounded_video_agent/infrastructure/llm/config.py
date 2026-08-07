"""Configuration for the DeepSeek language-model adapter."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class DeepSeekBackendConfig:
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    connect_timeout_seconds: float = 10.0
    completion_timeout_seconds: float = 120.0
    transient_retries: int = 2
    retry_backoff_seconds: float = 0.5
    default_max_output_tokens: int = 4_096
    default_temperature: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("model", "base_url", "api_key_env"):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        base_url = self.base_url.rstrip("/")
        parsed = urlparse(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an absolute HTTP(S) URL without credentials")
        if self.connect_timeout_seconds <= 0 or self.completion_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if self.transient_retries < 0:
            raise ValueError("transient_retries must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        if self.default_max_output_tokens <= 0:
            raise ValueError("default_max_output_tokens must be positive")
        if not 0 <= self.default_temperature <= 2:
            raise ValueError("default_temperature must be between zero and two")
        object.__setattr__(self, "base_url", base_url)
