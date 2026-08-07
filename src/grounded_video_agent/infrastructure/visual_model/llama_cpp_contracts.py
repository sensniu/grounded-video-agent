from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class LlamaCppBackendConfig:
    base_url: str
    allowed_roots: tuple[str | Path, ...]
    model_id: str | None = None
    connect_timeout_seconds: float = 5.0
    inference_timeout_seconds: float = 180.0
    max_frames_per_target: int = 4
    max_image_bytes: int = 8 * 1024 * 1024
    max_tokens: int = 512
    temperature: float = 0.1
    transient_retries: int = 1
    use_json_schema: bool = True
    api_key: str | None = None

    def __post_init__(self) -> None:
        base_url = self.base_url.rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        roots = tuple(Path(root).expanduser().resolve() for root in self.allowed_roots)
        if not roots:
            raise ValueError("allowed_roots must not be empty")
        if self.model_id is not None and not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.connect_timeout_seconds <= 0 or self.inference_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_frames_per_target <= 0:
            raise ValueError("max_frames_per_target must be positive")
        if self.max_image_bytes <= 0:
            raise ValueError("max_image_bytes must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between zero and two")
        if self.transient_retries < 0:
            raise ValueError("transient_retries must be non-negative")
        if self.api_key is not None and not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "allowed_roots", roots)
