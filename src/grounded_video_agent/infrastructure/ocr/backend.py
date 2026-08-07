from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class OCRModelInfo:
    model_name: str
    model_version: str
    provider: str

    def __post_init__(self) -> None:
        for field_name in ("model_name", "model_version", "provider"):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True, slots=True)
class OCRFrameInput:
    frame_id: str
    uri: str

    def __post_init__(self) -> None:
        if not self.frame_id.strip() or not self.uri.strip():
            raise ValueError("OCR frame id and URI must not be empty")


@dataclass(frozen=True, slots=True)
class OCRDetection:
    text: str
    polygon: tuple[tuple[float, float], ...]
    confidence: float

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("OCR detection text must not be empty")
        if len(self.polygon) < 4:
            raise ValueError("OCR detection polygon requires at least four points")
        if not 0 <= self.confidence <= 1:
            raise ValueError("OCR detection confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class OCRFrameResult:
    frame_id: str
    width: int
    height: int
    detections: tuple[OCRDetection, ...]

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame_id must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("OCR image dimensions must be positive")


class OCRBackend(Protocol):
    def get_model_info(self) -> OCRModelInfo: ...

    def recognize(self, frames: tuple[OCRFrameInput, ...]) -> tuple[OCRFrameResult, ...]: ...

