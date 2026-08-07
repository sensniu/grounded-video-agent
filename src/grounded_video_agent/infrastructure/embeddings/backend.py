from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class EmbeddingModelInfo:
    model_name: str
    model_version: str
    embedding_space: str
    dimensions: int

    def __post_init__(self) -> None:
        for field_name in ("model_name", "model_version", "embedding_space"):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")


class TextEmbeddingBackend(Protocol):
    def get_model_info(self) -> EmbeddingModelInfo: ...

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


def normalize_vector(vector: tuple[float, ...], dimensions: int) -> tuple[float, ...]:
    if len(vector) != dimensions:
        raise ValueError("embedding vector dimensions do not match model information")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("embedding vectors must contain finite values")
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        raise ValueError("embedding vectors must not be zero")
    return tuple(value / magnitude for value in vector)

