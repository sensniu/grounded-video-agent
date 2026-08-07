from __future__ import annotations

from importlib.metadata import version
from typing import Any

from grounded_video_agent.infrastructure.embeddings.backend import EmbeddingModelInfo


class SentenceTransformerBackend:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        if not model_name.strip() or not device.strip():
            raise ValueError("model_name and device must not be empty")
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device=device)
        dimensions = model.get_sentence_embedding_dimension()
        if dimensions is None or int(dimensions) <= 0:
            raise ValueError("sentence-transformer did not report embedding dimensions")
        self._model = model
        self._info = EmbeddingModelInfo(
            model_name=model_name,
            model_version=version("sentence-transformers"),
            embedding_space=f"sentence-transformers:{model_name}",
            dimensions=int(dimensions),
        )

    def get_model_info(self) -> EmbeddingModelInfo:
        return self._info

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        return self._encode(texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        if not text.strip():
            raise ValueError("embedding query must not be empty")
        return self._encode((text,))[0]

    def _encode(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        encoded = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return tuple(tuple(float(value) for value in row) for row in encoded)

