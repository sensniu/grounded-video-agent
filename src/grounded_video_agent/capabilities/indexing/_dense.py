from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from grounded_video_agent.capabilities._support import write_json
from grounded_video_agent.capabilities.indexing._bm25 import TextIndexDocument
from grounded_video_agent.domain import TimeRange
from grounded_video_agent.infrastructure.embeddings import normalize_vector


@dataclass(frozen=True, slots=True)
class ScoredDenseDocument:
    document: TextIndexDocument
    score: float


def write_dense_index(
    path: Path,
    embedding_path: Path,
    *,
    video_id: str,
    modality: str,
    embedding_space: str,
    dimensions: int,
    documents: tuple[TextIndexDocument, ...],
    vectors: tuple[tuple[float, ...], ...],
) -> None:
    if len(documents) != len(vectors):
        raise ValueError("dense documents and vectors must have equal lengths")
    normalized = tuple(normalize_vector(vector, dimensions) for vector in vectors)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(normalized, dtype=np.float32).reshape(len(normalized), dimensions)
    np.save(embedding_path, matrix, allow_pickle=False)
    write_json(
        path,
        {
            "schema_version": "1",
            "video_id": video_id,
            "modality": modality,
            "embedding_space": embedding_space,
            "dimensions": dimensions,
            "embedding_uri": str(embedding_path),
            "documents": [
                {
                    "item_id": document.item_id,
                    "text": document.text,
                    "start_ms": document.time_range.start_ms,
                    "end_ms": document.time_range.end_ms,
                    "source_ids": document.source_ids,
                    "tags": document.tags,
                }
                for document in documents
            ],
        },
    )


def search_dense_index(
    path: str | Path,
    query_vector: tuple[float, ...],
    *,
    embedding_space: str,
    dimensions: int,
    top_k: int,
    min_score: float,
    within: TimeRange | None = None,
    required_source_ids: frozenset[str] = frozenset(),
    required_tags: frozenset[str] = frozenset(),
    expected_video_id: str | None = None,
    expected_modality: str | None = None,
) -> tuple[ScoredDenseDocument, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_documents = payload.get("documents")
    if payload.get("schema_version") != "1" or not isinstance(raw_documents, list):
        raise ValueError("unsupported or invalid dense index artifact")
    if expected_video_id is not None and payload.get("video_id") != expected_video_id:
        raise ValueError("dense index video identity does not match its manifest")
    if expected_modality is not None and payload.get("modality") != expected_modality:
        raise ValueError("dense index modality does not match its manifest")
    if payload.get("embedding_space") != embedding_space:
        raise ValueError("query and index use different embedding spaces")
    if payload.get("dimensions") != dimensions:
        raise ValueError("query and index embedding dimensions do not match")
    embedding_uri = payload.get("embedding_uri")
    if not isinstance(embedding_uri, str) or not embedding_uri.strip():
        raise ValueError("dense index does not reference an embedding artifact")
    matrix = np.load(embedding_uri, allow_pickle=False)
    if matrix.shape != (len(raw_documents), dimensions):
        raise ValueError("dense index matrix shape does not match its metadata")
    normalized_query = np.asarray(
        normalize_vector(query_vector, dimensions),
        dtype=np.float32,
    )
    scores = matrix @ normalized_query
    scored: list[ScoredDenseDocument] = []
    for raw, score in zip(raw_documents, scores, strict=True):
        document = _load_document(raw)
        if within is not None and not document.time_range.overlaps(within):
            continue
        if required_source_ids and required_source_ids.isdisjoint(document.source_ids):
            continue
        if required_tags and not required_tags.issubset(document.tags):
            continue
        numeric_score = float(score)
        if numeric_score >= min_score:
            scored.append(ScoredDenseDocument(document, numeric_score))
    scored.sort(key=lambda item: (-item.score, item.document.time_range, item.document.item_id))
    return tuple(scored[:top_k])


def _load_document(raw: Any) -> TextIndexDocument:
    if not isinstance(raw, dict):
        raise ValueError("invalid dense index document")
    try:
        return TextIndexDocument(
            item_id=str(raw["item_id"]),
            text=str(raw["text"]),
            time_range=TimeRange(int(raw["start_ms"]), int(raw["end_ms"])),
            source_ids=tuple(str(item) for item in raw["source_ids"]),
            tags=tuple(str(item) for item in raw.get("tags", ())),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid dense index document") from error
