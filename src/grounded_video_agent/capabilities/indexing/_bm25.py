from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grounded_video_agent.capabilities._support import write_json
from grounded_video_agent.domain import TimeRange

_WORD = re.compile(r"[a-zA-Z0-9_]+|[\u3400-\u9fff]")


@dataclass(frozen=True, slots=True)
class TextIndexDocument:
    item_id: str
    text: str
    time_range: TimeRange
    source_ids: tuple[str, ...]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScoredTextDocument:
    document: TextIndexDocument
    score: float


def tokenize(text: str) -> tuple[str, ...]:
    units = tuple(item.lower() for item in _WORD.findall(text))
    cjk = tuple(item for item in units if len(item) == 1 and "\u3400" <= item <= "\u9fff")
    bigrams = tuple(first + second for first, second in zip(cjk, cjk[1:], strict=False))
    return (*units, *bigrams)


def unique_values(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def write_index(
    path: Path,
    *,
    video_id: str,
    modality: str,
    documents: tuple[TextIndexDocument, ...],
) -> None:
    payload = {
        "schema_version": "1",
        "video_id": video_id,
        "modality": modality,
        "documents": [
            {
                "item_id": document.item_id,
                "text": document.text,
                "start_ms": document.time_range.start_ms,
                "end_ms": document.time_range.end_ms,
                "source_ids": document.source_ids,
                "tags": document.tags,
                "token_counts": dict(Counter(tokenize(document.text))),
            }
            for document in documents
        ],
    }
    write_json(path, payload)


def search_index(
    path: str | Path,
    query: str,
    *,
    top_k: int,
    min_score: float,
    within: TimeRange | None = None,
    required_source_ids: frozenset[str] = frozenset(),
    required_tags: frozenset[str] = frozenset(),
    expected_video_id: str | None = None,
    expected_modality: str | None = None,
) -> tuple[ScoredTextDocument, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_documents = payload.get("documents")
    if payload.get("schema_version") != "1" or not isinstance(raw_documents, list):
        raise ValueError("unsupported or invalid text index artifact")
    if expected_video_id is not None and payload.get("video_id") != expected_video_id:
        raise ValueError("text index video identity does not match its manifest")
    if expected_modality is not None and payload.get("modality") != expected_modality:
        raise ValueError("text index modality does not match its manifest")
    query_terms = tokenize(query)
    if not query_terms:
        return ()
    prepared: list[tuple[TextIndexDocument, Counter[str]]] = []
    for raw in raw_documents:
        document, counts = _load_document(raw)
        if within is not None and not document.time_range.overlaps(within):
            continue
        if required_source_ids and required_source_ids.isdisjoint(document.source_ids):
            continue
        if required_tags and not required_tags.issubset(document.tags):
            continue
        prepared.append((document, counts))
    if not prepared:
        return ()
    lengths = [sum(counts.values()) for _, counts in prepared]
    average_length = sum(lengths) / len(lengths) or 1.0
    document_frequency = Counter(
        term for _, counts in prepared for term in set(query_terms) if term in counts
    )
    scored: list[ScoredTextDocument] = []
    for (document, counts), length in zip(prepared, lengths, strict=True):
        score = _bm25_score(
            query_terms,
            counts,
            length,
            average_length,
            len(prepared),
            document_frequency,
        )
        if score >= min_score and score > 0:
            scored.append(ScoredTextDocument(document, score))
    scored.sort(key=lambda item: (-item.score, item.document.time_range, item.document.item_id))
    return tuple(scored[:top_k])


def _load_document(raw: Any) -> tuple[TextIndexDocument, Counter[str]]:
    if not isinstance(raw, dict):
        raise ValueError("invalid text index document")
    try:
        document = TextIndexDocument(
            item_id=str(raw["item_id"]),
            text=str(raw["text"]),
            time_range=TimeRange(int(raw["start_ms"]), int(raw["end_ms"])),
            source_ids=tuple(str(item) for item in raw["source_ids"]),
            tags=tuple(str(item) for item in raw.get("tags", ())),
        )
        counts = Counter({str(key): int(value) for key, value in raw["token_counts"].items()})
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid text index document") from error
    return document, counts


def _bm25_score(
    query_terms: tuple[str, ...],
    counts: Counter[str],
    document_length: int,
    average_length: float,
    document_count: int,
    document_frequency: Counter[str],
) -> float:
    k1 = 1.5
    b = 0.75
    score = 0.0
    for term, query_frequency in Counter(query_terms).items():
        frequency = counts[term]
        if not frequency:
            continue
        frequency_in_documents = document_frequency[term]
        inverse_frequency = math.log(
            1 + (document_count - frequency_in_documents + 0.5) / (frequency_in_documents + 0.5)
        )
        denominator = frequency + k1 * (
            1 - b + b * document_length / average_length
        )
        score += query_frequency * inverse_frequency * frequency * (k1 + 1) / denominator
    return score
