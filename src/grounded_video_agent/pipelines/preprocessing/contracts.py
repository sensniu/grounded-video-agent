from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast

from grounded_video_agent.capabilities._support import json_value
from grounded_video_agent.domain import CapabilityUsage


class PipelineStatus(StrEnum):
    READY = "ready"
    PARTIAL = "partial"
    FAILED = "failed"


class PipelineStage(StrEnum):
    REGISTRATION = "registration"
    MEDIA_INSPECTION = "media_inspection"
    SHOT_DETECTION = "shot_detection"
    EMBEDDED_SUBTITLES = "embedded_subtitles"
    AUDIO_EXTRACTION = "audio_extraction"
    SPEECH_TRANSCRIPTION = "speech_transcription"
    CHUNKING = "chunking"
    SPARSE_INDEXING = "sparse_indexing"
    DENSE_INDEXING = "dense_indexing"
    CATALOG_AUDIT = "catalog_audit"


class PipelineStageStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"
    CACHE_HIT = "cache_hit"


@dataclass(frozen=True, slots=True)
class PipelineError:
    code: str
    message: str
    stage: PipelineStage
    retryable: bool = False

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.message.strip():
            raise ValueError("pipeline error code and message must not be empty")


@dataclass(frozen=True, slots=True)
class PipelineStageReport:
    stage: PipelineStage
    status: PipelineStageStatus
    entry_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: PipelineError | None = None
    usage: CapabilityUsage = field(default_factory=CapabilityUsage)

    def __post_init__(self) -> None:
        if self.status is PipelineStageStatus.FAILED and self.error is None:
            raise ValueError("failed pipeline stages require an error")
        if self.status is not PipelineStageStatus.FAILED and self.error is not None:
            raise ValueError("only failed pipeline stages may contain an error")
        if len(set(self.entry_ids)) != len(self.entry_ids):
            raise ValueError("pipeline stage entry ids must be unique")


@dataclass(frozen=True, slots=True)
class PipelineReadiness:
    media_ready: bool = False
    shots_ready: bool = False
    transcript_ready: bool = False
    timeline_ready: bool = False
    sparse_search_ready: bool = False
    dense_search_ready: bool = False
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PipelineCatalogEntries:
    media_inspection_entry_id: str | None = None
    shots_entry_id: str | None = None
    transcript_entry_id: str | None = None
    audio_entry_id: str | None = None
    chunks_entry_id: str | None = None
    sparse_index_entry_id: str | None = None
    embedding_entry_id: str | None = None
    dense_index_entry_id: str | None = None


@dataclass(frozen=True, slots=True)
class PreprocessingRequest:
    filename: str
    force_refresh: bool = False
    trace_id: str | None = None

    def __post_init__(self) -> None:
        if not self.filename.strip():
            raise ValueError("filename must not be empty")
        if self.trace_id is not None and not self.trace_id.strip():
            raise ValueError("trace_id must not be empty")


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    run_id: str
    status: PipelineStatus
    video_id: str | None
    catalog_revision: int | None
    readiness: PipelineReadiness
    stages: tuple[PipelineStageReport, ...]
    entries: PipelineCatalogEntries = field(default_factory=PipelineCatalogEntries)
    warnings: tuple[str, ...] = ()
    error: PipelineError | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if self.status is PipelineStatus.FAILED and self.error is None:
            raise ValueError("failed preprocessing result requires an error")
        if self.status is not PipelineStatus.FAILED and self.error is not None:
            raise ValueError("only failed preprocessing results may contain an error")
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("pipeline timestamps must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json_value(self))

    def to_json(self, *, indent: int | None = 2) -> str:
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)
