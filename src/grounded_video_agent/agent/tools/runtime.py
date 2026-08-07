"""Per-question runtime state hidden from LLM tool arguments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from grounded_video_agent.capabilities._support import json_value
from grounded_video_agent.domain import (
    ArtifactRef,
    CapabilityRequestContext,
    CapabilityUsage,
    EvidenceBundle,
    EvidenceItem,
    FrameManifest,
    ResourceLimits,
    TimeRange,
)
from grounded_video_agent.workspace.catalog import ArtifactCatalog


class ToolBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DeliveryPolicy:
    evidence_clip_requested: bool = False
    verified_evidence_ids: frozenset[str] = frozenset()
    max_clips: int = 5
    max_clip_duration_ms: int = 45_000
    max_total_duration_ms: int = 120_000
    max_padding_ms: int = 5_000
    merge_gap_ms: int = 3_000

    def __post_init__(self) -> None:
        if any(not item.strip() for item in self.verified_evidence_ids):
            raise ValueError("verified_evidence_ids must not contain empty values")
        for name in (
            "max_clips",
            "max_clip_duration_ms",
            "max_total_duration_ms",
            "max_padding_ms",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.merge_gap_ms < 0:
            raise ValueError("merge_gap_ms must be non-negative")

    def permits(self, evidence_ids: tuple[str, ...]) -> bool:
        return self.evidence_clip_requested and set(evidence_ids).issubset(
            self.verified_evidence_ids
        )


@dataclass(frozen=True, slots=True)
class DeliveryState:
    delivery_id: str
    catalog_entry_id: str
    artifact: ArtifactRef
    evidence_ids: tuple[str, ...]


class DeliveryLedger:
    def __init__(self) -> None:
        self._items: dict[str, DeliveryState] = {}

    def put(self, state: DeliveryState) -> None:
        self._items[state.delivery_id] = state

    def get(self, delivery_id: str) -> DeliveryState:
        try:
            return self._items[delivery_id]
        except KeyError as error:
            raise KeyError(f"unknown delivery_id: {delivery_id}") from error

    @property
    def items(self) -> tuple[DeliveryState, ...]:
        return tuple(self._items.values())


def fingerprint(value: Any) -> str:
    payload = json.dumps(
        json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{fingerprint(value)[:24]}"


@dataclass(slots=True)
class CandidateState:
    candidate_id: str
    chunk_id: str
    exact_range: TimeRange
    inspection_range: TimeRange
    evidence_id: str
    matched_queries: list[str] = field(default_factory=list)


class CandidateLedger:
    def __init__(self) -> None:
        self._items: dict[str, CandidateState] = {}

    def register(self, state: CandidateState, query: str) -> tuple[CandidateState, bool]:
        existing = self._items.get(state.candidate_id)
        if existing is None:
            state.matched_queries.append(query)
            self._items[state.candidate_id] = state
            return state, True
        if query not in existing.matched_queries:
            existing.matched_queries.append(query)
        return existing, False

    def get(self, candidate_id: str) -> CandidateState:
        try:
            return self._items[candidate_id]
        except KeyError as error:
            raise KeyError(f"unknown candidate_id: {candidate_id}") from error


@dataclass(frozen=True, slots=True)
class ContextWindowState:
    context_window_id: str
    ranges: tuple[TimeRange, ...]
    chunk_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


class ContextLedger:
    def __init__(self) -> None:
        self._items: dict[str, ContextWindowState] = {}

    def put(self, state: ContextWindowState) -> None:
        self._items[state.context_window_id] = state

    def get(self, context_window_id: str) -> ContextWindowState:
        try:
            return self._items[context_window_id]
        except KeyError as error:
            raise KeyError(f"unknown context_window_id: {context_window_id}") from error


class EvidenceLedger:
    def __init__(self) -> None:
        self._items: dict[str, EvidenceItem] = {}

    def add(self, item: EvidenceItem) -> bool:
        is_new = item.evidence_id not in self._items
        self._items[item.evidence_id] = item
        return is_new

    def get(self, evidence_id: str) -> EvidenceItem:
        return self._items[evidence_id]

    @property
    def items(self) -> tuple[EvidenceItem, ...]:
        return tuple(self._items.values())


class ObservationMemory:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}
        self._frame_manifests: list[FrameManifest] = []

    def get(self, key: str, expected_type: type[Any]) -> Any | None:
        value = self._values.get(key)
        return value if isinstance(value, expected_type) else None

    def put(self, key: str, value: object) -> None:
        self._values[key] = value
        if isinstance(value, FrameManifest) and all(
            item.ref.manifest_id != value.ref.manifest_id for item in self._frame_manifests
        ):
            self._frame_manifests.append(value)

    def frame_manifest_for(self, frame_ids: tuple[str, ...]) -> FrameManifest | None:
        wanted = set(frame_ids)
        return next(
            (
                manifest
                for manifest in reversed(self._frame_manifests)
                if wanted.issubset({frame.frame_id for frame in manifest.frames})
            ),
            None,
        )


class CoverageLedger:
    def __init__(self) -> None:
        self._ranges: tuple[TimeRange, ...] = ()

    @property
    def ranges(self) -> tuple[TimeRange, ...]:
        return self._ranges

    def add(self, ranges: tuple[TimeRange, ...]) -> tuple[TimeRange, ...]:
        newly_covered = tuple(
            part for item in ranges for part in self._subtract(item, self._ranges)
        )
        self._ranges = merge_ranges((*self._ranges, *ranges))
        return merge_ranges(newly_covered)

    def _is_covered(self, item: TimeRange) -> bool:
        return any(known.contains_range(item) for known in self._ranges)

    @staticmethod
    def _subtract(
        item: TimeRange,
        known_ranges: tuple[TimeRange, ...],
    ) -> tuple[TimeRange, ...]:
        cursor = item.start_ms
        remaining: list[TimeRange] = []
        for known in known_ranges:
            overlap = known.intersection(item)
            if overlap is None:
                continue
            if cursor < overlap.start_ms:
                remaining.append(TimeRange(cursor, overlap.start_ms))
            cursor = max(cursor, overlap.end_ms)
        if cursor < item.end_ms:
            remaining.append(TimeRange(cursor, item.end_ms))
        return tuple(remaining)

    def unseen(self, total: TimeRange) -> tuple[TimeRange, ...]:
        cursor = total.start_ms
        unseen: list[TimeRange] = []
        for item in self._ranges:
            overlap = item.intersection(total)
            if overlap is None:
                continue
            if cursor < overlap.start_ms:
                unseen.append(TimeRange(cursor, overlap.start_ms))
            cursor = max(cursor, overlap.end_ms)
        if cursor < total.end_ms:
            unseen.append(TimeRange(cursor, total.end_ms))
        return tuple(unseen)


@dataclass(frozen=True, slots=True)
class SearchAttempt:
    query_fingerprint: str
    returned_candidate_ids: tuple[str, ...]
    new_candidate_ids: tuple[str, ...]
    exhausted: bool


@dataclass(slots=True)
class ToolRuntimeContext:
    video_id: str
    catalog: ArtifactCatalog
    trace_id: str | None = None
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    max_tool_calls: int = 20
    delivery_policy: DeliveryPolicy = field(default_factory=DeliveryPolicy)
    candidates: CandidateLedger = field(default_factory=CandidateLedger)
    contexts: ContextLedger = field(default_factory=ContextLedger)
    evidence: EvidenceLedger = field(default_factory=EvidenceLedger)
    memory: ObservationMemory = field(default_factory=ObservationMemory)
    coverage: CoverageLedger = field(default_factory=CoverageLedger)
    deliveries: DeliveryLedger = field(default_factory=DeliveryLedger)
    search_attempts: list[SearchAttempt] = field(default_factory=list)
    call_count: int = 0
    model_calls: int = 0
    consecutive_no_information_gain: int = 0

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video_id must not be empty")
        if self.trace_id is not None and not self.trace_id.strip():
            raise ValueError("trace_id must not be empty")
        if self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be positive")

    def start_call(self, tool_name: str) -> str:
        if self.call_count >= self.max_tool_calls:
            raise ToolBudgetExceeded("tool call budget exhausted")
        self.call_count += 1
        return f"{tool_name}_{self.call_count:04d}"

    def capability_context(self, call_id: str, suffix: str) -> CapabilityRequestContext:
        return CapabilityRequestContext(
            operation_id=f"{call_id}_{suffix}",
            limits=self.limits,
            trace_id=self.trace_id,
        )

    def record_usage(self, usage: CapabilityUsage) -> None:
        self.model_calls += usage.model_calls

    def record_information_gain(self, gained: bool) -> None:
        self.consecutive_no_information_gain = (
            0 if gained else self.consecutive_no_information_gain + 1
        )

    @property
    def requires_replan(self) -> bool:
        """Signal the graph router after two consecutive zero-gain tool calls."""

        return self.consecutive_no_information_gain >= 2

    def build_evidence_bundle(self, question: str) -> EvidenceBundle:
        """Freeze the evidence accumulated so far for deterministic verification."""

        if not question.strip():
            raise ValueError("question must not be empty")
        items = self.evidence.items
        covered_ranges = merge_ranges(tuple(item.time_range for item in items))
        return EvidenceBundle(
            stable_id(
                "bundle",
                (self.video_id, question, tuple(item.evidence_id for item in items)),
            ),
            question,
            items,
            covered_ranges,
        )


def merge_ranges(ranges: tuple[TimeRange, ...]) -> tuple[TimeRange, ...]:
    merged: list[TimeRange] = []
    for item in sorted(ranges):
        if not merged or merged[-1].end_ms < item.start_ms:
            merged.append(item)
            continue
        previous = merged[-1]
        merged[-1] = TimeRange(previous.start_ms, max(previous.end_ms, item.end_ms))
    return tuple(merged)
