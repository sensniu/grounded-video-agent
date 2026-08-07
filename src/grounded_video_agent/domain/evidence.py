"""Retrieval, local analysis, aggregation, and sufficiency records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from grounded_video_agent.domain._invariants import (
    require_finite_number,
    require_probability,
    require_text,
    require_unique_texts,
)
from grounded_video_agent.domain.artifacts import ArtifactRef
from grounded_video_agent.domain.timeline import TimeRange


class EvidenceModality(StrEnum):
    TRANSCRIPT = "transcript"
    OCR = "ocr"
    FRAME = "frame"
    CLIP = "clip"
    VISUAL_DESCRIPTION = "visual_description"
    VLM_OBSERVATION = "vlm_observation"


@dataclass(frozen=True, slots=True)
class EvidenceScore:
    name: str
    value: float

    def __post_init__(self) -> None:
        require_text(self.name, "name")
        require_finite_number(self.value, "value")


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    video_id: str
    time_range: TimeRange
    modality: EvidenceModality
    source_ids: tuple[str, ...]
    text: str | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    scores: tuple[EvidenceScore, ...] = ()
    confidence: float | None = None

    def __post_init__(self) -> None:
        require_text(self.evidence_id, "evidence_id")
        require_text(self.video_id, "video_id")
        require_unique_texts(self.source_ids, "source_ids")
        if not self.source_ids:
            raise ValueError("evidence requires at least one source id")
        if self.text is not None:
            require_text(self.text, "text")
        if self.text is None and not self.artifacts:
            raise ValueError("evidence requires text or an artifact")
        artifact_ids = tuple(artifact.artifact_id for artifact in self.artifacts)
        require_unique_texts(artifact_ids, "artifact_ids")
        score_names = tuple(score.name for score in self.scores)
        require_unique_texts(score_names, "score_names")
        require_probability(self.confidence)


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    rank: int
    item: EvidenceItem

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be a positive integer")


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    hits: tuple[RetrievalHit, ...]
    searched_modalities: tuple[EvidenceModality, ...]
    candidate_ranges: tuple[TimeRange, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.query, "query")
        ranks = tuple(hit.rank for hit in self.hits)
        if ranks != tuple(range(1, len(self.hits) + 1)):
            raise ValueError("retrieval hit ranks must be consecutive and start at one")
        evidence_ids = tuple(hit.item.evidence_id for hit in self.hits)
        require_unique_texts(evidence_ids, "evidence_ids")
        if len(set(self.searched_modalities)) != len(self.searched_modalities):
            raise ValueError("searched_modalities must be unique")


@dataclass(frozen=True, slots=True)
class RerankingResult:
    query: str
    ranked_items: tuple[RetrievalHit, ...]
    removed_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.query, "query")
        ranks = tuple(hit.rank for hit in self.ranked_items)
        if ranks != tuple(range(1, len(self.ranked_items) + 1)):
            raise ValueError("reranked hit ranks must be consecutive and start at one")
        require_unique_texts(self.removed_evidence_ids, "removed_evidence_ids")


@dataclass(frozen=True, slots=True)
class EvidenceConflict:
    conflict_id: str
    evidence_ids: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        require_text(self.conflict_id, "conflict_id")
        require_unique_texts(self.evidence_ids, "evidence_ids")
        if len(self.evidence_ids) < 2:
            raise ValueError("evidence conflict requires at least two evidence items")
        require_text(self.description, "description")


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    bundle_id: str
    question: str
    items: tuple[EvidenceItem, ...]
    covered_ranges: tuple[TimeRange, ...] = ()
    conflicts: tuple[EvidenceConflict, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.bundle_id, "bundle_id")
        require_text(self.question, "question")
        evidence_ids = tuple(item.evidence_id for item in self.items)
        require_unique_texts(evidence_ids, "evidence_ids")
        known_ids = set(evidence_ids)
        if any(
            evidence_id not in known_ids
            for conflict in self.conflicts
            for evidence_id in conflict.evidence_ids
        ):
            raise ValueError("conflicts must reference evidence in the bundle")

    @property
    def modalities(self) -> frozenset[EvidenceModality]:
        return frozenset(item.modality for item in self.items)


@dataclass(frozen=True, slots=True)
class CandidateWindow:
    candidate_id: str
    rank: int
    video_id: str
    time_range: TimeRange
    evidence_ids: tuple[str, ...]
    modalities: tuple[EvidenceModality, ...]
    chunk_ids: tuple[str, ...] = ()
    shot_ids: tuple[str, ...] = ()
    scores: tuple[EvidenceScore, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.candidate_id, "candidate_id")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("candidate rank must be a positive integer")
        require_text(self.video_id, "video_id")
        require_unique_texts(self.evidence_ids, "evidence_ids")
        if not self.evidence_ids:
            raise ValueError("candidate window requires evidence")
        if not self.modalities or len(set(self.modalities)) != len(self.modalities):
            raise ValueError("candidate modalities must be non-empty and unique")
        require_unique_texts(self.chunk_ids, "chunk_ids")
        require_unique_texts(self.shot_ids, "shot_ids")
        require_unique_texts((score.name for score in self.scores), "score_names")


@dataclass(frozen=True, slots=True)
class CandidateWindowSet:
    query: str
    video_id: str
    windows: tuple[CandidateWindow, ...]
    evidence: EvidenceBundle

    def __post_init__(self) -> None:
        require_text(self.query, "query")
        require_text(self.video_id, "video_id")
        ranks = tuple(window.rank for window in self.windows)
        if ranks != tuple(range(1, len(self.windows) + 1)):
            raise ValueError("candidate ranks must be consecutive and start at one")
        require_unique_texts(
            (window.candidate_id for window in self.windows),
            "candidate_ids",
        )
        if any(window.video_id != self.video_id for window in self.windows):
            raise ValueError("candidate windows must belong to video_id")
        if self.evidence.question != self.query:
            raise ValueError("candidate evidence question must match query")
        if any(item.video_id != self.video_id for item in self.evidence.items):
            raise ValueError("candidate evidence must belong to video_id")
        evidence_ids = {item.evidence_id for item in self.evidence.items}
        if any(
            evidence_id not in evidence_ids
            for window in self.windows
            for evidence_id in window.evidence_ids
        ):
            raise ValueError("candidate windows must reference bundled evidence")


@dataclass(frozen=True, slots=True)
class EvidenceClaim:
    claim_id: str
    text: str
    time_range: TimeRange
    supporting_evidence_ids: tuple[str, ...]
    confidence: float | None = None
    uncertainties: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.claim_id, "claim_id")
        require_text(self.text, "text")
        require_unique_texts(self.supporting_evidence_ids, "supporting_evidence_ids")
        require_unique_texts(self.uncertainties, "uncertainties")
        require_probability(self.confidence)


@dataclass(frozen=True, slots=True)
class LocalVLMAnalysis:
    analysis_id: str
    video_id: str
    question: str
    time_range: TimeRange
    observations: tuple[EvidenceItem, ...]
    claims: tuple[EvidenceClaim, ...]
    confidence: float | None = None

    def __post_init__(self) -> None:
        require_text(self.analysis_id, "analysis_id")
        require_text(self.video_id, "video_id")
        require_text(self.question, "question")
        require_probability(self.confidence)
        if any(item.video_id != self.video_id for item in self.observations):
            raise ValueError("all observations must belong to the analysis video")
        if any(not self.time_range.contains_range(item.time_range) for item in self.observations):
            raise ValueError("observations must be contained by the analysis time range")


class EvidenceVerificationStatus(StrEnum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"
    UNANSWERABLE = "unanswerable"
    OVER_BUDGET = "over_budget"


class EvidenceAction(StrEnum):
    SEARCH_ANOTHER_CHANNEL = "search_another_channel"
    EXPAND_TIME_WINDOW = "expand_time_window"
    INCREASE_FRAME_DENSITY = "increase_frame_density"
    RUN_OCR = "run_ocr"
    INSPECT_ADJACENT_CONTEXT = "inspect_adjacent_context"
    FIND_SECOND_EVIDENCE = "find_second_evidence"
    ANSWER = "answer"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class EvidenceVerificationReport:
    status: EvidenceVerificationStatus
    direct_support: bool
    temporal_coverage: float
    cross_modal_consistency: float
    missing_evidence: tuple[str, ...] = ()
    conflicts: tuple[EvidenceConflict, ...] = ()
    confidence: float | None = None
    recommended_actions: tuple[EvidenceAction, ...] = ()

    def __post_init__(self) -> None:
        require_probability(self.temporal_coverage, "temporal_coverage")
        require_probability(self.cross_modal_consistency, "cross_modal_consistency")
        require_probability(self.confidence)
        require_unique_texts(self.missing_evidence, "missing_evidence")
        if len(set(self.recommended_actions)) != len(self.recommended_actions):
            raise ValueError("recommended_actions must be unique")
        if self.status is EvidenceVerificationStatus.SUFFICIENT:
            if not self.direct_support or EvidenceAction.ANSWER not in self.recommended_actions:
                raise ValueError("sufficient evidence must directly support an answer")
