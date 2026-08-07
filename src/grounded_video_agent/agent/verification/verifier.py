"""Deterministic claim-to-evidence integrity and coverage checks."""

from __future__ import annotations

from statistics import fmean

from grounded_video_agent.agent.contracts import (
    AgentCitation,
    AgentClaim,
    AnswerDraft,
    QuestionIntent,
)
from grounded_video_agent.agent.tools.runtime import merge_ranges
from grounded_video_agent.domain import (
    EvidenceAction,
    EvidenceBundle,
    EvidenceItem,
    EvidenceModality,
    EvidenceVerificationReport,
    EvidenceVerificationStatus,
    TimeRange,
)

from .contracts import VerificationOutcome

_VISUAL_MODALITIES = {
    EvidenceModality.FRAME,
    EvidenceModality.VISUAL_DESCRIPTION,
    EvidenceModality.VLM_OBSERVATION,
}


class EvidenceVerifier:
    def verify(
        self,
        draft: AnswerDraft,
        bundle: EvidenceBundle,
        *,
        video_id: str,
        intent: QuestionIntent,
        duration_ms: int | None,
        min_global_coverage: float,
        coverage_ranges: tuple[TimeRange, ...] | None = None,
    ) -> VerificationOutcome:
        known = {item.evidence_id: item for item in bundle.items}
        missing: list[str] = []
        claims: list[AgentClaim] = []
        cited_ids: list[str] = []
        for index, draft_claim in enumerate(draft.claims, start=1):
            if intent is not QuestionIntent.METADATA and not draft_claim.evidence_ids:
                missing.append(f"Claim {index} has no supporting evidence.")
            unknown = tuple(item for item in draft_claim.evidence_ids if item not in known)
            if unknown:
                missing.append(
                    f"Claim {index} references unknown evidence: {', '.join(unknown)}."
                )
            valid_ids = tuple(item for item in draft_claim.evidence_ids if item in known)
            cited_ids.extend(valid_ids)
            claims.append(
                AgentClaim(
                    f"claim_{index:04d}",
                    draft_claim.text,
                    valid_ids,
                    draft_claim.confidence,
                    draft_claim.uncertainties,
                )
            )

        selected = tuple(known[item] for item in dict.fromkeys(cited_ids))
        wrong_video = tuple(item.evidence_id for item in selected if item.video_id != video_id)
        if wrong_video:
            missing.append(f"Evidence belongs to another video: {', '.join(wrong_video)}.")
        modalities = {item.modality for item in selected}
        if intent is QuestionIntent.VISUAL and not modalities.intersection(_VISUAL_MODALITIES):
            missing.append("The visual question has no visual-model or frame evidence.")
        if intent is QuestionIntent.SCREEN_TEXT and EvidenceModality.OCR not in modalities:
            missing.append("The screen-text question has no OCR evidence.")
        if intent is QuestionIntent.CAUSAL and len(selected) < 2:
            missing.append("The causal question lacks corroborating evidence.")

        temporal_coverage = _coverage_ratio(
            coverage_ranges if coverage_ranges is not None else bundle.covered_ranges,
            duration_ms,
        )
        if (
            intent in {QuestionIntent.GLOBAL, QuestionIntent.COUNT}
            and temporal_coverage < min_global_coverage
        ):
            missing.append(
                f"Timeline coverage {temporal_coverage:.2f} is below the required "
                f"{min_global_coverage:.2f}."
            )
        if bundle.conflicts:
            status = EvidenceVerificationStatus.CONFLICTING
        elif missing:
            status = EvidenceVerificationStatus.INSUFFICIENT
        else:
            status = EvidenceVerificationStatus.SUFFICIENT
        direct_support = not missing and (
            intent is QuestionIntent.METADATA or bool(selected)
        )
        actions = _recommended_actions(intent, missing, bundle.conflicts)
        confidence_values = tuple(
            claim.confidence for claim in claims if claim.confidence is not None
        )
        confidence = fmean(confidence_values) if confidence_values else None
        report = EvidenceVerificationReport(
            status=status,
            direct_support=direct_support,
            temporal_coverage=temporal_coverage,
            cross_modal_consistency=0.0 if bundle.conflicts else 1.0,
            missing_evidence=tuple(dict.fromkeys(missing)),
            conflicts=bundle.conflicts,
            confidence=confidence,
            recommended_actions=actions,
        )
        verified_ids = tuple(
            item.evidence_id
            for item in selected
            if item.video_id == video_id and item.evidence_id not in wrong_video
        )
        citations = tuple(_citation(known[item]) for item in dict.fromkeys(verified_ids))
        return VerificationOutcome(report, tuple(claims), citations, verified_ids)


def _coverage_ratio(
    ranges: tuple[TimeRange, ...],
    duration_ms: int | None,
) -> float:
    if duration_ms is None or duration_ms <= 0:
        return 0.0
    merged = merge_ranges(ranges)
    covered = sum(item.duration_ms for item in merged)
    return min(1.0, covered / duration_ms)


def _recommended_actions(
    intent: QuestionIntent,
    missing: list[str],
    conflicts: tuple[object, ...],
) -> tuple[EvidenceAction, ...]:
    if not missing and not conflicts:
        return (EvidenceAction.ANSWER,)
    actions: list[EvidenceAction] = []
    if intent is QuestionIntent.VISUAL:
        actions.append(EvidenceAction.INCREASE_FRAME_DENSITY)
    elif intent is QuestionIntent.SCREEN_TEXT:
        actions.append(EvidenceAction.RUN_OCR)
    elif intent in {QuestionIntent.GLOBAL, QuestionIntent.COUNT}:
        actions.append(EvidenceAction.SEARCH_ANOTHER_CHANNEL)
    elif intent is QuestionIntent.LOCAL_EVENT:
        actions.append(EvidenceAction.INSPECT_ADJACENT_CONTEXT)
    else:
        actions.append(EvidenceAction.FIND_SECOND_EVIDENCE)
    if conflicts and EvidenceAction.FIND_SECOND_EVIDENCE not in actions:
        actions.append(EvidenceAction.FIND_SECOND_EVIDENCE)
    return tuple(actions)


def _citation(item: EvidenceItem) -> AgentCitation:
    excerpt = item.text
    if excerpt is not None and len(excerpt) > 500:
        excerpt = excerpt[:497] + "..."
    return AgentCitation(
        item.evidence_id,
        item.time_range,
        item.modality.value,
        excerpt,
    )
