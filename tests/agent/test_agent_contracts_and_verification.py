from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from grounded_video_agent.agent import (
    AgentAction,
    AgentDecision,
    AnswerClaimDraft,
    AnswerDraft,
    QuestionIntent,
)
from grounded_video_agent.agent.tools import ToolRuntimeContext
from grounded_video_agent.agent.verification import EvidenceVerifier
from grounded_video_agent.domain import (
    EvidenceBundle,
    EvidenceItem,
    EvidenceModality,
    EvidenceVerificationStatus,
    TimeRange,
)
from grounded_video_agent.workspace.catalog import ArtifactCatalog


def test_agent_decision_enforces_single_action_shape() -> None:
    decision = AgentDecision(
        intent=QuestionIntent.LOCAL_EVENT,
        current_goal="Find the relevant event",
        action=AgentAction.CALL_TOOL,
        tool_name="search_video_transcript",
        tool_arguments={"query": "door"},
    )

    assert decision.tool_name == "search_video_transcript"
    with pytest.raises(ValidationError, match="tool calls"):
        AgentDecision(
            intent=QuestionIntent.LOCAL_EVENT,
            current_goal="Answer",
            action=AgentAction.ANSWER,
            tool_name="search_video_transcript",
            supporting_evidence_ids=("evidence-1",),
        )
    with pytest.raises(ValidationError, match="supporting_evidence_ids"):
        AgentDecision(
            intent=QuestionIntent.LOCAL_EVENT,
            current_goal="Answer",
            action=AgentAction.ANSWER,
        )


def test_evidence_verifier_rejects_unknown_evidence_ids() -> None:
    evidence = EvidenceItem(
        "evidence-1",
        "video-1",
        TimeRange(1_000, 2_000),
        EvidenceModality.TRANSCRIPT,
        ("chunk-1",),
        text="A door opens.",
    )
    bundle = EvidenceBundle("bundle-1", "What happened?", (evidence,))
    draft = AnswerDraft(
        answer="A door opens.",
        claims=(AnswerClaimDraft(text="A door opens.", evidence_ids=("made-up",)),),
    )

    result = EvidenceVerifier().verify(
        draft,
        bundle,
        video_id="video-1",
        intent=QuestionIntent.LOCAL_EVENT,
        duration_ms=10_000,
        min_global_coverage=0.7,
    )

    assert result.report.status is EvidenceVerificationStatus.INSUFFICIENT
    assert result.report.direct_support is False
    assert result.verified_evidence_ids == ()


def test_evidence_verifier_requires_visual_modality_for_visual_intent() -> None:
    evidence = EvidenceItem(
        "evidence-1",
        "video-1",
        TimeRange(1_000, 2_000),
        EvidenceModality.TRANSCRIPT,
        ("chunk-1",),
        text="A speaker says the car is red.",
    )
    bundle = EvidenceBundle("bundle-1", "What color is the car?", (evidence,))
    draft = AnswerDraft(
        answer="The car is red.",
        claims=(
            AnswerClaimDraft(text="The car is red.", evidence_ids=("evidence-1",)),
        ),
    )

    result = EvidenceVerifier().verify(
        draft,
        bundle,
        video_id="video-1",
        intent=QuestionIntent.VISUAL,
        duration_ms=10_000,
        min_global_coverage=0.7,
    )

    assert result.report.status is EvidenceVerificationStatus.INSUFFICIENT
    assert "visual" in " ".join(result.report.missing_evidence).lower()


def test_tool_runtime_snapshot_restores_durable_ledgers() -> None:
    catalog = cast(ArtifactCatalog, object())
    runtime = ToolRuntimeContext("video-1", catalog, max_tool_calls=4)
    evidence = EvidenceItem(
        "evidence-1",
        "video-1",
        TimeRange(1_000, 2_000),
        EvidenceModality.TRANSCRIPT,
        ("chunk-1",),
        text="A door opens.",
    )
    runtime.evidence.add(evidence)
    runtime.coverage.add((evidence.time_range,))
    runtime.start_call("search_video_transcript")
    runtime.record_information_gain(False)

    restored = ToolRuntimeContext.from_snapshot(runtime.snapshot(), catalog)

    assert restored.evidence.get("evidence-1") == evidence
    assert restored.coverage.ranges == (TimeRange(1_000, 2_000),)
    assert restored.call_count == 1
    assert restored.consecutive_no_information_gain == 1
