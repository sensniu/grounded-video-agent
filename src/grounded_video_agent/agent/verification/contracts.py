from __future__ import annotations

from dataclasses import dataclass

from grounded_video_agent.agent.contracts import AgentCitation, AgentClaim
from grounded_video_agent.domain import EvidenceVerificationReport


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    report: EvidenceVerificationReport
    claims: tuple[AgentClaim, ...]
    citations: tuple[AgentCitation, ...]
    verified_evidence_ids: tuple[str, ...]
