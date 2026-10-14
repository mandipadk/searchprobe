"""Per-document judgment records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from searchprobe.cases.models import TestCase, Verdict

VerdictSource = Literal["checker", "llm", "human"]
EvidenceSource = Literal["page", "snippet"]

# UMBRELA-style graded relevance (0-3), judged against the full query.
RELEVANCE_SCALE = {
    0: "Unrelated to the query.",
    1: "Related to the topic but does not answer the query or misses its requirements.",
    2: "Relevant: addresses the query's main requirement, with minor gaps or ambiguity.",
    3: "Fully answers the query and satisfies all of its requirements.",
}


class ConstraintVerdict(BaseModel):
    """Verdict on one constraint for one document."""

    constraint_id: str
    verdict: Verdict
    source: VerdictSource = "llm"
    evidence: str = Field(default="", description="Quote or reason supporting the verdict")


class DocJudgment(BaseModel):
    """A judged (case, document) pair."""

    case_id: str
    content_hash: str
    url: str
    judge_id: str
    prompt_version: str
    relevance: int = Field(..., ge=0, le=3)
    verdicts: list[ConstraintVerdict]
    evidence_source: EvidenceSource = "page"
    rationale: str = ""
    cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def verdict_for(self, constraint_id: str) -> Verdict:
        for v in self.verdicts:
            if v.constraint_id == constraint_id:
                return v.verdict
        return "unaddressed"

    def is_violating(self, case: TestCase) -> bool:
        """True if any hard constraint is violated."""
        return any(self.verdict_for(c.id) == "violated" for c in case.hard_constraints)

    def is_fully_satisfying(self, case: TestCase) -> bool:
        """True if every hard constraint is satisfied."""
        hard = case.hard_constraints
        return bool(hard) and all(self.verdict_for(c.id) == "satisfied" for c in hard)

    def gain(self, case: TestCase) -> float:
        """Graded gain for nDCG: relevance, zeroed if the document violates the query."""
        return 0.0 if self.is_violating(case) else float(self.relevance)
