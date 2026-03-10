"""Data models for constraint verification."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ConstraintStatus(str, Enum):
    """Status of a constraint check."""
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"  # Could not determine


class VerificationResult(BaseModel):
    """Result of verifying a single constraint against a document."""

    constraint_type: str
    constraint_description: str = ""
    status: ConstraintStatus
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str = Field(default="", description="Text evidence for the verdict")

    @property
    def satisfied(self) -> bool:
        return self.status == ConstraintStatus.SATISFIED
