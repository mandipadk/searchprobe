"""Research intelligence layer for cross-module analysis and insight generation."""

from searchprobe.intelligence.models import SignalVector, CategoryIntelligenceProfile
from searchprobe.intelligence.taxonomy import FailureMode, FailureClassifier
from searchprobe.intelligence.session import SharedContext, ResearchSession

__all__ = [
    "SignalVector",
    "CategoryIntelligenceProfile",
    "FailureMode",
    "FailureClassifier",
    "SharedContext",
    "ResearchSession",
]
