"""Research intelligence layer for cross-module analysis and insight generation."""

from searchprobe.intelligence.models import CategoryIntelligenceProfile, SignalVector
from searchprobe.intelligence.session import ResearchSession, SharedContext
from searchprobe.intelligence.taxonomy import FailureClassifier, FailureMode

__all__ = [
    "SignalVector",
    "CategoryIntelligenceProfile",
    "FailureMode",
    "FailureClassifier",
    "SharedContext",
    "ResearchSession",
]
