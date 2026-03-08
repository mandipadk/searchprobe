"""Core foundation layer for SearchProbe framework."""

from searchprobe.core.exceptions import (
    SearchProbeError,
    ProviderError,
    RateLimitError,
    BudgetExhaustedError,
    EvaluationError,
    ConfigurationError,
    PipelineError,
)
from searchprobe.core.signals import Signal, SignalBus, SignalType
from searchprobe.core.protocols import AnalysisResult, Analyzer

__all__ = [
    "SearchProbeError",
    "ProviderError",
    "RateLimitError",
    "BudgetExhaustedError",
    "EvaluationError",
    "ConfigurationError",
    "PipelineError",
    "Signal",
    "SignalBus",
    "SignalType",
    "AnalysisResult",
    "Analyzer",
]
