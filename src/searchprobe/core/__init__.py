"""Core foundation layer for SearchProbe framework."""

from searchprobe.core.exceptions import (
    BudgetExhaustedError,
    ConfigurationError,
    EvaluationError,
    PipelineError,
    ProviderError,
    RateLimitError,
    SearchProbeError,
)
from searchprobe.core.protocols import AnalysisResult, Analyzer
from searchprobe.core.signals import Signal, SignalBus, SignalType

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
