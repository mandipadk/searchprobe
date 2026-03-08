"""Tests for the exception hierarchy."""

from searchprobe.core.exceptions import (
    BudgetExhaustedError,
    ConfigurationError,
    EvaluationError,
    PipelineError,
    ProviderError,
    RateLimitError,
    SearchProbeError,
)


def test_hierarchy():
    """All exceptions inherit from SearchProbeError."""
    assert issubclass(ProviderError, SearchProbeError)
    assert issubclass(RateLimitError, ProviderError)
    assert issubclass(BudgetExhaustedError, SearchProbeError)
    assert issubclass(EvaluationError, SearchProbeError)
    assert issubclass(ConfigurationError, SearchProbeError)
    assert issubclass(PipelineError, SearchProbeError)


def test_provider_error_attributes():
    err = ProviderError("connection failed", provider="exa", status_code=500)
    assert err.provider == "exa"
    assert err.status_code == 500
    assert "connection failed" in str(err)


def test_rate_limit_error():
    err = RateLimitError(provider="tavily", retry_after=30.0)
    assert err.status_code == 429
    assert err.retry_after == 30.0
    assert isinstance(err, ProviderError)


def test_budget_exhausted():
    err = BudgetExhaustedError(spent=15.0, limit=10.0)
    assert err.spent == 15.0
    assert err.limit == 10.0


def test_pipeline_error_stage():
    err = PipelineError("stage failed", stage="evaluation")
    assert err.stage == "evaluation"
    assert "stage failed" in str(err)


def test_catch_broad():
    """Catching SearchProbeError catches all subtypes."""
    errors = [
        ProviderError("a", provider="x"),
        RateLimitError(provider="y"),
        BudgetExhaustedError(),
        EvaluationError("e"),
        ConfigurationError("c"),
        PipelineError("p"),
    ]
    for err in errors:
        try:
            raise err
        except SearchProbeError:
            pass  # All caught
