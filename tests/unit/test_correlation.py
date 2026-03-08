"""Tests for the cross-module correlation engine."""

from searchprobe.core.protocols import AnalysisResult
from searchprobe.core.signals import Signal, SignalType
from searchprobe.intelligence.correlation import CorrelationEngine
from searchprobe.intelligence.models import CategoryIntelligenceProfile, SignalVector
from searchprobe.intelligence.session import SharedContext


def _make_context() -> SharedContext:
    """Build a SharedContext with mock analysis results across modules."""
    ctx = SharedContext()

    # Geometry results
    geo = AnalysisResult(
        analysis_type="geometry",
        categories=["negation", "polysemy", "numeric_precision", "temporal_constraint"],
        summary={},
        details=[
            {"category": "negation", "vulnerability_score": 0.9},
            {"category": "polysemy", "vulnerability_score": 0.6},
            {"category": "numeric_precision", "vulnerability_score": 0.85},
            {"category": "temporal_constraint", "vulnerability_score": 0.3},
        ],
    )
    ctx.add_result("geometry", geo)

    # Perturbation results
    perturb = AnalysisResult(
        analysis_type="perturbation",
        categories=["negation", "polysemy", "numeric_precision", "temporal_constraint"],
        summary={
            "stability_by_category": {
                "negation": 0.2,
                "polysemy": 0.5,
                "numeric_precision": 0.25,
                "temporal_constraint": 0.8,
            }
        },
    )
    ctx.add_result("perturbation", perturb)

    # Validation results
    val = AnalysisResult(
        analysis_type="validation",
        summary={
            "improvements_by_category": {
                "negation": 0.4,
                "polysemy": 0.15,
                "numeric_precision": 0.35,
                "temporal_constraint": 0.05,
            }
        },
    )
    ctx.add_result("validation", val)

    return ctx


def test_build_signal_vectors():
    engine = CorrelationEngine()
    ctx = _make_context()
    vectors = engine.build_signal_vectors(ctx)

    assert "negation" in vectors
    assert vectors["negation"].vulnerability_score == 0.9
    assert vectors["negation"].perturbation_stability == 0.2
    assert vectors["negation"].embedding_gap == 0.4


def test_compute_correlation_matrix():
    engine = CorrelationEngine()
    ctx = _make_context()
    vectors = engine.build_signal_vectors(ctx)
    matrix = engine.compute_correlation_matrix(vectors)

    # Should have correlations between vulnerability, stability, embedding_gap
    if matrix:
        assert "vulnerability_score" in matrix
        # Vulnerability and stability should be negatively correlated
        # (high vulnerability -> low stability)
        rho = matrix["vulnerability_score"].get("perturbation_stability")
        if rho is not None:
            assert rho < 0  # Expect negative correlation


def test_generate_profiles():
    engine = CorrelationEngine()
    ctx = _make_context()
    profiles = engine.generate_profiles(ctx)

    assert len(profiles) > 0
    # Profiles sorted by risk descending
    assert profiles[0].risk_score >= profiles[-1].risk_score

    # Negation should be high risk
    neg_profile = next((p for p in profiles if p.category == "negation"), None)
    assert neg_profile is not None
    assert neg_profile.risk_level() == "HIGH"

    # Temporal should be lower risk
    temp_profile = next((p for p in profiles if p.category == "temporal_constraint"), None)
    assert temp_profile is not None
    assert temp_profile.risk_score < neg_profile.risk_score


def test_signal_vector_completeness():
    v = SignalVector(category="test", vulnerability_score=0.5, perturbation_stability=0.3)
    assert v.completeness == 2 / 5

    v2 = SignalVector(
        category="test",
        vulnerability_score=0.5,
        perturbation_stability=0.3,
        embedding_gap=0.1,
        evolution_fitness=0.8,
        evaluation_score=0.6,
    )
    assert v2.completeness == 1.0


def test_profile_risk_level():
    sv = SignalVector(category="test")
    p = CategoryIntelligenceProfile(category="test", signal_vector=sv, risk_score=0.8)
    assert p.risk_level() == "HIGH"

    p.risk_score = 0.5
    assert p.risk_level() == "MEDIUM"

    p.risk_score = 0.2
    assert p.risk_level() == "LOW"


def test_profile_to_dict():
    sv = SignalVector(category="negation", vulnerability_score=0.9)
    p = CategoryIntelligenceProfile(
        category="negation",
        signal_vector=sv,
        risk_score=0.85,
        primary_failure_modes=["negation_collapse"],
        recommendations=["Fix negation handling"],
    )
    d = p.to_dict()
    assert d["category"] == "negation"
    assert d["risk_level"] == "HIGH"
    assert d["signal_vector"]["vulnerability"] == 0.9
