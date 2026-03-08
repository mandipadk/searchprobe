"""Tests for ground truth validation strategies."""

from searchprobe.intelligence.ground_truth import (
    DomainMatchStrategy,
    EntityMatchStrategy,
    GroundTruthEngine,
    MustContainStrategy,
    MustNotContainStrategy,
    NumericRangeStrategy,
    PatternMatchStrategy,
)


SAMPLE_RESULTS = [
    {
        "title": "Python Data Science",
        "url": "https://example.com/python",
        "snippet": "Python has 1500 GitHub stars",
        "content": "Python libraries for data science. Founded in 2020.",
    },
    {
        "title": "Java vs Python",
        "url": "https://blog.test.com/java",
        "snippet": "Comparing Java and Python",
        "content": "Java is faster but Python is easier.",
    },
    {
        "title": "Rust Frameworks",
        "url": "https://rust.example.com/fw",
        "snippet": "Rust frameworks with 50 contributors",
        "content": "Actix-web and Rocket are popular.",
    },
]


class TestMustContain:
    def test_terms_present(self):
        s = MustContainStrategy()
        outcome = s.validate(SAMPLE_RESULTS, {"terms": ["Python"]})
        assert outcome.passed
        assert outcome.score > 0.5
        assert len(outcome.matched_results) == 2

    def test_terms_absent(self):
        s = MustContainStrategy()
        outcome = s.validate(SAMPLE_RESULTS, {"terms": ["Haskell"]})
        assert not outcome.passed
        assert outcome.score == 0.0

    def test_empty_terms(self):
        s = MustContainStrategy()
        outcome = s.validate(SAMPLE_RESULTS, {"terms": []})
        assert outcome.passed


class TestMustNotContain:
    def test_no_violations(self):
        s = MustNotContainStrategy()
        outcome = s.validate(SAMPLE_RESULTS, {"terms": ["Haskell"]})
        assert outcome.passed
        assert outcome.score == 1.0

    def test_violations(self):
        s = MustNotContainStrategy()
        outcome = s.validate(SAMPLE_RESULTS, {"terms": ["Java"]})
        assert not outcome.passed
        assert len(outcome.matched_results) == 1


class TestEntityMatch:
    def test_entities_found(self):
        s = EntityMatchStrategy()
        outcome = s.validate(SAMPLE_RESULTS, {"entities": ["Python", "Rust"]})
        assert outcome.passed
        assert outcome.score == 1.0

    def test_entities_missing(self):
        s = EntityMatchStrategy()
        outcome = s.validate(SAMPLE_RESULTS, {"entities": ["Go", "Haskell"]})
        assert not outcome.passed
        assert outcome.score == 0.0


class TestNumericRange:
    def test_in_range(self):
        s = NumericRangeStrategy()
        outcome = s.validate(SAMPLE_RESULTS, {"min": 40, "max": 60})
        assert outcome.score > 0
        assert len(outcome.matched_results) >= 1

    def test_out_of_range(self):
        s = NumericRangeStrategy()
        outcome = s.validate(SAMPLE_RESULTS, {"min": 5000, "max": 10000})
        assert outcome.score == 0.0


class TestDomainMatch:
    def test_matching_domains(self):
        s = DomainMatchStrategy()
        outcome = s.validate(SAMPLE_RESULTS, {"domains": ["example.com"]})
        assert outcome.score > 0
        assert len(outcome.matched_results) >= 1

    def test_no_matching_domains(self):
        s = DomainMatchStrategy()
        outcome = s.validate(SAMPLE_RESULTS, {"domains": ["github.com"]})
        assert outcome.score == 0.0


class TestPatternMatch:
    def test_regex_match(self):
        s = PatternMatchStrategy()
        outcome = s.validate(SAMPLE_RESULTS, {"pattern": r"\d{4}"})
        assert outcome.score > 0  # Should match "2020", "1500"

    def test_invalid_regex(self):
        s = PatternMatchStrategy()
        outcome = s.validate(SAMPLE_RESULTS, {"pattern": r"[invalid"})
        assert not outcome.passed


class TestGroundTruthEngine:
    def test_full_validation(self):
        engine = GroundTruthEngine()
        ground_truth = {
            "must_contain": {"terms": ["Python"]},
            "must_not_contain": {"terms": ["Haskell"]},
            "entity_match": {"entities": ["Rust"]},
        }
        outcomes = engine.validate(SAMPLE_RESULTS, ground_truth)
        assert len(outcomes) == 3
        assert all(o.passed for o in outcomes.values())

    def test_objective_score(self):
        engine = GroundTruthEngine()
        ground_truth = {
            "must_contain": {"terms": ["Python"]},
            "must_not_contain": {"terms": ["Java"]},
        }
        outcomes = engine.validate(SAMPLE_RESULTS, ground_truth)
        score = engine.compute_objective_score(outcomes)
        assert 0.0 <= score <= 1.0

    def test_unknown_strategy_ignored(self):
        engine = GroundTruthEngine()
        outcomes = engine.validate(SAMPLE_RESULTS, {"unknown_strategy": {"foo": "bar"}})
        assert len(outcomes) == 0
