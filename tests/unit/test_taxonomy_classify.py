"""Tests for the failure mode taxonomy and classifier."""

from searchprobe.intelligence.taxonomy import FailureClassifier, FailureMode


def test_classify_negation():
    classifier = FailureClassifier()
    modes = classifier.classify(
        "Results violating the negation constraint, returning opposite items",
        category="negation",
    )
    assert FailureMode.NEGATION_COLLAPSE in modes


def test_classify_numeric():
    classifier = FailureClassifier()
    modes = classifier.classify(
        "Wrong number of employees, expected exactly 50 but got results with 500",
        category="numeric_precision",
    )
    assert FailureMode.NUMERIC_BLINDNESS in modes


def test_classify_partial_constraint():
    classifier = FailureClassifier()
    modes = classifier.classify(
        "Only satisfies one of three constraints, partial match",
        category="multi_constraint",
    )
    assert FailureMode.PARTIAL_CONSTRAINT in modes


def test_category_boost():
    classifier = FailureClassifier()
    # "not" is an indicator for NEGATION_COLLAPSE, but if category is negation
    # it should be boosted higher
    modes_with_cat = classifier.classify("not relevant", category="negation")
    modes_without_cat = classifier.classify("not relevant", category="polysemy")

    # NEGATION_COLLAPSE should appear in both, but rank higher with negation category
    assert FailureMode.NEGATION_COLLAPSE in modes_with_cat
    if FailureMode.NEGATION_COLLAPSE in modes_without_cat:
        idx_with = modes_with_cat.index(FailureMode.NEGATION_COLLAPSE)
        idx_without = modes_without_cat.index(FailureMode.NEGATION_COLLAPSE)
        assert idx_with <= idx_without


def test_classify_empty():
    classifier = FailureClassifier()
    modes = classifier.classify("")
    assert modes == []


def test_classify_evaluation():
    classifier = FailureClassifier()
    eval_result = {
        "failure_modes": ["negation constraint violated", "opposite results returned"],
        "category": "negation",
    }
    modes = classifier.classify_evaluation(eval_result)
    assert len(modes) > 0


def test_get_root_cause():
    classifier = FailureClassifier()
    info = classifier.get_root_cause(FailureMode.NEGATION_COLLAPSE)
    assert info["severity"] == "high"
    assert "negation" in info["categories"]


def test_all_modes_have_root_causes():
    from searchprobe.intelligence.taxonomy import FAILURE_ROOT_CAUSES
    for mode in FailureMode:
        assert mode in FAILURE_ROOT_CAUSES, f"Missing root cause for {mode}"
