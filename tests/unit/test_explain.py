"""Tests for triplet probes and failure prediction."""

import math

import numpy as np
import pytest

from searchprobe.explain.probe import predict_failures, probe_markdown, roc_auc, run_probe
from searchprobe.explain.scorers import BM25Scorer, prompts_for
from searchprobe.explain.triplets import Triplet, load_triplets, save_triplets
from searchprobe.geometry.metrics import isotropy_score, local_intrinsic_dimensionality


class KeywordScorer:
    """Scores by whether the doc contains 'good' (a perfect oracle for the fixtures)."""

    name = "oracle"

    def score(self, query, docs):
        return [1.0 if "good" in d else 0.0 for d in docs]


def _triplets():
    out = []
    for i in range(6):
        out.append(Triplet(f"case{i}", "negation", f"query {i}", "good doc", "bad doc",
                           "p", "n", "known_sets"))
    return out


def test_probe_accuracy_and_roundtrip(tmp_path):
    triplets = _triplets()
    save_triplets(triplets, tmp_path / "t.jsonl")
    assert load_triplets(tmp_path / "t.jsonl") == triplets
    scores = run_probe(triplets, [KeywordScorer(), BM25Scorer()])
    oracle = [s for s in scores if s.scorer == "oracle"]
    assert all(s.correct for s in oracle)
    md = probe_markdown(scores)
    assert "oracle" in md and "100%" in md


def test_bm25_prefers_query_terms():
    s = BM25Scorer().score("landlocked country", ["a landlocked country in Africa",
                                                  "an island nation"])
    assert s[0] > s[1]


def test_roc_auc():
    assert roc_auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0
    assert roc_auc([0.5, 0.5], [1, 0]) == 0.5
    assert math.isnan(roc_auc([0.5], [1]))


def test_predict_failures_detects_linked_cases():
    from searchprobe.explain.probe import TripletScore

    scores = []
    violated = {}
    for i in range(20):
        hard = i < 10  # open models fail on the first 10 cases...
        scores.append(TripletScore(f"c{i}", "negation", "m", 0.0 if hard else 1.0,
                                   1.0 if hard else 0.0))
        violated[("exa", f"c{i}")] = hard  # ...and so does the provider
    (pred,) = predict_failures(scores, violated, n_bootstrap=200)
    assert pred.auc == 1.0 and pred.n_cases == 20 and pred.n_violated == 10


def test_embedding_prompts_are_model_specific():
    assert prompts_for("intfloat/e5-base-v2") == ("query: ", "passage: ")
    assert prompts_for("sentence-transformers/all-MiniLM-L6-v2") == ("", "")


def test_geometry_stats_refuse_tiny_samples():
    rng = np.random.default_rng(0)
    small = rng.normal(size=(12, 384))
    assert math.isnan(isotropy_score(small))
    assert math.isnan(local_intrinsic_dimensionality(small))
    big = rng.normal(size=(400, 16))
    assert isotropy_score(big) == pytest.approx(1.0, abs=0.05)
