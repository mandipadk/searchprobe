"""Tests for ranking-similarity metrics used by perturbation/metamorphic analysis."""

import numpy as np
import pytest

from searchprobe.perturbation.operators import PerturbationType, apply_perturbation
from searchprobe.perturbation.stability import normalize_url, rank_biased_overlap
from searchprobe.validation.cross_encoder import _compute_ndcg, to_probability

TOP10 = [f"u{i}" for i in range(10)]


class TestRBO:
    def test_identical_lists_score_one(self):
        assert rank_biased_overlap(TOP10, TOP10) == pytest.approx(1.0)

    def test_disjoint_lists_score_zero(self):
        assert rank_biased_overlap(TOP10, [f"v{i}" for i in range(10)]) == 0.0

    def test_top_weighted(self):
        swap_top = ["u1", "u0"] + TOP10[2:]
        swap_bottom = TOP10[:8] + ["u9", "u8"]
        assert rank_biased_overlap(TOP10, swap_bottom) > rank_biased_overlap(TOP10, swap_top)

    def test_symmetric_and_bounded(self):
        a, b = TOP10, ["u3", "x", "u0", "y", "u7"]
        assert rank_biased_overlap(a, b) == pytest.approx(rank_biased_overlap(b, a))
        assert 0.0 <= rank_biased_overlap(a, b) <= 1.0


class TestNDCG:
    def test_logits_are_mapped_to_probabilities(self):
        probs = to_probability(np.array([-3.0, 5.0, -8.0, 2.0]))
        assert probs.min() > 0 and probs.max() < 1
        assert 0.0 <= _compute_ndcg(list(probs)) <= 1.0

    def test_negative_gains_rejected(self):
        with pytest.raises(ValueError):
            _compute_ndcg([-2.0, -1.0])


def test_normalize_url_merges_equivalent_urls():
    assert normalize_url("https://www.Example.com/a/?utm_source=x") == normalize_url(
        "http://example.com/a"
    )


def test_word_delete_covers_every_word():
    query = "one two three four five six seven eight"
    variants = apply_perturbation(query, PerturbationType.WORD_DELETE, max_variants=5)
    assert len(variants) == 8


def test_word_swap_is_reproducible():
    import random

    q = "a b c d e f g"
    v1 = apply_perturbation(q, PerturbationType.WORD_SWAP, 3, rng=random.Random(1))
    v2 = apply_perturbation(q, PerturbationType.WORD_SWAP, 3, rng=random.Random(1))
    assert v1 == v2
