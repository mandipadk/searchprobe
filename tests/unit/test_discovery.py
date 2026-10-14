"""Tests for adversarial discovery (checker-decided fitness, no LLM)."""

import asyncio
import random

from searchprobe.cases.models import ExcludeConstraint, KnownEntity, TestCase, TopicConstraint
from searchprobe.discovery.mutations import add_distractor_context, question_form, rephrase_operator
from searchprobe.discovery.search import CaseEvaluator, MapElites, cell_of, discover, promote
from searchprobe.pipeline.cost_tracker import CostTracker
from tests.fakes import FakeProvider

ANSWERS = ["Norway", "Switzerland", "Iceland"]
NEGATIVES = ["France", "Germany", "Spain"]


def _wiki(name):
    return f"https://en.wikipedia.org/wiki/{name}"


def _case():
    return TestCase(
        text="European countries that are not members of the European Union",
        category="negation",
        operators=["negation"],
        constraints=[TopicConstraint(id="c1", subject="a European country"),
                     ExcludeConstraint(id="c2", concept="an EU member state")],
        answer_set=[KnownEntity(label=n, wikipedia_url=_wiki(n)) for n in ANSWERS],
        negative_set=[KnownEntity(label=n, wikipedia_url=_wiki(n)) for n in NEGATIVES],
        answer_set_complete=True,
    )


def target_engine(query):
    """Handles the explicit 'are not' but fails on every rephrasing of the operator."""
    if "that are not members" in query and "does not count" not in query \
            and "unlike" not in query \
            and "including" not in query and not query.endswith("?"):
        return [_wiki(n) for n in ANSWERS]
    return [_wiki(n) for n in NEGATIVES]


def reference_engine(query):
    return [_wiki(n) for n in ANSWERS]


def test_mutations_keep_constraints_and_answers():
    rng = random.Random(0)
    case = _case()
    for fn in (rephrase_operator, add_distractor_context, question_form):
        child = fn(case, rng)
        assert child is not None and child.text != case.text
        assert child.constraints == case.constraints
        assert child.answer_set == case.answer_set
        assert child.provenance.generator == "discovered"
        assert child.id != case.id


def test_archive_keeps_best_per_cell():
    from searchprobe.discovery.search import Evaluation

    archive = MapElites()
    case = _case()
    assert archive.consider(case, Evaluation(0.1, None, None, True))
    assert not archive.consider(case, Evaluation(0.05, None, None, True))
    assert archive.consider(case, Evaluation(0.9, None, None, True))
    assert archive.cells[cell_of(case)].evaluation.fitness == 0.9


def test_discovery_finds_answerable_failures_and_promotes_them():
    providers = {"target": FakeProvider(target_engine, name="target"),
                 "ref": FakeProvider(reference_engine, name="ref")}
    evaluator = CaseEvaluator(providers, "target", "ref", judge=None, k=3,
                              cost_tracker=CostTracker(budget_limit=100))
    seed_eval = asyncio.run(evaluator.evaluate(_case()))
    assert seed_eval.fitness == 0.0 and seed_eval.answerable

    result = asyncio.run(discover([_case()], evaluator, iterations=30, reevaluate=1, seed=1))
    best = result.archive.elites()[0]
    assert best.evaluation.fitness == 1.0
    assert best.case.provenance.generator == "discovered"
    promoted = promote(result, min_fitness=0.5)
    assert promoted and all(c.answer_set for c in promoted if not c.twin_of)


def test_unanswerable_cases_score_zero():
    providers = {"target": FakeProvider(lambda q: [_wiki(n) for n in NEGATIVES], name="t"),
                 "ref": FakeProvider(lambda q: [_wiki(n) for n in NEGATIVES], name="r")}
    evaluator = CaseEvaluator(providers, "target", "ref", judge=None, k=3)
    ev = asyncio.run(evaluator.evaluate(_case()))
    assert ev.fitness == 0.0 and not ev.answerable
