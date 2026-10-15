"""Tests for TestCase v2: schema, conversion, twins, generators and I/O."""

import asyncio
import json
from datetime import date

import httpx
import pytest
from pydantic import ValidationError

from searchprobe.cases.convert import query_to_case
from searchprobe.cases.generators.llm import build_cases
from searchprobe.cases.generators.structured import generate_structured
from searchprobe.cases.generators.wikidata import WikidataClient, _not_member
from searchprobe.cases.io import load_cases, merge_cases, save_cases
from searchprobe.cases.models import (
    DateConstraint,
    ExcludeConstraint,
    NumericConstraint,
    RelationConstraint,
    TestCase,
    TopicConstraint,
)
from searchprobe.cases.twins import (
    antonym_flip,
    argument_swap,
    attach_twins,
    date_shift,
    negation_flip,
    numeric_scale,
)
from searchprobe.queries.taxonomy import AdversarialCategory


class TestSchema:
    def test_between_requires_ordered_bounds(self):
        with pytest.raises(ValidationError):
            NumericConstraint(id="c1", attribute="h", op="between", value=200, value_max=150)
        with pytest.raises(ValidationError):
            NumericConstraint(id="c1", attribute="h", op="between", value=200)

    def test_duplicate_constraint_ids_rejected(self):
        with pytest.raises(ValidationError):
            TestCase(text="q", category="negation", constraints=[
                TopicConstraint(id="c1", subject="a"), TopicConstraint(id="c1", subject="b")])

    def test_ids_are_deterministic(self):
        a = TestCase(text="Companies  NOT in AI", category="negation")
        b = TestCase(text="companies not in ai", category="negation")
        assert a.id == b.id

    def test_numeric_check(self):
        c = NumericConstraint(id="c1", attribute="x", op="approx", value=100, tolerance=0.1)
        assert c.check(105) and not c.check(120)

    def test_roundtrip_discriminated_union(self, tmp_path):
        case = TestCase(text="q", category="temporal_constraint", constraints=[
            DateConstraint(id="c1", field="published", start=date(2025, 1, 1))])
        save_cases([case], tmp_path / "x.jsonl")
        (loaded,) = load_cases(tmp_path / "x.jsonl")
        assert isinstance(loaded.constraints[0], DateConstraint)
        assert loaded == case


class TestConvert:
    def test_v1_ground_truth_maps_to_typed_constraints(self):
        case = query_to_case(
            "Companies NOT in AI", "negation",
            {"strategy": "must_not_contain", "must_not_contain_keywords": ["AI", "ML"],
             "numeric_min": 1, "numeric_max": 5, "expected_language": "de",
             "date_min": "2025-01-01T00:00:00"},
        )
        types = [c.type for c in case.constraints]
        # The query itself is always appended as a hard criterion.
        assert types == ["exclude", "numeric", "date", "language", "freeform"]
        assert case.operators == ["negation"]

    def test_case_without_ground_truth_gets_a_hard_criterion(self):
        case = query_to_case("Something hard", "compositional", None)
        assert case.hard_constraints and case.constraints[0].type == "freeform"


def _neg_case():
    return TestCase(text="Companies that are NOT in the AI industry", category="negation",
                    operators=["negation"], constraints=[
                        TopicConstraint(id="c1", subject="a company"),
                        ExcludeConstraint(id="c2", concept="an AI company")])


class TestTwins:
    def test_negation_flip_turns_exclusion_into_requirement(self):
        twin = negation_flip(_neg_case())
        assert twin.text == "Companies that are in the AI industry"
        assert twin.constraint("c2").type == "topic"
        assert twin.twin_of == _neg_case().id

    def test_numeric_scale_changes_text_and_constraint(self):
        case = TestCase(text="Startups with exactly 50 employees", category="numeric_precision",
                        operators=["numeric"],
                        constraints=[NumericConstraint(id="c1", attribute="employees", op="eq",
                                                       value=50)])
        twin = numeric_scale(case)
        assert twin.text == "Startups with exactly 500 employees"
        assert twin.constraint("c1").value == 500

    def test_numeric_scale_skips_thresholds_whose_answers_nest(self):
        case = TestCase(text="Startups with more than 50 employees", category="numeric_precision",
                        constraints=[NumericConstraint(id="c1", attribute="employees", op="gt",
                                                       value=50)])
        assert numeric_scale(case) is None  # every answer to "> 500" also answers "> 50"

    def test_numeric_scale_refuses_when_number_not_in_text(self):
        case = TestCase(text="Companies with exactly fifty employees", category="numeric_precision",
                        constraints=[NumericConstraint(id="c1", attribute="e", op="eq", value=50)])
        assert numeric_scale(case) is None

    def test_date_shift(self):
        case = TestCase(text="Tech layoffs announced in January 2025", category="temporal_constraint",
                        constraints=[DateConstraint(id="c1", start=date(2025, 1, 1),
                                                    end=date(2025, 1, 31))])
        twin = date_shift(case)
        assert twin.text == "Tech layoffs announced in January 2024"
        assert twin.constraint("c1").start == date(2024, 1, 1)

    def test_argument_swap(self):
        case = TestCase(text="Companies that Google acquired from Motorola", category="compositional",
                        constraints=[RelationConstraint(id="c1", subject="Google",
                                                        predicate="acquired from",
                                                        object="Motorola")])
        twin = argument_swap(case)
        assert twin.text == "Companies that Motorola acquired from Google"

    def test_antonym_flip(self):
        case = TestCase(text="Strategies to decrease employee turnover",
                        category="antonym_confusion")
        assert antonym_flip(case).text == "Strategies to increase employee turnover"

    def test_attach_twins_links_both_ways_and_never_twins_twins(self):
        out = attach_twins([_neg_case()])
        base = next(c for c in out if not c.twin_of)
        twin = next(c for c in out if c.twin_of)
        assert base.twins == {"negation_flip": twin.id}
        assert len(attach_twins(out)) == len(out)


def test_merge_never_overwrites_gold():
    gold = _neg_case().model_copy(update={"validation": "gold", "notes": "kept"})
    auto = _neg_case().model_copy(update={"validation": "auto", "notes": "new"})
    (merged,) = merge_cases([gold], [auto])
    assert merged.notes == "kept"


def test_structured_generator_uses_mechanical_constraints():
    cases = generate_structured()
    langs = [c for c in cases if c.category == AdversarialCategory.CROSS_LINGUAL]
    assert langs and all(any(k.type == "language" for k in c.constraints) for c in langs)
    pub = [c for c in cases if any(getattr(k, "field", None) == "published"
                                   for k in c.constraints)]
    assert pub and all(c.twins or c.twin_of for c in pub)
    assert len({c.id for c in cases}) == len(cases)


def _flat(**kw):
    base = {k: None for k in ["subject", "concept", "attribute", "op", "value", "value_max",
                              "unit", "field", "start", "end", "name", "predicate", "object",
                              "code", "requirement"]}
    base.update({"wrong": [], "allowed": [], "disallowed": [], "hard": True, "description": ""})
    base.update(kw)
    return base


def test_llm_build_cases_validates_and_twins():
    items = [
        {"text": "Rivers in Europe longer than 1000 km", "difficulty": "hard",
         "adversarial_reason": "r",
         "constraints": [_flat(id="c1", type="topic", subject="a river in Europe"),
                         _flat(id="c2", type="numeric", attribute="length", op="gt", value=1000,
                               unit="km")],
         "twin": {"text": "Rivers in Europe longer than 100 km", "relation": "numeric_scale",
                  "constraints": [_flat(id="c1", type="topic", subject="a river in Europe"),
                                  _flat(id="c2", type="numeric", attribute="length", op="gt",
                                        value=100, unit="km")]}},
        {"text": "Bad range", "difficulty": "hard", "adversarial_reason": "r", "twin": None,
         "constraints": [_flat(id="c1", type="numeric", attribute="x", op="between", value=5,
                               value_max=1)]},
        {"text": "Only vibes", "difficulty": "medium", "adversarial_reason": "r", "twin": None,
         "constraints": [_flat(id="c1", type="freeform", requirement="be good")]},
    ]
    cases, rejected = build_cases(items, AdversarialCategory.NUMERIC_PRECISION, "m")
    texts = {c.text for c in cases}
    assert texts == {"Rivers in Europe longer than 1000 km", "Rivers in Europe longer than 100 km"}
    assert len(rejected) == 2
    base = next(c for c in cases if not c.twin_of)
    assert base.twins == {"numeric_scale": next(c for c in cases if c.twin_of).id}


def _sparql_transport():
    def row(qid, label, member=False):
        return {"item": {"value": f"http://www.wikidata.org/entity/{qid}"},
                "itemLabel": {"value": label},
                "article": {"value": f"https://en.wikipedia.org/wiki/{label.replace(' ', '_')}"}}

    all_rows = [row("Q1", "France"), row("Q2", "Germany"), row("Q3", "Norway"),
                row("Q4", "Switzerland"), row("Q5", "Spain")]
    members = [all_rows[0], all_rows[1], all_rows[4]]

    def handler(request: httpx.Request) -> httpx.Response:
        query = dict(httpx.QueryParams(request.content.decode()))["query"]
        rows = members if "ps:P463 wd:Q458" in query else all_rows
        return httpx.Response(200, json={"results": {"bindings": rows}})

    return httpx.MockTransport(handler)


def test_wikidata_not_member_builds_case_and_twin_with_answer_sets():
    async def go():
        async with httpx.AsyncClient(transport=_sparql_transport()) as http:
            return await _not_member(WikidataClient(client=http), "Europe", "Q458",
                                     "the European Union")

    base, twin = asyncio.run(go())
    assert base.text == "European countries that are not members of the European Union"
    assert {e.label for e in base.answer_set} == {"Norway", "Switzerland"}
    assert {e.label for e in base.negative_set} == {"France", "Germany", "Spain"}
    assert base.twins["negation_flip"] == twin.id
    assert {e.label for e in twin.answer_set} == {"France", "Germany", "Spain"}
    assert json.loads(base.model_dump_json())["provenance"]["generator"] == "wikidata"


def test_verify_constants_flags_wrong_ids():
    from searchprobe.cases.generators.wikidata import verify_constants

    def handler(request):
        rows = [{"item": {"value": "http://www.wikidata.org/entity/Q170481"},
                 "itemLabel": {"value": "Asia-Pacific Economic Cooperation"}}]
        return httpx.Response(200, json={"results": {"bindings": rows}})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await verify_constants(WikidataClient(client=http), {"Q170481": "Mercosur"})

    (problem,) = asyncio.run(go())
    assert "Q170481" in problem and "Mercosur" in problem


def test_membership_excludes_ended_memberships():
    from searchprobe.cases.generators.wikidata import member_of

    pattern = member_of("Q458")
    assert "pq:P582" in pattern and "FILTER NOT EXISTS" in pattern
    assert "wd:Q756617 wd:Q35" in pattern  # kingdom-held memberships count for the country


def test_membership_corrections_are_applied():
    from searchprobe.cases.generators.wikidata import member_of

    assert "wd:Q214" in member_of("Q8908")  # Slovakia added to the Council of Europe
    assert "NOT IN (wd:Q408)" in member_of("Q7768")  # Australia removed from ASEAN
    assert "NOT IN" not in member_of("Q458")


def test_verify_memberships_flags_wrong_counts():
    from searchprobe.cases.generators.wikidata import verify_memberships

    def handler(request):
        return httpx.Response(200, json={"results": {"bindings": [{"n": {"value": "26"}}]}})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await verify_memberships(WikidataClient(client=http), {"Q458": 27})

    (problem,) = asyncio.run(go())
    assert "Q458" in problem and "27" in problem and "26" in problem
