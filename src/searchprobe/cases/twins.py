"""Minimal-pair ("twin") generation.

A twin is the same query with exactly one logical operator changed: negation
removed, a number scaled, a year shifted, the arguments of a relation swapped,
or a polarity word flipped. Twins have the opposite (or a disjoint) correct
answer, so a search engine that returns the same results for a case and its
twin is ignoring the operator. That makes twins usable without any ground
truth (see ``searchprobe.metamorphic``).

Rule-based transforms are deterministic and conservative: they return None
rather than produce a mangled query.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date

from searchprobe.cases.models import (
    DateConstraint,
    ExcludeConstraint,
    FreeformConstraint,
    NumericConstraint,
    RelationConstraint,
    TestCase,
    TopicConstraint,
)

TwinFn = Callable[[TestCase], TestCase | None]

# (pattern, replacement) pairs that remove a negation, most specific first.
_NEGATION_REMOVALS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(are|is|was|were)\s+not\s+", re.IGNORECASE), r"\1 "),
    (re.compile(r"\b(have|has|had)\s+never\s+", re.IGNORECASE), r"\1 "),
    (re.compile(r"\b(do|does|did)\s+not\s+", re.IGNORECASE), ""),
    (re.compile(r"\bnever\s+", re.IGNORECASE), ""),
    (re.compile(r"\bwithout\b", re.IGNORECASE), "with"),
    (re.compile(r"\bbut\s+not\b", re.IGNORECASE), "and"),
    (re.compile(r",\s*not\s+", re.IGNORECASE), " and "),
    (re.compile(r"\bexcluding\b", re.IGNORECASE), "including"),
    (re.compile(r"\bnot\s+", re.IGNORECASE), ""),
]

ANTONYMS: dict[str, str] = {
    "increase": "decrease",
    "increasing": "decreasing",
    "reduce": "increase",
    "raise": "lower",
    "failed": "succeeded",
    "success": "failure",
    "simplify": "complicate",
    "slow down": "speed up",
    "accelerate": "slow down",
    "profitable": "unprofitable",
    "cheapest": "most expensive",
    "smallest": "largest",
    "oldest": "newest",
    "before": "after",
    "acquired": "were acquired by",
    "gain": "loss",
}
ANTONYMS.update({v: k for k, v in list(ANTONYMS.items()) if v not in ANTONYMS})


def _derive(base: TestCase, text: str, constraints: list, relation: str) -> TestCase | None:
    """Build the twin. Free-text requirements that quote the base query are rewritten
    for the twin's query; any other free-text requirement cannot be updated
    mechanically, so the transform is rejected rather than left contradictory."""
    updated = []
    for c in constraints:
        if isinstance(c, FreeformConstraint) and text not in c.requirement:
            if base.text not in c.requirement:
                return None
            c = c.model_copy(
                update={"requirement": c.requirement.replace(base.text, text), "description": ""}
            )
        updated.append(c)
    return TestCase(
        text=text,
        category=base.category,
        operators=base.operators,
        difficulty=base.difficulty,
        constraints=updated,
        twin_of=base.id,
        twin_relation=relation,
        provenance=base.provenance.model_copy(update={"notes": f"twin ({relation}) of {base.id}"}),
        validation="auto",
    )


def negation_flip(case: TestCase) -> TestCase | None:
    """Remove the negation: 'companies that are NOT in AI' -> 'companies that are in AI'."""
    excludes = [c for c in case.constraints if isinstance(c, ExcludeConstraint)]
    if len(excludes) != 1:  # one negation is removed from the text, so one exclusion flips
        return None
    for pattern, repl in _NEGATION_REMOVALS:
        new_text, n = pattern.subn(repl, case.text, count=1)
        if n:
            break
    else:
        return None
    new_text = re.sub(r"\s{2,}", " ", new_text).strip()
    if new_text.lower() == case.text.lower():
        return None
    constraints = []
    for c in case.constraints:
        if isinstance(c, ExcludeConstraint):
            constraints.append(
                TopicConstraint(
                    id=c.id,
                    subject=c.concept,
                    description=f"The result IS {c.concept}",
                )
            )
        else:
            constraints.append(c)
    return _derive(case, new_text, constraints, "negation_flip")


def _fmt_number(x: float) -> str:
    return f"{int(x):,}" if float(x).is_integer() else f"{x:g}"


def _disjoint_after_scaling(c: NumericConstraint, factor: float) -> bool:
    """A twin is useful only if no answer satisfies both it and its base.

    "more than 50" vs "more than 500" are nested (a company with 1,000 employees
    answers both), so threshold comparisons are not scaled.
    """
    if c.op in ("gt", "gte", "lt", "lte"):
        return False
    if c.op == "between":
        assert c.value_max is not None
        lo, hi = c.value * factor, c.value_max * factor
        return lo > c.value_max or hi < c.value
    band, scaled_band = abs(c.value) * c.tolerance, abs(c.value * factor) * c.tolerance
    return abs(c.value * factor - c.value) > band + scaled_band


def numeric_scale(case: TestCase, factor: float = 10.0) -> TestCase | None:
    """Scale the constrained number(s) by ``factor`` in text and constraint."""
    nums = [c for c in case.constraints if isinstance(c, NumericConstraint)]
    if len(nums) != 1:
        return None
    c = nums[0]
    if not _disjoint_after_scaling(c, factor):
        return None
    # Locate every bound in the *original* text, then replace all spans at once, so a
    # replaced bound is never matched again ("between 10 and 100" x10 must become
    # "between 100 and 1,000", not "between 1,000 and 100").
    spans: list[tuple[int, int, str]] = []
    for value in (c.value, c.value_max):
        if value is None:
            continue
        renderings = {
            _fmt_number(value),
            f"{value:g}",
            str(int(value)) if value.is_integer() else "",
        }
        taken = {(a, b) for a, b, _ in spans}
        matches = {
            (m.start(), m.end())
            for rendered in renderings
            if rendered
            for m in re.finditer(rf"(?<![\d.,]){re.escape(rendered)}(?![\d.,])", case.text)
        } - taken
        # Exactly one occurrence: in "Fortune 500 companies with more than 500 employees"
        # the number that expresses the constraint cannot be told apart.
        if len(matches) != 1:
            return None
        match = min(matches)
        spans.append((match[0], match[1], _fmt_number(value * factor)))
    text = case.text
    for start, end, new in sorted(spans, reverse=True):
        text = text[:start] + new + text[end:]
    new_c = c.model_copy(
        update={
            "value": c.value * factor,
            "value_max": c.value_max * factor if c.value_max is not None else None,
            "description": "",
        }
    )
    constraints = [new_c if x.id == c.id else x for x in case.constraints]
    return _derive(case, text, constraints, f"numeric_x{factor:g}")


def _shift_year(d: date | None, years: int) -> date | None:
    if d is None:
        return None
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # Feb 29
        return d.replace(year=d.year + years, day=28)


def date_shift(case: TestCase, years: int = -1) -> TestCase | None:
    """Shift the constrained date window (and the years in the text) by ``years``."""
    dates = [c for c in case.constraints if isinstance(c, DateConstraint)]
    if len(dates) != 1:
        return None
    c = dates[0]
    for d in (c.start, c.end):  # Feb 29 would be clamped to Feb 28 while the text keeps 29
        if d is not None and (d.month, d.day) == (2, 29) and _shift_year(d, years).day != 29:
            return None
    years_in_window = {d.year for d in (c.start, c.end) if d}
    # One pass over the original text, so a shifted year is never shifted again
    # (2024-2025 shifted by +1 must become 2025-2026, not 2026-2026).
    text = re.sub(
        r"\b\d{4}\b",
        lambda m: str(int(m.group()) + years) if int(m.group()) in years_in_window else m.group(),
        case.text,
    )
    if text == case.text:
        return None
    new_c = c.model_copy(
        update={
            "start": _shift_year(c.start, years),
            "end": _shift_year(c.end, years),
            "description": "",
        }
    )
    constraints = [new_c if x.id == c.id else x for x in case.constraints]
    return _derive(case, text, constraints, f"date_shift_{years:+d}y")


def argument_swap(case: TestCase) -> TestCase | None:
    """Swap subject and object of a directed relation in text and constraint."""
    rels = [c for c in case.constraints if isinstance(c, RelationConstraint) and c.directed]
    if len(rels) != 1:
        return None
    c = rels[0]
    text = case.text
    if c.subject not in text or c.object not in text:
        return None
    if c.subject in c.object or c.object in c.subject:  # "York" vs "New York"
        return None
    placeholder = "\x00"
    text = text.replace(c.subject, placeholder).replace(c.object, c.subject)
    text = text.replace(placeholder, c.object)
    new_c = c.model_copy(update={"subject": c.object, "object": c.subject, "description": ""})
    constraints = [new_c if x.id == c.id else x for x in case.constraints]
    return _derive(case, text, constraints, "argument_swap")


def antonym_flip(case: TestCase) -> TestCase | None:
    """Flip the first polarity word that has a known antonym."""
    lowered = case.text.lower()
    for word in sorted(ANTONYMS, key=len, reverse=True):
        m = re.search(rf"\b{re.escape(word)}\b", lowered)
        if m:
            replacement = ANTONYMS[word]
            original = case.text[m.start() : m.end()]
            if original[:1].isupper():
                replacement = replacement[:1].upper() + replacement[1:]
            text = case.text[: m.start()] + replacement + case.text[m.end() :]
            # Polarity changes what counts as correct; keep only non-topical
            # constraints and state the flipped requirement explicitly.
            from searchprobe.cases.models import FreeformConstraint

            constraints = [
                FreeformConstraint(
                    id="c1", requirement=f"The result directly satisfies the query: '{text}'"
                )
            ]
            return _derive(case, text, constraints, "antonym_flip")
    return None


TWIN_FUNCTIONS: dict[str, TwinFn] = {
    "negation_flip": negation_flip,
    "numeric_scale": numeric_scale,
    "date_shift": date_shift,
    "argument_swap": argument_swap,
    "antonym_flip": antonym_flip,
}

# Which transforms to try for each operator, in preference order.
OPERATOR_TWINS: dict[str, list[str]] = {
    "negation": ["negation_flip"],
    "boolean": ["negation_flip"],
    "numeric": ["numeric_scale"],
    "temporal": ["date_shift"],
    "relation": ["argument_swap"],
    "polarity": ["antonym_flip"],
    "conjunction": ["numeric_scale", "date_shift", "negation_flip"],
}


def make_twins(case: TestCase) -> list[TestCase]:
    """Generate every applicable twin for a case (deduplicated by text)."""
    names: list[str] = []
    for op in case.operators or []:
        names.extend(n for n in OPERATOR_TWINS.get(op, []) if n not in names)
    if not names:
        names = list(TWIN_FUNCTIONS)
    twins: list[TestCase] = []
    seen = {case.text.lower()}
    for name in names:
        twin = TWIN_FUNCTIONS[name](case)
        if twin is not None and twin.text.lower() not in seen:
            seen.add(twin.text.lower())
            twins.append(twin)
    return twins


def attach_twins(cases: list[TestCase]) -> list[TestCase]:
    """Return cases plus generated twins, with ``twins`` links filled in on the bases."""
    out: list[TestCase] = []
    existing = {c.id for c in cases}
    for case in cases:
        if case.twin_of:  # never twin a twin
            out.append(case)
            continue
        twins = make_twins(case)
        links = dict(case.twins)
        for t in twins:
            links.setdefault(t.twin_relation or "twin", t.id)
        out.append(case.model_copy(update={"twins": links}))
        out.extend(t for t in twins if t.id not in existing)
    return out
