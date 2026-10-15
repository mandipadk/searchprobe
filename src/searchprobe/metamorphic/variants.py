"""Query variants for metamorphic relations.

DIR (directional) variants change meaning: they come from a case's twins
(negation removed, number scaled, year shifted, arguments swapped). A good
engine should return *different* results for them.

INV (invariance) variants preserve meaning: a typo, different casing, or an
LLM paraphrase that keeps every constraint. A good engine should return
*similar* results for them.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Literal

from searchprobe.cases.models import TestCase
from searchprobe.llm import structured_call

RelationKind = Literal["DIR", "INV", "REPEAT"]


@dataclass(frozen=True)
class Variant:
    """A query derived from a base case under a metamorphic relation."""

    case_id: str
    kind: RelationKind
    relation: str  # e.g. "negation_flip", "typo", "paraphrase", "repeat"
    text: str
    twin_case_id: str | None = None


def typo_variant(text: str, rng: random.Random) -> str | None:
    """Swap two adjacent interior letters of one long word (a realistic typo)."""
    words = text.split()
    candidates = [i for i, w in enumerate(words) if len(w) >= 6 and w.isalpha()]
    if not candidates:
        return None
    i = rng.choice(candidates)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    return " ".join(words)


def casing_variant(text: str) -> str | None:
    lowered = text.lower()
    return lowered if lowered != text else None


PARAPHRASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"paraphrases": {"type": "array", "items": {"type": "string"}}},
    "required": ["paraphrases"],
    "additionalProperties": False,
}

PARAPHRASE_SYSTEM = (
    "Rewrite search queries without changing their meaning. Every constraint (negations, "
    "numbers, dates, entities, document types, languages) must be preserved exactly; change "
    "only wording and word order. Never add or drop a requirement."
)


async def llm_paraphrases(client: Any, model: str, case: TestCase, n: int = 2) -> list[str]:
    """Meaning-preserving paraphrases of a case's query."""
    constraints = "\n".join(f"- {c.describe()}" for c in case.constraints)
    prompt = (
        f'Query: "{case.text}"\nConstraints that must be preserved:\n{constraints}\n\n'
        f"Give {n} paraphrases."
    )
    result = await structured_call(
        client,
        model=model,
        system=PARAPHRASE_SYSTEM,
        prompt=prompt,
        schema=PARAPHRASE_SCHEMA,
        effort="low",
    )
    seen = {case.text.lower()}
    out = []
    for p in result.data.get("paraphrases", []):
        p = re.sub(r"\s+", " ", p).strip().strip('"')
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out[:n]


def build_variants(
    cases: list[TestCase],
    repeats: int = 1,
    typos: bool = True,
    seed: int = 0,
) -> list[Variant]:
    """Deterministic variants for every base case that has at least one twin."""
    rng = random.Random(seed)
    by_id = {c.id: c for c in cases}
    variants: list[Variant] = []
    for case in cases:
        if case.twin_of:
            continue
        twins = [(rel, by_id[tid]) for rel, tid in case.twins.items() if tid in by_id]
        if not twins:
            continue
        for _ in range(repeats):
            variants.append(Variant(case.id, "REPEAT", "repeat", case.text))
        for rel, twin in twins:
            variants.append(Variant(case.id, "DIR", rel, twin.text, twin_case_id=twin.id))
        if typos:
            t = typo_variant(case.text, rng)
            if t:
                variants.append(Variant(case.id, "INV", "typo", t))
    return variants
