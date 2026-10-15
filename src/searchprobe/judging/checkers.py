"""Deterministic constraint checks.

These run before the LLM judge and decide constraints that are mechanical:
the page's language, its publication date, and membership of the page's
subject in a case's known answer set or negative set (with known attribute
values for numeric constraints). Anything they cannot decide is left for the
LLM; they never guess.
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit

from searchprobe.cases.models import (
    DateConstraint,
    ExcludeConstraint,
    KnownEntity,
    LanguageConstraint,
    NumericConstraint,
    TestCase,
)
from searchprobe.evidence.store import Snapshot
from searchprobe.judging.models import ConstraintVerdict

LANGUAGE_CONFIDENCE_MIN = 0.6

_WIKI_RE = re.compile(r"^(?:[a-z\-]+\.)?(?:m\.)?wikipedia\.org$")
_NON_ARTICLE_NAMESPACES = {
    "Special",
    "Talk",
    "Category",
    "File",
    "Help",
    "Wikipedia",
    "Template",
    "Portal",
    "User",
}


def wikipedia_title(url: str) -> tuple[str, str] | None:
    """('en', 'Paris, Texas') for a Wikipedia article URL, else None."""
    parts = urlsplit(url)
    host = parts.netloc.lower()
    if not _WIKI_RE.match(host) or not parts.path.startswith("/wiki/"):
        return None
    title = unquote(parts.path[len("/wiki/") :]).replace("_", " ").split("#")[0]
    namespace = title.split(":", 1)[0] if ":" in title else None
    if not title or namespace in _NON_ARTICLE_NAMESPACES:
        return None
    lang = host.split(".")[0] if host.count(".") >= 2 else "en"
    return lang, title


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("_", " ").split())


def match_entity(url: str, entities: list[KnownEntity]) -> KnownEntity | None:
    """Return the entity a Wikipedia URL is about, if it is in ``entities``."""
    wiki = wikipedia_title(url)
    if wiki is None:
        return None
    _, title = wiki
    title_n = _norm(title)
    for entity in entities:
        if entity.wikipedia_url:
            ewiki = wikipedia_title(entity.wikipedia_url)
            if ewiki and _norm(ewiki[1]) == title_n:
                return entity
        if title_n == _norm(entity.label) or title_n in {_norm(a) for a in entity.aliases}:
            return entity
    return None


def run_checkers(case: TestCase, url: str, snapshot: Snapshot | None) -> list[ConstraintVerdict]:
    """Return verdicts for every constraint that can be decided mechanically."""
    verdicts: dict[str, ConstraintVerdict] = {}

    # 1. Membership in the known answer / negative sets (Wikipedia articles).
    negative = match_entity(url, case.negative_set)
    if negative is not None:
        targets = negative.violates or [
            c.id for c in case.constraints if isinstance(c, ExcludeConstraint)
        ]
        for cid in targets:
            verdicts[cid] = ConstraintVerdict(
                constraint_id=cid,
                verdict="violated",
                source="checker",
                evidence=f"Page is about {negative.label} ({negative.qid or 'known negative'})",
            )
    positive = match_entity(url, case.answer_set)
    if positive is not None and negative is None:
        for c in case.hard_constraints:
            if isinstance(c, NumericConstraint) and c.attribute in positive.values:
                try:
                    value = float(positive.values[c.attribute])
                except (TypeError, ValueError):
                    continue
                verdicts[c.id] = ConstraintVerdict(
                    constraint_id=c.id,
                    verdict="satisfied" if c.check(value) else "violated",
                    source="checker",
                    evidence=f"{positive.label}: {c.attribute} = {value:g} (Wikidata)",
                )
            elif c.id not in verdicts and c.type in ("topic", "exclude", "entity", "freeform"):
                verdicts[c.id] = ConstraintVerdict(
                    constraint_id=c.id,
                    verdict="satisfied",
                    source="checker",
                    evidence=f"Page is about {positive.label}, a known correct answer",
                )

    # 2. Page-level properties from the snapshot.
    if snapshot is not None and snapshot.ok:
        for c in case.constraints:
            if c.id in verdicts:
                continue
            if isinstance(c, LanguageConstraint) and snapshot.language:
                if snapshot.language_confidence >= LANGUAGE_CONFIDENCE_MIN:
                    verdicts[c.id] = ConstraintVerdict(
                        constraint_id=c.id,
                        verdict="satisfied" if snapshot.language == c.code else "violated",
                        source="checker",
                        evidence=f"Detected language {snapshot.language} "
                        f"(confidence {snapshot.language_confidence:.2f})",
                    )
            elif (
                isinstance(c, DateConstraint)
                and c.field == "published"
                and snapshot.published_date is not None
            ):
                verdicts[c.id] = ConstraintVerdict(
                    constraint_id=c.id,
                    verdict="satisfied" if c.contains(snapshot.published_date) else "violated",
                    source="checker",
                    evidence=f"Published {snapshot.published_date.isoformat()}",
                )
    return list(verdicts.values())
