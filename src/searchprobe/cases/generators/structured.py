"""Structured-template cases whose key constraint is mechanically checkable.

- Cross-lingual: "an introduction to photosynthesis in German" -> the
  LanguageConstraint is decided by the language detector, not an LLM.
- Publication window: "news about X published in March 2025" -> the
  DateConstraint(field="published") is decided from the page's extracted
  publication date.
- Document type: "academic papers only: X" -> DocTypeConstraint (LLM-judged,
  but typed and twinned with the unrestricted query).
"""

from __future__ import annotations

import calendar
from datetime import UTC, date, datetime

from searchprobe.cases.models import (
    DateConstraint,
    DocTypeConstraint,
    LanguageConstraint,
    Provenance,
    TestCase,
    TopicConstraint,
)

GENERATOR_VERSION = "1"

TOPICS = [
    "photosynthesis",
    "black holes",
    "machine learning",
    "the French Revolution",
    "inflation in economics",
    "how vaccines work",
    "plate tectonics",
    "blockchain technology",
    "climate change",
    "the Roman Empire",
]

LANGUAGES = {
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ru": "Russian",
}

NEWS_TOPICS = [
    "the James Webb Space Telescope",
    "electric vehicle sales",
    "central bank interest rate decisions",
    "semiconductor export controls",
    "large language model releases",
]

DOC_TYPES = {
    "academic papers only": (
        ["peer-reviewed academic paper", "preprint"],
        ["news article", "blog post", "marketing page"],
    ),
    "personal blog posts, not documentation": (
        ["personal blog post"],
        ["official documentation", "tutorial from a vendor"],
    ),
    "official documentation only": (
        ["official documentation"],
        ["blog post", "forum thread", "news article"],
    ),
}
DOC_TYPE_TOPICS = [
    "Kubernetes autoscaling",
    "transformer architectures",
    "Rust ownership",
    "PostgreSQL indexing",
    "React server components",
]


def _prov(source: str) -> Provenance:
    return Provenance(
        generator="structured",
        version=GENERATOR_VERSION,
        source=source,
        created=datetime.now(UTC).date().isoformat(),
    )


def cross_lingual_cases() -> list[TestCase]:
    cases = []
    for topic in TOPICS:
        for code, name in LANGUAGES.items():
            cases.append(
                TestCase(
                    text=f"An introduction to {topic} in {name}",
                    category="cross_lingual",
                    operators=["language"],
                    constraints=[
                        TopicConstraint(id="c1", subject=topic),
                        LanguageConstraint(id="c2", code=code),
                    ],
                    provenance=_prov("template: An introduction to {topic} in {language}"),
                    validation="auto",
                    adversarial_reason="Multilingual embeddings match the topic and ignore "
                    "the requested language",
                )
            )
    return cases


def _month_window(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def publication_window_cases(year: int = 2025, months: tuple[int, ...] = (3, 9)) -> list[TestCase]:
    cases = []
    for topic in NEWS_TOPICS:
        for month in months:
            start, end = _month_window(year, month)
            month_name = calendar.month_name[month]

            def make(
                y: int, s: date, e: date, topic: str = topic, month_name: str = month_name
            ) -> TestCase:
                return TestCase(
                    text=f"News articles about {topic} published in {month_name} {y}",
                    category="temporal_constraint",
                    operators=["temporal"],
                    constraints=[
                        TopicConstraint(id="c1", subject=f"news about {topic}"),
                        DateConstraint(id="c2", field="published", start=s, end=e),
                    ],
                    provenance=_prov(
                        "template: News articles about {topic} published in {month} {year}"
                    ),
                    validation="auto",
                    adversarial_reason="Month/year tokens barely move the query embedding",
                )

            base = make(year, start, end)
            ps, pe = _month_window(year - 1, month)
            twin = make(year - 1, ps, pe)
            twin = twin.model_copy(update={"twin_of": base.id, "twin_relation": "date_shift_-1y"})
            base = base.model_copy(update={"twins": {"date_shift_-1y": twin.id}})
            cases += [base, twin]
    return cases


def doc_type_cases() -> list[TestCase]:
    cases = []
    for topic in DOC_TYPE_TOPICS:
        for phrase, (allowed, disallowed) in DOC_TYPES.items():
            base = TestCase(
                text=f"{topic}: {phrase}",
                category="instruction_following",
                operators=["doc_type"],
                constraints=[
                    TopicConstraint(id="c1", subject=topic),
                    DocTypeConstraint(id="c2", allowed=allowed, disallowed=disallowed),
                ],
                provenance=_prov("template: {topic}: {instruction}"),
                validation="auto",
                adversarial_reason="Meta-instructions about document type are weak signals "
                "next to the topic",
            )
            cases.append(base)
    return cases


def generate_structured() -> list[TestCase]:
    return cross_lingual_cases() + publication_window_cases() + doc_type_cases()
