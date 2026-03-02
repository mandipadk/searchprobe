"""Template-based query generation for systematic coverage."""

import itertools
import random
from typing import Any

from pydantic import BaseModel

from searchprobe.queries.models import GroundTruth, Query
from searchprobe.queries.taxonomy import AdversarialCategory


class QueryTemplate(BaseModel):
    """A parameterized template for generating queries."""

    template: str
    category: AdversarialCategory
    difficulty: str = "medium"
    slots: dict[str, list[str]]  # slot_name -> possible values
    ground_truth_template: GroundTruth | None = None
    adversarial_reason: str | None = None
    max_combinations: int = 10  # Limit combinations per template


# Template definitions per category
TEMPLATES: dict[AdversarialCategory, list[QueryTemplate]] = {
    AdversarialCategory.NEGATION: [
        QueryTemplate(
            template="{entity_type} that are NOT {attribute}",
            category=AdversarialCategory.NEGATION,
            slots={
                "entity_type": ["companies", "startups", "organizations", "firms"],
                "attribute": [
                    "profitable",
                    "venture-backed",
                    "publicly traded",
                    "in the tech industry",
                    "based in Silicon Valley",
                ],
            },
            ground_truth_template=GroundTruth(
                strategy="must_not_contain", must_not_contain_keywords=[]
            ),
            adversarial_reason="Negation 'NOT {attribute}' collapses to '{attribute}' in embeddings",
        ),
        QueryTemplate(
            template="{entity_type} without {feature}",
            category=AdversarialCategory.NEGATION,
            slots={
                "entity_type": ["products", "services", "apps", "tools", "platforms"],
                "feature": [
                    "subscription fees",
                    "ads",
                    "user tracking",
                    "login requirements",
                    "paywalls",
                ],
            },
            adversarial_reason="'Without' is a weak negation signal",
        ),
        QueryTemplate(
            template="{thing} that have never {action}",
            category=AdversarialCategory.NEGATION,
            difficulty="hard",
            slots={
                "thing": ["companies", "countries", "athletes", "musicians"],
                "action": [
                    "gone bankrupt",
                    "won a championship",
                    "had a #1 hit",
                    "hosted the Olympics",
                ],
            },
            adversarial_reason="'Never' temporal negation is especially weak in embeddings",
        ),
    ],
    AdversarialCategory.NUMERIC_PRECISION: [
        QueryTemplate(
            template="{entity_type} with exactly {number} {unit}",
            category=AdversarialCategory.NUMERIC_PRECISION,
            difficulty="hard",
            slots={
                "entity_type": ["companies", "startups", "organizations"],
                "number": ["10", "25", "50", "100", "250", "500"],
                "unit": ["employees", "team members", "staff"],
            },
            adversarial_reason="Exact numbers are semantic tokens, not mathematical values",
        ),
        QueryTemplate(
            template="{entity_type} valued at exactly ${amount}",
            category=AdversarialCategory.NUMERIC_PRECISION,
            difficulty="hard",
            slots={
                "entity_type": ["startups", "companies", "unicorns"],
                "amount": [
                    "1 billion",
                    "500 million",
                    "100 million",
                    "10 billion",
                ],
            },
            adversarial_reason="Dollar amounts require precise numeric understanding",
        ),
        QueryTemplate(
            template="{things} between {min_val} and {max_val} {unit}",
            category=AdversarialCategory.NUMERIC_PRECISION,
            difficulty="hard",
            slots={
                "things": ["buildings", "mountains", "structures"],
                "min_val": ["100", "200", "500"],
                "max_val": ["150", "300", "600"],
                "unit": ["meters tall", "feet high", "stories"],
            },
            adversarial_reason="Range constraints require numeric comparison",
        ),
    ],
    AdversarialCategory.TEMPORAL_CONSTRAINT: [
        QueryTemplate(
            template="{event_type} in {time_period}",
            category=AdversarialCategory.TEMPORAL_CONSTRAINT,
            slots={
                "event_type": [
                    "product launches",
                    "funding announcements",
                    "acquisitions",
                    "layoffs",
                    "IPOs",
                ],
                "time_period": [
                    "Q1 2025",
                    "January 2025",
                    "the last 30 days",
                    "December 2024",
                    "the week of January 15, 2025",
                ],
            },
            adversarial_reason="Time periods have weak embedding differentiation",
        ),
        QueryTemplate(
            template="News about {topic} from {specific_date}",
            category=AdversarialCategory.TEMPORAL_CONSTRAINT,
            difficulty="hard",
            slots={
                "topic": ["AI", "startups", "tech layoffs", "cryptocurrency"],
                "specific_date": [
                    "January 15, 2025",
                    "December 1, 2024",
                    "yesterday",
                    "this morning",
                ],
            },
            adversarial_reason="Specific dates are weakly encoded",
        ),
    ],
    AdversarialCategory.MULTI_CONSTRAINT: [
        QueryTemplate(
            template="{role} at {company_type} in {location} industry:{industry}",
            category=AdversarialCategory.MULTI_CONSTRAINT,
            difficulty="hard",
            slots={
                "role": ["Female CEO", "Founder", "CTO", "VP of Engineering"],
                "company_type": ["Fortune 500 company", "startup", "unicorn"],
                "location": ["Silicon Valley", "New York", "Europe", "Asia"],
                "industry": ["healthcare", "fintech", "AI", "climate tech"],
            },
            adversarial_reason="4 constraints that ALL must be satisfied",
        ),
        QueryTemplate(
            template="{tech_type} that is {license} and supports {feature} written in {language}",
            category=AdversarialCategory.MULTI_CONSTRAINT,
            difficulty="hard",
            slots={
                "tech_type": ["database", "framework", "library", "tool"],
                "license": ["open-source", "MIT licensed", "Apache 2.0"],
                "feature": ["ACID transactions", "async/await", "GPU acceleration"],
                "language": ["Rust", "Go", "Python", "TypeScript"],
            },
            adversarial_reason="Multiple technical constraints",
        ),
    ],
    AdversarialCategory.POLYSEMY: [
        QueryTemplate(
            template="{ambiguous_word} {context_hint}",
            category=AdversarialCategory.POLYSEMY,
            slots={
                "ambiguous_word": ["Mercury", "Python", "Java", "Apple", "Amazon"],
                "context_hint": [
                    "programming tutorials",
                    "environmental impact",
                    "business strategy",
                    "chemical properties",
                    "river ecosystem",
                ],
            },
            adversarial_reason="Ambiguous words blend multiple senses in embeddings",
        ),
    ],
    AdversarialCategory.ANTONYM_CONFUSION: [
        QueryTemplate(
            template="How to {action} {metric}",
            category=AdversarialCategory.ANTONYM_CONFUSION,
            slots={
                "action": ["decrease", "reduce", "minimize", "lower"],
                "metric": [
                    "latency",
                    "costs",
                    "employee turnover",
                    "carbon footprint",
                    "technical debt",
                ],
            },
            adversarial_reason="'Decrease' and 'increase' are embedding neighbors",
        ),
        QueryTemplate(
            template="{entity} that {negative_outcome} despite {positive_factor}",
            category=AdversarialCategory.ANTONYM_CONFUSION,
            slots={
                "entity": ["companies", "products", "startups", "projects"],
                "negative_outcome": ["failed", "shut down", "lost market share"],
                "positive_factor": [
                    "high revenue",
                    "strong funding",
                    "great reviews",
                    "large user base",
                ],
            },
            adversarial_reason="Contrasting positive and negative creates embedding confusion",
        ),
    ],
    AdversarialCategory.ENTITY_DISAMBIGUATION: [
        QueryTemplate(
            template="{common_name} the {disambiguator}",
            category=AdversarialCategory.ENTITY_DISAMBIGUATION,
            slots={
                "common_name": [
                    "Michael Jordan",
                    "Paris",
                    "Cambridge",
                    "Jordan",
                    "Washington",
                ],
                "disambiguator": [
                    "professor",
                    "city in Texas",
                    "in Massachusetts",
                    "country",
                    "state",
                ],
            },
            adversarial_reason="Less famous entities with common names are overshadowed",
        ),
    ],
    AdversarialCategory.BOOLEAN_LOGIC: [
        QueryTemplate(
            template="({option1} OR {option2}) AND {required} NOT {excluded}",
            category=AdversarialCategory.BOOLEAN_LOGIC,
            difficulty="hard",
            slots={
                "option1": ["Python", "JavaScript", "Rust", "Go"],
                "option2": ["TypeScript", "Ruby", "Kotlin", "Swift"],
                "required": [
                    "web development",
                    "machine learning",
                    "backend",
                    "mobile",
                ],
                "excluded": ["tutorial", "beginner", "introduction", "basics"],
            },
            adversarial_reason="Boolean operators don't exist in embedding space",
        ),
    ],
    AdversarialCategory.INSTRUCTION_FOLLOWING: [
        QueryTemplate(
            template="{content_type} only: {topic}",
            category=AdversarialCategory.INSTRUCTION_FOLLOWING,
            slots={
                "content_type": [
                    "Academic papers",
                    "Blog posts",
                    "News articles",
                    "Video tutorials",
                    "Documentation",
                ],
                "topic": [
                    "transformer architecture",
                    "Kubernetes deployment",
                    "machine learning",
                    "system design",
                    "microservices",
                ],
            },
            adversarial_reason="Content type instructions are often ignored",
        ),
        QueryTemplate(
            template="{personal_type} about {topic}, not {excluded_type}",
            category=AdversarialCategory.INSTRUCTION_FOLLOWING,
            slots={
                "personal_type": [
                    "Personal experiences",
                    "First-hand accounts",
                    "Opinion pieces",
                ],
                "topic": [
                    "learning Rust",
                    "startup failure",
                    "career change",
                    "remote work",
                ],
                "excluded_type": [
                    "tutorials",
                    "news reports",
                    "documentation",
                    "guides",
                ],
            },
            adversarial_reason="Distinguishing content types requires meta-understanding",
        ),
    ],
}


def generate_from_template(template: QueryTemplate, slot_values: dict[str, str]) -> Query:
    """Generate a single query from a template with specific slot values."""
    # Format the template
    text = template.template.format(**slot_values)

    # Create ground truth if template has one
    ground_truth = None
    if template.ground_truth_template:
        # Copy and customize ground truth
        gt_dict = template.ground_truth_template.model_dump(exclude_none=True)

        # Replace any slot references in ground truth
        for key, value in gt_dict.items():
            if isinstance(value, str):
                gt_dict[key] = value.format(**slot_values)
            elif isinstance(value, list):
                gt_dict[key] = [
                    v.format(**slot_values) if isinstance(v, str) else v for v in value
                ]

        ground_truth = GroundTruth(**gt_dict)

    # Create adversarial reason
    adversarial_reason = template.adversarial_reason
    if adversarial_reason:
        adversarial_reason = adversarial_reason.format(**slot_values)

    return Query(
        text=text,
        category=template.category,
        difficulty=template.difficulty,
        tier="template",
        ground_truth=ground_truth,
        adversarial_reason=adversarial_reason,
        metadata={"template": template.template, "slots": slot_values},
    )


def generate_all_from_template(
    template: QueryTemplate, max_queries: int | None = None
) -> list[Query]:
    """Generate all (or limited) combinations from a template."""
    # Get all slot combinations
    slot_names = list(template.slots.keys())
    slot_values_lists = [template.slots[name] for name in slot_names]

    combinations = list(itertools.product(*slot_values_lists))

    # Limit combinations
    limit = max_queries or template.max_combinations
    if len(combinations) > limit:
        combinations = random.sample(combinations, limit)

    queries = []
    for combo in combinations:
        slot_values = dict(zip(slot_names, combo))
        queries.append(generate_from_template(template, slot_values))

    return queries


def generate_for_category(
    category: AdversarialCategory, max_queries_per_template: int = 5
) -> list[Query]:
    """Generate template-based queries for a category."""
    templates = TEMPLATES.get(category, [])
    queries = []

    for template in templates:
        queries.extend(generate_all_from_template(template, max_queries_per_template))

    return queries


def generate_all(max_per_template: int = 5) -> list[Query]:
    """Generate template-based queries for all categories."""
    all_queries = []

    for category in AdversarialCategory:
        all_queries.extend(generate_for_category(category, max_per_template))

    return all_queries
