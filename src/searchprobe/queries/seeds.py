"""Human-curated seed queries for calibration and baseline."""

import json
from pathlib import Path
from typing import Any

from searchprobe.queries.models import GroundTruth, Query
from searchprobe.queries.taxonomy import AdversarialCategory

# Path to seed data files
SEEDS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "seeds"


def load_seeds_from_file(filepath: Path) -> list[Query]:
    """Load seed queries from a JSON file."""
    if not filepath.exists():
        return []

    with open(filepath) as f:
        data = json.load(f)

    queries = []
    for item in data:
        ground_truth = None
        if "ground_truth" in item and item["ground_truth"]:
            ground_truth = GroundTruth(**item["ground_truth"])

        queries.append(
            Query(
                text=item["text"],
                category=AdversarialCategory(item["category"]),
                difficulty=item.get("difficulty", "medium"),
                tier="seed",
                ground_truth=ground_truth,
                adversarial_reason=item.get("adversarial_reason"),
                metadata=item.get("metadata", {}),
            )
        )

    return queries


def load_all_seeds() -> list[Query]:
    """Load all seed queries from the data/seeds directory."""
    all_queries = []

    if SEEDS_DIR.exists():
        for filepath in SEEDS_DIR.glob("*.json"):
            all_queries.extend(load_seeds_from_file(filepath))

    # If no files exist, return built-in seeds
    if not all_queries:
        all_queries = get_builtin_seeds()

    return all_queries


def load_seeds_for_category(category: AdversarialCategory) -> list[Query]:
    """Load seed queries for a specific category."""
    filepath = SEEDS_DIR / f"{category.value}.json"
    if filepath.exists():
        return load_seeds_from_file(filepath)

    # Fall back to built-in seeds for this category
    return [q for q in get_builtin_seeds() if q.category == category]


def get_builtin_seeds() -> list[Query]:
    """Return built-in seed queries when no external files exist.

    These are carefully curated to test specific embedding weaknesses.
    """
    seeds = [
        # NEGATION
        Query(
            text="Startups that have never raised venture capital",
            category=AdversarialCategory.NEGATION,
            difficulty="medium",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="must_not_contain",
                must_not_contain_keywords=["Series A", "Series B", "raised", "funding round", "venture capital"],
                notes="Results should be bootstrapped or self-funded companies",
            ),
            adversarial_reason="Embeddings collapse 'never raised' to 'raised' due to semantic similarity",
        ),
        Query(
            text="Programming languages that are NOT object-oriented",
            category=AdversarialCategory.NEGATION,
            difficulty="easy",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="must_not_contain",
                must_not_contain_keywords=["Java", "C++", "Python", "Ruby", "C#"],
                must_contain_keywords=["Haskell", "Erlang", "C", "Rust", "Go", "functional"],
            ),
            adversarial_reason="'NOT object-oriented' should exclude OOP languages",
        ),
        Query(
            text="Companies that are NOT in the AI industry",
            category=AdversarialCategory.NEGATION,
            difficulty="medium",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="must_not_contain",
                must_not_contain_keywords=["AI", "artificial intelligence", "machine learning", "neural network"],
            ),
            adversarial_reason="Will Bryk's canonical example - 'shirts without stripes' returns striped shirts",
        ),
        # NUMERIC_PRECISION
        Query(
            text="Companies with exactly 50 employees",
            category=AdversarialCategory.NUMERIC_PRECISION,
            difficulty="hard",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="numeric_range",
                numeric_field="employee_count",
                numeric_min=45,
                numeric_max=55,
                notes="Allow small margin of error around 50",
            ),
            adversarial_reason="Embeddings treat '50' as a semantic token, not a mathematical value",
        ),
        Query(
            text="Startups that raised exactly $10 million in Series A",
            category=AdversarialCategory.NUMERIC_PRECISION,
            difficulty="hard",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="numeric_range",
                numeric_field="funding_amount",
                numeric_min=9_000_000,
                numeric_max=11_000_000,
            ),
            adversarial_reason="Exact funding amounts require precise numeric matching",
        ),
        # TEMPORAL_CONSTRAINT
        Query(
            text="Tech layoffs announced in January 2025",
            category=AdversarialCategory.TEMPORAL_CONSTRAINT,
            difficulty="medium",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="date_range",
                date_min="2025-01-01T00:00:00",
                date_max="2025-01-31T23:59:59",
            ),
            adversarial_reason="'January 2025' and 'January 2024' have similar embeddings",
        ),
        Query(
            text="Research papers submitted to NeurIPS 2024",
            category=AdversarialCategory.TEMPORAL_CONSTRAINT,
            difficulty="medium",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="must_contain",
                must_contain_keywords=["NeurIPS", "2024"],
            ),
            adversarial_reason="Specific conference + year combination",
        ),
        # MULTI_CONSTRAINT
        Query(
            text="Female CEOs of Fortune 500 companies in the healthcare sector",
            category=AdversarialCategory.MULTI_CONSTRAINT,
            difficulty="hard",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="manual",
                manual_judgment="Verify: (1) CEO is female, (2) company is Fortune 500, (3) company is in healthcare",
            ),
            adversarial_reason="Three constraints that ALL must be satisfied",
        ),
        Query(
            text="Open-source databases written in Rust that support ACID transactions",
            category=AdversarialCategory.MULTI_CONSTRAINT,
            difficulty="hard",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="must_contain",
                must_contain_keywords=["Rust", "ACID"],
                notes="Verify open-source license and database category",
            ),
            adversarial_reason="Four constraints: open-source, database, Rust, ACID",
        ),
        # POLYSEMY
        Query(
            text="Mercury contamination in rivers",
            category=AdversarialCategory.POLYSEMY,
            difficulty="medium",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="must_contain",
                must_contain_keywords=["chemical", "toxic", "pollution", "Hg", "metal"],
                must_not_contain_keywords=["planet", "NASA", "spacecraft"],
            ),
            adversarial_reason="'Mercury' could be planet, element, or brand - context should disambiguate",
        ),
        Query(
            text="Java performance optimization for enterprise applications",
            category=AdversarialCategory.POLYSEMY,
            difficulty="easy",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="must_contain",
                must_contain_keywords=["JVM", "Java", "programming", "code"],
                must_not_contain_keywords=["Indonesia", "coffee", "island"],
            ),
            adversarial_reason="'Java' is language, island, and coffee - tech context needed",
        ),
        # ANTONYM_CONFUSION
        Query(
            text="Strategies to decrease employee turnover",
            category=AdversarialCategory.ANTONYM_CONFUSION,
            difficulty="medium",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="must_contain",
                must_contain_keywords=["retention", "reduce", "lower", "decrease", "prevent"],
                must_not_contain_keywords=["increase turnover", "high turnover rate"],
            ),
            adversarial_reason="'Decrease' and 'increase' are embedding neighbors",
        ),
        Query(
            text="Companies that failed despite high revenue",
            category=AdversarialCategory.ANTONYM_CONFUSION,
            difficulty="medium",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="must_contain",
                must_contain_keywords=["failed", "bankrupt", "collapsed", "shutdown"],
            ),
            adversarial_reason="'Failed' and 'succeeded' often appear in similar contexts",
        ),
        # ENTITY_DISAMBIGUATION
        Query(
            text="Michael Jordan the professor at Berkeley",
            category=AdversarialCategory.ENTITY_DISAMBIGUATION,
            difficulty="medium",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="entity_match",
                expected_entity="Michael I. Jordan (professor)",
                wrong_entities=["Michael Jordan (basketball)", "Michael B. Jordan (actor)"],
            ),
            adversarial_reason="Famous basketball player dominates 'Michael Jordan' embeddings",
        ),
        Query(
            text="Paris the city in Texas",
            category=AdversarialCategory.ENTITY_DISAMBIGUATION,
            difficulty="easy",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="entity_match",
                expected_entity="Paris, Texas",
                wrong_entities=["Paris, France"],
                must_contain_keywords=["Texas", "TX"],
            ),
            adversarial_reason="Paris, France is the dominant 'Paris' in embedding space",
        ),
        # BOOLEAN_LOGIC
        Query(
            text="Machine learning frameworks that support PyTorch but NOT TensorFlow",
            category=AdversarialCategory.BOOLEAN_LOGIC,
            difficulty="hard",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="must_not_contain",
                must_contain_keywords=["PyTorch"],
                must_not_contain_keywords=["TensorFlow", "tf."],
            ),
            adversarial_reason="Boolean NOT is lost in embedding averaging",
        ),
        # INSTRUCTION_FOLLOWING
        Query(
            text="Academic papers only: transformer architecture improvements",
            category=AdversarialCategory.INSTRUCTION_FOLLOWING,
            difficulty="medium",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="type_match",
                expected_types=["academic paper", "research paper", "arxiv", "conference paper"],
            ),
            adversarial_reason="'Academic papers only' is a meta-instruction embeddings may ignore",
        ),
        Query(
            text="Personal blog posts about learning Kubernetes, not official documentation",
            category=AdversarialCategory.INSTRUCTION_FOLLOWING,
            difficulty="medium",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="type_match",
                expected_types=["blog", "personal", "experience", "learning"],
                must_not_contain_keywords=["docs", "documentation", "kubernetes.io"],
            ),
            adversarial_reason="Distinguishing blog posts from docs requires type understanding",
        ),
        # COUNTERFACTUAL
        Query(
            text="What if Bitcoin had never been invented - economic analysis",
            category=AdversarialCategory.COUNTERFACTUAL,
            difficulty="hard",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="must_contain",
                must_contain_keywords=["hypothetical", "alternative", "if", "would"],
                notes="Should return counterfactual analysis, not Bitcoin history",
            ),
            adversarial_reason="Embeddings match factual Bitcoin content instead of counterfactuals",
        ),
        # COMPOSITIONAL
        Query(
            text="Companies that were acquired by their former subsidiaries",
            category=AdversarialCategory.COMPOSITIONAL,
            difficulty="hard",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="manual",
                manual_judgment="Verify the acquirer was previously a subsidiary of the acquired",
            ),
            adversarial_reason="Word order matters: subsidiary->parent acquisition is specific",
        ),
        # SPECIFICITY_GRADIENT
        Query(
            text="The exact RGB color value used in Stripe's primary brand color",
            category=AdversarialCategory.SPECIFICITY_GRADIENT,
            difficulty="hard",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="must_contain",
                must_contain_keywords=["#", "RGB", "color", "635BFF"],
                notes="Stripe's primary purple is #635BFF",
            ),
            adversarial_reason="Extremely specific query testing embedding resolution limits",
        ),
        # CROSS_LINGUAL
        Query(
            text="Machine learning tutorials auf Deutsch",
            category=AdversarialCategory.CROSS_LINGUAL,
            difficulty="medium",
            tier="seed",
            ground_truth=GroundTruth(
                strategy="language_match",
                expected_language="de",
            ),
            adversarial_reason="Mixed English/German query should return German results",
        ),
    ]

    return seeds


def save_seeds_to_file(queries: list[Query], filepath: Path) -> None:
    """Save queries to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for q in queries:
        item: dict[str, Any] = {
            "text": q.text,
            "category": q.category.value,
            "difficulty": q.difficulty,
            "adversarial_reason": q.adversarial_reason,
        }
        if q.ground_truth:
            item["ground_truth"] = q.ground_truth.model_dump(exclude_none=True)
        if q.metadata:
            item["metadata"] = q.metadata
        data.append(item)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def export_builtin_seeds() -> None:
    """Export built-in seeds to individual category files."""
    seeds = get_builtin_seeds()

    # Group by category
    by_category: dict[str, list[Query]] = {}
    for q in seeds:
        cat = q.category.value
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(q)

    # Save each category
    for category, queries in by_category.items():
        filepath = SEEDS_DIR / f"{category}.json"
        save_seeds_to_file(queries, filepath)
        print(f"Saved {len(queries)} queries to {filepath}")
