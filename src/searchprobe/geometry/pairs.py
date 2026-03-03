"""Curated adversarial embedding pairs for all 13 categories.

Each category has 5-8 pairs of queries designed to expose embedding failures.
Pairs consist of semantically opposite/different queries that should have
LOW similarity but often have HIGH similarity due to embedding limitations.
"""

from searchprobe.geometry.models import EmbeddingPair


def get_adversarial_pairs(category: str) -> list[EmbeddingPair]:
    """Get adversarial embedding pairs for a category.

    Args:
        category: Adversarial category name

    Returns:
        List of EmbeddingPair objects
    """
    pairs_map = {
        "negation": _negation_pairs(),
        "numeric_precision": _numeric_precision_pairs(),
        "temporal_constraint": _temporal_constraint_pairs(),
        "multi_constraint": _multi_constraint_pairs(),
        "polysemy": _polysemy_pairs(),
        "compositional": _compositional_pairs(),
        "antonym_confusion": _antonym_confusion_pairs(),
        "specificity_gradient": _specificity_gradient_pairs(),
        "cross_lingual": _cross_lingual_pairs(),
        "counterfactual": _counterfactual_pairs(),
        "boolean_logic": _boolean_logic_pairs(),
        "entity_disambiguation": _entity_disambiguation_pairs(),
        "instruction_following": _instruction_following_pairs(),
    }
    return pairs_map.get(category, [])


def get_baseline_pairs(category: str) -> list[EmbeddingPair]:
    """Get same-topic baseline pairs (should have moderate similarity).

    Args:
        category: Adversarial category name

    Returns:
        List of baseline EmbeddingPair objects
    """
    baselines = {
        "negation": [
            EmbeddingPair("companies in AI", "AI startups", "negation", "same_topic",
                          "Same topic, no negation"),
            EmbeddingPair("venture capital firms", "investment companies", "negation", "same_topic"),
        ],
        "numeric_precision": [
            EmbeddingPair("large companies", "big corporations", "numeric_precision", "same_topic"),
            EmbeddingPair("startup funding", "venture capital investment", "numeric_precision", "same_topic"),
        ],
        "temporal_constraint": [
            EmbeddingPair("recent news", "latest updates", "temporal_constraint", "same_topic"),
            EmbeddingPair("tech trends", "technology developments", "temporal_constraint", "same_topic"),
        ],
        "polysemy": [
            EmbeddingPair("programming languages", "coding languages", "polysemy", "same_topic"),
            EmbeddingPair("river ecosystems", "freshwater habitats", "polysemy", "same_topic"),
        ],
        "antonym_confusion": [
            EmbeddingPair("employee retention", "keeping employees", "antonym_confusion", "same_topic"),
            EmbeddingPair("code quality", "software quality", "antonym_confusion", "same_topic"),
        ],
    }
    return baselines.get(category, [
        EmbeddingPair("machine learning algorithms", "ML techniques", category, "same_topic"),
        EmbeddingPair("web development", "building websites", category, "same_topic"),
    ])


def _negation_pairs() -> list[EmbeddingPair]:
    return [
        EmbeddingPair(
            "companies in AI", "companies NOT in AI",
            "negation", "adversarial",
            "Negation should flip meaning but embeddings collapse it",
        ),
        EmbeddingPair(
            "startups that raised venture capital",
            "startups that never raised venture capital",
            "negation", "adversarial",
        ),
        EmbeddingPair(
            "programming languages that are object-oriented",
            "programming languages that are NOT object-oriented",
            "negation", "adversarial",
        ),
        EmbeddingPair(
            "foods containing gluten", "foods containing no gluten",
            "negation", "adversarial",
        ),
        EmbeddingPair(
            "countries with a coastline", "countries without a coastline",
            "negation", "adversarial",
        ),
        EmbeddingPair(
            "open source software", "proprietary software that is not open source",
            "negation", "adversarial",
        ),
    ]


def _numeric_precision_pairs() -> list[EmbeddingPair]:
    return [
        EmbeddingPair(
            "companies with 50 employees", "companies with 5000 employees",
            "numeric_precision", "adversarial",
            "50 vs 5000 should be very different but share 'employees' context",
        ),
        EmbeddingPair(
            "buildings taller than 500 meters",
            "buildings shorter than 50 meters",
            "numeric_precision", "adversarial",
        ),
        EmbeddingPair(
            "startups that raised $10 million",
            "startups that raised $10 billion",
            "numeric_precision", "adversarial",
        ),
        EmbeddingPair(
            "products priced at $10", "products priced at $10000",
            "numeric_precision", "adversarial",
        ),
        EmbeddingPair(
            "cities with 100000 people", "cities with 10 million people",
            "numeric_precision", "adversarial",
        ),
    ]


def _temporal_constraint_pairs() -> list[EmbeddingPair]:
    return [
        EmbeddingPair(
            "news from January 2024", "news from January 2020",
            "temporal_constraint", "adversarial",
            "Different years but same month share heavy semantic overlap",
        ),
        EmbeddingPair(
            "tech layoffs in 2024", "tech layoffs in 2019",
            "temporal_constraint", "adversarial",
        ),
        EmbeddingPair(
            "events from last week", "events from last year",
            "temporal_constraint", "adversarial",
        ),
        EmbeddingPair(
            "startups founded in 2023", "startups founded in 2010",
            "temporal_constraint", "adversarial",
        ),
        EmbeddingPair(
            "articles published yesterday", "articles published a decade ago",
            "temporal_constraint", "adversarial",
        ),
    ]


def _multi_constraint_pairs() -> list[EmbeddingPair]:
    return [
        EmbeddingPair(
            "Python libraries for computer vision released after 2023",
            "Python libraries for computer vision",
            "multi_constraint", "adversarial",
            "Adding constraints should narrow results but barely changes embedding",
        ),
        EmbeddingPair(
            "female CEOs of Fortune 500 healthcare companies",
            "CEOs of Fortune 500 companies",
            "multi_constraint", "adversarial",
        ),
        EmbeddingPair(
            "open-source databases in Rust with ACID",
            "databases with ACID transactions",
            "multi_constraint", "adversarial",
        ),
        EmbeddingPair(
            "remote Rust jobs paying over $200k",
            "software engineering jobs",
            "multi_constraint", "adversarial",
        ),
        EmbeddingPair(
            "vegan restaurants in Tokyo with Michelin stars open on Mondays",
            "restaurants in Tokyo",
            "multi_constraint", "adversarial",
        ),
    ]


def _polysemy_pairs() -> list[EmbeddingPair]:
    return [
        EmbeddingPair(
            "Java programming performance", "Java island tourism",
            "polysemy", "adversarial",
            "Same word 'Java' but completely different senses",
        ),
        EmbeddingPair(
            "Python snake habitat", "Python programming tutorial",
            "polysemy", "adversarial",
        ),
        EmbeddingPair(
            "Apple fruit nutrition", "Apple company stock price",
            "polysemy", "adversarial",
        ),
        EmbeddingPair(
            "Mercury planet distance from sun", "Mercury element toxicity",
            "polysemy", "adversarial",
        ),
        EmbeddingPair(
            "crane bird migration", "crane construction equipment",
            "polysemy", "adversarial",
        ),
        EmbeddingPair(
            "bank of the river erosion", "bank financial services",
            "polysemy", "adversarial",
        ),
    ]


def _compositional_pairs() -> list[EmbeddingPair]:
    return [
        EmbeddingPair(
            "companies that acquired startups",
            "startups that acquired companies",
            "compositional", "adversarial",
            "Same words, opposite compositional meaning",
        ),
        EmbeddingPair(
            "dog bites man", "man bites dog",
            "compositional", "adversarial",
        ),
        EmbeddingPair(
            "teachers who became students",
            "students who became teachers",
            "compositional", "adversarial",
        ),
        EmbeddingPair(
            "the cause of the effect",
            "the effect of the cause",
            "compositional", "adversarial",
        ),
        EmbeddingPair(
            "parents learning from children",
            "children learning from parents",
            "compositional", "adversarial",
        ),
    ]


def _antonym_confusion_pairs() -> list[EmbeddingPair]:
    return [
        EmbeddingPair(
            "strategies to increase employee retention",
            "strategies to decrease employee retention",
            "antonym_confusion", "adversarial",
            "Antonyms (increase/decrease) are distributional neighbors",
        ),
        EmbeddingPair(
            "methods to accelerate development",
            "methods to slow down development",
            "antonym_confusion", "adversarial",
        ),
        EmbeddingPair(
            "ways to simplify complexity",
            "ways to add complexity",
            "antonym_confusion", "adversarial",
        ),
        EmbeddingPair(
            "companies that succeeded", "companies that failed",
            "antonym_confusion", "adversarial",
        ),
        EmbeddingPair(
            "technologies that improve security",
            "technologies that weaken security",
            "antonym_confusion", "adversarial",
        ),
        EmbeddingPair(
            "reasons to buy stocks", "reasons to sell stocks",
            "antonym_confusion", "adversarial",
        ),
    ]


def _specificity_gradient_pairs() -> list[EmbeddingPair]:
    return [
        EmbeddingPair(
            "machine learning", "the specific implementation of dropout regularization in ResNet-50 layer 3",
            "specificity_gradient", "adversarial",
            "Very different specificity levels",
        ),
        EmbeddingPair(
            "web development", "the exact CSS grid template used in Stripe's pricing page header",
            "specificity_gradient", "adversarial",
        ),
        EmbeddingPair(
            "databases", "the B+ tree rebalancing algorithm in PostgreSQL 16's btree AM",
            "specificity_gradient", "adversarial",
        ),
        EmbeddingPair(
            "artificial intelligence", "the attention head pruning strategy in GPT-4's 73rd transformer layer",
            "specificity_gradient", "adversarial",
        ),
        EmbeddingPair(
            "programming", "the exact memory layout of a Vec<T> in Rust 1.75 on ARM64",
            "specificity_gradient", "adversarial",
        ),
    ]


def _cross_lingual_pairs() -> list[EmbeddingPair]:
    return [
        EmbeddingPair(
            "artificial intelligence research",
            "recherche en intelligence artificielle",
            "cross_lingual", "adversarial",
            "Same meaning in English vs French",
        ),
        EmbeddingPair(
            "machine learning tutorial",
            "tutoriel d'apprentissage automatique",
            "cross_lingual", "adversarial",
        ),
        EmbeddingPair(
            "climate change effects",
            "Auswirkungen des Klimawandels",
            "cross_lingual", "adversarial",
        ),
        EmbeddingPair(
            "startup ecosystem",
            "ecosistema de startups",
            "cross_lingual", "adversarial",
        ),
        EmbeddingPair(
            "database optimization",
            "データベースの最適化",
            "cross_lingual", "adversarial",
        ),
    ]


def _counterfactual_pairs() -> list[EmbeddingPair]:
    return [
        EmbeddingPair(
            "the invention of the internet",
            "what if the internet was never invented",
            "counterfactual", "adversarial",
            "Factual vs counterfactual about same topic",
        ),
        EmbeddingPair(
            "Bitcoin's creation and growth",
            "what if Bitcoin had never been created",
            "counterfactual", "adversarial",
        ),
        EmbeddingPair(
            "history of social media",
            "a world without social media",
            "counterfactual", "adversarial",
        ),
        EmbeddingPair(
            "Apple's founding and success",
            "what if Apple had gone bankrupt in 1997",
            "counterfactual", "adversarial",
        ),
        EmbeddingPair(
            "the discovery of penicillin",
            "hypothetical timeline without antibiotics",
            "counterfactual", "adversarial",
        ),
    ]


def _boolean_logic_pairs() -> list[EmbeddingPair]:
    return [
        EmbeddingPair(
            "machine learning AND healthcare",
            "machine learning AND healthcare AND NOT radiology",
            "boolean_logic", "adversarial",
            "Adding NOT clause should change results but barely changes embedding",
        ),
        EmbeddingPair(
            "Python AND web frameworks",
            "Python AND (FastAPI OR Django) AND NOT Flask",
            "boolean_logic", "adversarial",
        ),
        EmbeddingPair(
            "renewable energy",
            "(solar OR wind) AND NOT residential AND commercial",
            "boolean_logic", "adversarial",
        ),
        EmbeddingPair(
            "database performance",
            "(PostgreSQL OR MySQL) AND performance AND NOT MongoDB",
            "boolean_logic", "adversarial",
        ),
        EmbeddingPair(
            "startup funding", "startup AND (acquired OR merged) AND 2024 AND NOT SPAC",
            "boolean_logic", "adversarial",
        ),
    ]


def _entity_disambiguation_pairs() -> list[EmbeddingPair]:
    return [
        EmbeddingPair(
            "Michael Jordan basketball career",
            "Michael Jordan professor machine learning Berkeley",
            "entity_disambiguation", "adversarial",
            "Same name, completely different entities",
        ),
        EmbeddingPair(
            "Paris France tourism", "Paris Texas community",
            "entity_disambiguation", "adversarial",
        ),
        EmbeddingPair(
            "Amazon online shopping", "Amazon river ecosystem",
            "entity_disambiguation", "adversarial",
        ),
        EmbeddingPair(
            "Apple iPhone features", "apple fruit varieties",
            "entity_disambiguation", "adversarial",
        ),
        EmbeddingPair(
            "Cambridge University UK", "Cambridge Massachusetts MIT",
            "entity_disambiguation", "adversarial",
        ),
        EmbeddingPair(
            "Jordan country Middle East", "Jordan basketball shoes Nike",
            "entity_disambiguation", "adversarial",
        ),
    ]


def _instruction_following_pairs() -> list[EmbeddingPair]:
    return [
        EmbeddingPair(
            "transformer architecture improvements",
            "academic papers only: transformer architecture improvements",
            "instruction_following", "adversarial",
            "Meta-instruction should change result type but barely changes embedding",
        ),
        EmbeddingPair(
            "Kubernetes deployment",
            "find blog posts, not documentation, about Kubernetes deployment",
            "instruction_following", "adversarial",
        ),
        EmbeddingPair(
            "learning Rust programming",
            "personal experiences with learning Rust, not tutorials",
            "instruction_following", "adversarial",
        ),
        EmbeddingPair(
            "OpenAI news", "news articles only about OpenAI, no blog posts",
            "instruction_following", "adversarial",
        ),
        EmbeddingPair(
            "climate change research",
            "peer-reviewed studies only about climate change, no opinion pieces",
            "instruction_following", "adversarial",
        ),
    ]


def get_random_pairs() -> list[EmbeddingPair]:
    """Get random semantically-unrelated pairs as a baseline."""
    return [
        EmbeddingPair("quantum computing algorithms", "Italian pasta recipes", "random", "random"),
        EmbeddingPair("medieval castle architecture", "Python web frameworks", "random", "random"),
        EmbeddingPair("deep sea marine biology", "real estate investment", "random", "random"),
        EmbeddingPair("ancient Egyptian hieroglyphics", "smartphone battery technology", "random", "random"),
        EmbeddingPair("jazz music improvisation", "compiler optimization techniques", "random", "random"),
        EmbeddingPair("volcanic eruption prediction", "fashion design trends", "random", "random"),
        EmbeddingPair("ballet dance training", "database sharding strategies", "random", "random"),
        EmbeddingPair("beekeeping for beginners", "spacecraft navigation systems", "random", "random"),
    ]
