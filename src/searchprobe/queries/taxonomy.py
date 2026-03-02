"""Adversarial query taxonomy with failure hypotheses grounded in embedding theory.

Each category targets a specific weakness of neural/embedding-based search systems.
The failure hypotheses explain WHY embeddings struggle with these query types.
"""

from enum import Enum

from pydantic import BaseModel, Field


class AdversarialCategory(str, Enum):
    """Adversarial query categories targeting specific embedding weaknesses.

    Each category is designed to probe a different failure mode of
    neural/embedding-based search systems.
    """

    NEGATION = "negation"
    NUMERIC_PRECISION = "numeric_precision"
    TEMPORAL_CONSTRAINT = "temporal_constraint"
    MULTI_CONSTRAINT = "multi_constraint"
    POLYSEMY = "polysemy"
    COMPOSITIONAL = "compositional"
    ANTONYM_CONFUSION = "antonym_confusion"
    SPECIFICITY_GRADIENT = "specificity_gradient"
    CROSS_LINGUAL = "cross_lingual"
    COUNTERFACTUAL = "counterfactual"
    BOOLEAN_LOGIC = "boolean_logic"
    ENTITY_DISAMBIGUATION = "entity_disambiguation"
    INSTRUCTION_FOLLOWING = "instruction_following"


class CategoryMetadata(BaseModel):
    """Metadata for an adversarial category."""

    category: AdversarialCategory
    display_name: str
    description: str = Field(..., description="Human-readable description")
    failure_hypothesis: str = Field(
        ..., description="Why embedding-based search fails on this category"
    )
    difficulty: str = Field(..., description="easy, medium, or hard")
    example_queries: list[str] = Field(
        default_factory=list, description="Example adversarial queries"
    )
    ground_truth_strategy: str = Field(
        ..., description="How to verify result correctness"
    )
    keyword_search_advantage: bool = Field(
        default=False, description="Whether keyword search typically outperforms here"
    )


# Category definitions with failure hypotheses
CATEGORY_METADATA: dict[AdversarialCategory, CategoryMetadata] = {
    AdversarialCategory.NEGATION: CategoryMetadata(
        category=AdversarialCategory.NEGATION,
        display_name="Negation",
        description="Queries requiring exclusion of certain concepts or entities",
        failure_hypothesis=(
            "Embeddings notoriously collapse negation. 'Companies NOT in AI' and "
            "'Companies in AI' produce nearly identical embeddings because semantic "
            "content (companies, AI) dominates the vector, while 'not' is a weak signal "
            "in embedding space. This is a fundamental limitation of distributional "
            "semantics - negation words appear in similar contexts to their positive counterparts."
        ),
        difficulty="medium",
        example_queries=[
            "Startups that have never raised venture capital",
            "Programming languages that are NOT object-oriented",
            "Countries without a coastline that have a navy",
            "Companies that are NOT in the AI industry",
            "Foods that contain no gluten and no dairy",
        ],
        ground_truth_strategy="Verify results do NOT contain the negated attribute",
        keyword_search_advantage=False,  # Keyword search also struggles with negation
    ),
    AdversarialCategory.NUMERIC_PRECISION: CategoryMetadata(
        category=AdversarialCategory.NUMERIC_PRECISION,
        display_name="Numeric Precision",
        description="Queries requiring exact numeric constraints",
        failure_hypothesis=(
            "Embeddings represent numbers as semantic tokens, not mathematical values. "
            "'12 employees' and '1200 employees' have similar embeddings because they "
            "share semantic context (employees, numbers). Exact numeric constraints "
            "require symbolic reasoning that vector similarity cannot perform. The "
            "embedding space has no concept of greater-than, less-than, or equality."
        ),
        difficulty="hard",
        example_queries=[
            "Companies with exactly 50 employees",
            "Buildings taller than 500 meters but shorter than 600 meters",
            "Countries with a GDP per capita between $15,000 and $20,000",
            "Startups that raised exactly $10 million in Series A",
            "Products priced at exactly $99.99",
        ],
        ground_truth_strategy="Extract numeric values from results and verify constraints",
        keyword_search_advantage=False,
    ),
    AdversarialCategory.TEMPORAL_CONSTRAINT: CategoryMetadata(
        category=AdversarialCategory.TEMPORAL_CONSTRAINT,
        display_name="Temporal Constraint",
        description="Queries with specific date/time requirements",
        failure_hypothesis=(
            "Embeddings encode time references weakly. 'January 2024' and 'January 2025' "
            "have very similar vector representations because they share most semantic "
            "content. Fine-grained temporal constraints (specific weeks, narrow date ranges) "
            "are especially difficult because the embedding space lacks temporal structure."
        ),
        difficulty="hard",
        example_queries=[
            "Articles published in the third week of January 2025",
            "Tech layoffs announced between March 15 and March 22, 2024",
            "Research papers submitted to NeurIPS 2024 but not accepted",
            "Startup funding rounds closed in Q3 2024",
            "Product launches from the last 72 hours",
        ],
        ground_truth_strategy="Cross-reference with known publication/event dates",
        keyword_search_advantage=False,
    ),
    AdversarialCategory.MULTI_CONSTRAINT: CategoryMetadata(
        category=AdversarialCategory.MULTI_CONSTRAINT,
        display_name="Multi-Constraint",
        description="Queries with 3+ simultaneous constraints that ALL must be satisfied",
        failure_hypothesis=(
            "Embedding similarity measures holistic semantic meaning, not constraint "
            "satisfaction. A query with 3 constraints may match results satisfying only "
            "1-2 constraints because the overall semantic similarity is high enough. "
            "Vector cosine similarity 'averages' the meaning rather than requiring "
            "conjunction of all constraints."
        ),
        difficulty="hard",
        example_queries=[
            "Python libraries for computer vision released after 2023 with more than 1000 GitHub stars",
            "Female CEOs of Fortune 500 companies in the healthcare sector",
            "Open-source databases written in Rust that support ACID transactions",
            "Series A startups in fintech with female founders based in Europe",
            "Remote software engineering jobs paying over $200k requiring Rust experience",
        ],
        ground_truth_strategy="Verify each constraint independently in results",
        keyword_search_advantage=False,
    ),
    AdversarialCategory.POLYSEMY: CategoryMetadata(
        category=AdversarialCategory.POLYSEMY,
        display_name="Polysemy / Ambiguity",
        description="Queries where key terms have multiple meanings",
        failure_hypothesis=(
            "Words with multiple meanings (bank, Java, Apple, Mercury) cause embeddings "
            "to blend all senses into a single vector. The query context may be insufficient "
            "to disambiguate, leading to results that mix different word senses. This is "
            "a consequence of embeddings being trained on word co-occurrence patterns."
        ),
        difficulty="medium",
        example_queries=[
            "Mercury contamination in rivers",  # planet vs element vs brand
            "Java performance optimization",  # language vs island vs coffee
            "Crane operations near the bay",  # bird vs construction equipment
            "Python in the Amazon",  # snake vs language vs company
            "Apple's environmental impact",  # fruit vs company
        ],
        ground_truth_strategy="Verify results match the intended word sense",
        keyword_search_advantage=False,
    ),
    AdversarialCategory.COMPOSITIONAL: CategoryMetadata(
        category=AdversarialCategory.COMPOSITIONAL,
        display_name="Compositional",
        description="Queries where word ORDER and compositional structure matter",
        failure_hypothesis=(
            "Embeddings capture bag-of-words semantics better than compositional structure. "
            "'Dog bites man' and 'Man bites dog' may have similar embeddings because they "
            "contain the same words. Queries requiring understanding of who-does-what-to-whom "
            "or temporal sequences challenge the flat structure of embedding space."
        ),
        difficulty="hard",
        example_queries=[
            "Diseases that are caused by the cure of other diseases",  # iatrogenic
            "Companies that were acquired by their former subsidiaries",
            "Athletes who became politicians and then returned to sports",
            "Students who taught their teachers",
            "Startups that pivoted from B2C to B2B and back to B2C",
        ],
        ground_truth_strategy="Verify compositional structure matches query intent",
        keyword_search_advantage=False,
    ),
    AdversarialCategory.ANTONYM_CONFUSION: CategoryMetadata(
        category=AdversarialCategory.ANTONYM_CONFUSION,
        display_name="Antonym Confusion",
        description="Queries where antonyms might be confused due to embedding similarity",
        failure_hypothesis=(
            "Embedding models often place antonyms close together in vector space because "
            "they appear in similar contexts. 'Increase' and 'decrease' are distributional "
            "neighbors despite having opposite meanings. This is the distributional "
            "semantics paradox: words are similar to their opposites because they're used "
            "in the same contexts."
        ),
        difficulty="medium",
        example_queries=[
            "Strategies to decrease employee turnover",
            "Companies that failed despite high revenue",
            "Technologies that simplify complexity rather than add it",
            "Methods to reduce code verbosity",
            "Approaches that slow down rather than accelerate development",
        ],
        ground_truth_strategy="Verify results match the correct polarity (positive/negative)",
        keyword_search_advantage=True,  # Keyword search can match exact words
    ),
    AdversarialCategory.SPECIFICITY_GRADIENT: CategoryMetadata(
        category=AdversarialCategory.SPECIFICITY_GRADIENT,
        display_name="Specificity Gradient",
        description="Testing precision at different query specificity levels",
        failure_hypothesis=(
            "Embeddings have resolution limits. Very specific queries may return overly "
            "generic results because the embedding space doesn't have enough granularity "
            "to distinguish fine-grained concepts. Conversely, very broad queries may "
            "return incoherent mixtures. This tests the 'zoom level' of the embedding space."
        ),
        difficulty="medium",
        example_queries=[
            "The third paragraph of Paul Graham's essay 'Do Things That Don't Scale'",
            "The exact implementation of the attention mechanism in GPT-4",
            "The specific RGB color value used in Stripe's logo",
            "The phone number of the first investor in Airbnb",
            "The name of the second employee hired at OpenAI",
        ],
        ground_truth_strategy="Measure precision at each specificity level",
        keyword_search_advantage=False,
    ),
    AdversarialCategory.CROSS_LINGUAL: CategoryMetadata(
        category=AdversarialCategory.CROSS_LINGUAL,
        display_name="Cross-Lingual",
        description="Queries mixing languages or seeking specific-language content",
        failure_hypothesis=(
            "Multilingual embeddings may not perfectly align semantic spaces across "
            "languages. A query in French may not retrieve relevant English documents "
            "even if the embeddings are supposedly multilingual. Mixed-language queries "
            "create ambiguous points in embedding space."
        ),
        difficulty="medium",
        example_queries=[
            "Recherche sur l'intelligence artificielle",  # French
            "Machine learning tutorials auf Deutsch",  # German
            "日本のスタートアップ ecosystem",  # Mixed Japanese/English
            "Documentación técnica en español about Kubernetes",
            "Статьи о машинном обучении on TensorFlow",  # Russian/English mix
        ],
        ground_truth_strategy="Verify results are in expected language or correctly multilingual",
        keyword_search_advantage=False,
    ),
    AdversarialCategory.COUNTERFACTUAL: CategoryMetadata(
        category=AdversarialCategory.COUNTERFACTUAL,
        display_name="Counterfactual / Hypothetical",
        description="Queries about hypothetical or alternative scenarios",
        failure_hypothesis=(
            "Embedding search optimizes for semantic similarity to existing content. "
            "Hypothetical queries like 'What if the internet was never invented' are "
            "semantically similar to factual content about the internet's invention. "
            "The search engine returns factual content instead of counterfactual analysis."
        ),
        difficulty="hard",
        example_queries=[
            "What if Bitcoin had never been invented",
            "Hypothetical scenarios where Moore's Law breaks down",
            "Alternative history where Apple acquired Google",
            "Analysis of a world without social media",
            "Speculative timeline where AI was invented in the 1800s",
        ],
        ground_truth_strategy="Verify results address the hypothetical, not the factual counterpart",
        keyword_search_advantage=False,
    ),
    AdversarialCategory.BOOLEAN_LOGIC: CategoryMetadata(
        category=AdversarialCategory.BOOLEAN_LOGIC,
        display_name="Boolean Logic",
        description="Queries with explicit AND, OR, NOT operators",
        failure_hypothesis=(
            "Embedding similarity has no native AND/OR/NOT operators. Complex boolean "
            "queries get 'averaged out' in vector space. A query with (A AND B) OR (C AND D) "
            "becomes a blurred combination in embedding space rather than a precise logical "
            "expression. Even Exa's include_text/exclude_text filters only support single strings."
        ),
        difficulty="hard",
        example_queries=[
            "(machine learning OR deep learning) AND healthcare AND NOT radiology",
            "Python AND (FastAPI OR Django) AND deployment AND NOT Heroku",
            "Renewable energy AND (solar OR wind) AND NOT residential",
            "(startup OR company) AND (acquired OR merged) AND 2024",
            "Database AND (PostgreSQL OR MySQL) AND performance AND NOT MongoDB",
        ],
        ground_truth_strategy="Verify each boolean clause independently in results",
        keyword_search_advantage=True,  # Advanced keyword search can handle boolean
    ),
    AdversarialCategory.ENTITY_DISAMBIGUATION: CategoryMetadata(
        category=AdversarialCategory.ENTITY_DISAMBIGUATION,
        display_name="Entity Disambiguation",
        description="Queries requiring distinction between same-named entities",
        failure_hypothesis=(
            "When entities share names, embeddings may conflate them. 'Michael Jordan' "
            "embeds primarily as the basketball player because that's the dominant "
            "representation in training data. Queries about less-famous same-named entities "
            "retrieve results about the famous one instead."
        ),
        difficulty="medium",
        example_queries=[
            "Michael Jordan the professor at Berkeley",
            "Paris the city in Texas",
            "Amazon the river's ecological impact",
            "Jordan the country's tech startup scene",
            "Cambridge the city in Massachusetts vs Cambridge UK",
        ],
        ground_truth_strategy="Verify results reference the correct entity",
        keyword_search_advantage=False,
    ),
    AdversarialCategory.INSTRUCTION_FOLLOWING: CategoryMetadata(
        category=AdversarialCategory.INSTRUCTION_FOLLOWING,
        display_name="Instruction Following",
        description="Queries with meta-instructions about result format or type",
        failure_hypothesis=(
            "Embedding similarity optimizes for content relevance, not meta-instruction "
            "compliance. Instructions like 'academic papers only' or 'find blog posts, not "
            "documentation' are semantic modifiers that may not strongly influence the "
            "embedding vector compared to the main query content."
        ),
        difficulty="medium",
        example_queries=[
            "Academic papers only: transformer architecture improvements",
            "Find blog posts, not documentation, about Kubernetes",
            "Personal experiences with learning Rust, not tutorials",
            "News articles only about the OpenAI drama",
            "First-hand accounts, not news reports, of the event",
        ],
        ground_truth_strategy="Verify result types match instructions",
        keyword_search_advantage=False,
    ),
}


def get_all_categories() -> list[AdversarialCategory]:
    """Return all adversarial categories."""
    return list(AdversarialCategory)


def get_category_metadata(category: AdversarialCategory) -> CategoryMetadata:
    """Get metadata for a specific category."""
    return CATEGORY_METADATA[category]


def get_categories_by_difficulty(difficulty: str) -> list[AdversarialCategory]:
    """Get categories filtered by difficulty level."""
    return [
        cat
        for cat, meta in CATEGORY_METADATA.items()
        if meta.difficulty == difficulty
    ]


def get_example_queries(category: AdversarialCategory | None = None) -> list[str]:
    """Get example queries, optionally filtered by category."""
    if category:
        return CATEGORY_METADATA[category].example_queries
    return [q for meta in CATEGORY_METADATA.values() for q in meta.example_queries]
