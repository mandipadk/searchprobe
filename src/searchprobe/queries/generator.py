"""LLM-based query generation for creative adversarial probing."""

import json
from typing import Any

from pydantic import BaseModel

from searchprobe.config import get_anthropic_client, get_settings
from searchprobe.queries.models import GroundTruth, Query, QuerySet
from searchprobe.queries.seeds import get_builtin_seeds, load_all_seeds
from searchprobe.queries.taxonomy import (
    AdversarialCategory,
    CategoryMetadata,
    get_category_metadata,
)
from searchprobe.queries.templates import generate_all as generate_all_templates


class GeneratedQueryResponse(BaseModel):
    """Response format for LLM-generated queries."""

    queries: list[dict[str, Any]]


GENERATION_PROMPT = """You are an expert in neural search systems and adversarial testing.

Your task is to generate adversarial queries that expose failure modes in embedding-based search engines like Exa.ai.

## Category: {category_name}

### Description
{category_description}

### Failure Hypothesis
{failure_hypothesis}

### Why Embeddings Fail Here
{detailed_explanation}

### Example Queries (for inspiration, DO NOT repeat these)
{example_queries}

## Instructions

Generate {count} NEW adversarial queries for this category that:
1. Are realistic queries a real user might ask
2. Exploit the specific embedding weakness described above
3. Have verifiable ground truth (we can check if results are correct)
4. Are DIFFERENT from the examples provided
5. Vary in difficulty (some medium, some hard)
6. Cover diverse topics and domains

For each query, provide:
- The query text
- Difficulty level (medium or hard)
- Why this query is adversarial (brief explanation)
- Ground truth hints (how to verify correctness)

## Response Format

Respond with a JSON object containing a "queries" array:
```json
{{
  "queries": [
    {{
      "text": "the query text",
      "difficulty": "medium" or "hard",
      "adversarial_reason": "why this challenges embeddings",
      "ground_truth_hints": "how to verify results are correct"
    }}
  ]
}}
```

Generate exactly {count} queries. Be creative but realistic.
"""


class QueryGenerator:
    """Generates adversarial queries using LLM."""

    def __init__(self, model: str | None = None, temperature: float = 0.9):
        """Initialize the generator."""
        settings = get_settings()
        self.model = model or settings.generation_model
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        """Lazy-load Anthropic client (direct API or Vertex AI)."""
        if self._client is None:
            self._client = get_anthropic_client()
        return self._client

    def generate_for_category(
        self,
        category: AdversarialCategory,
        count: int = 10,
        existing_queries: list[str] | None = None,
    ) -> list[Query]:
        """Generate adversarial queries for a specific category using LLM."""
        metadata = get_category_metadata(category)

        # Build the prompt
        example_queries = "\n".join(f"- {q}" for q in metadata.example_queries)
        if existing_queries:
            example_queries += "\n\nAlready generated (DO NOT repeat):\n"
            example_queries += "\n".join(f"- {q}" for q in existing_queries[:10])

        prompt = GENERATION_PROMPT.format(
            category_name=metadata.display_name,
            category_description=metadata.description,
            failure_hypothesis=metadata.failure_hypothesis,
            detailed_explanation=self._get_detailed_explanation(metadata),
            example_queries=example_queries,
            count=count,
        )

        # Call the LLM
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse response
        content = response.content[0].text

        try:
            from searchprobe.utils.parsing import extract_json_from_llm_response

            data = extract_json_from_llm_response(content)
        except ValueError as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Content: {content[:500]}")
            return []

        # Convert to Query objects
        queries = []
        for item in data.get("queries", []):
            ground_truth = None
            if "ground_truth_hints" in item:
                ground_truth = GroundTruth(
                    strategy="manual", manual_judgment=item["ground_truth_hints"]
                )

            queries.append(
                Query(
                    text=item["text"],
                    category=category,
                    difficulty=item.get("difficulty", "medium"),
                    tier="llm",
                    ground_truth=ground_truth,
                    adversarial_reason=item.get("adversarial_reason"),
                    metadata={"model": self.model},
                )
            )

        return queries

    def _get_detailed_explanation(self, metadata: CategoryMetadata) -> str:
        """Get detailed technical explanation of why embeddings fail."""
        explanations = {
            AdversarialCategory.NEGATION: (
                "In vector space, negation words like 'not', 'never', 'without' have "
                "relatively small influence on the overall embedding direction. The "
                "semantic content of the query dominates, causing 'X NOT Y' to embed "
                "similarly to 'X Y'. This is because word embeddings are based on "
                "distributional semantics - words appearing in similar contexts get "
                "similar vectors, and negation words appear with both positive and "
                "negative contexts."
            ),
            AdversarialCategory.NUMERIC_PRECISION: (
                "Numbers in text are tokenized as individual tokens or subword pieces. "
                "'50' and '500' might share tokens, and all numbers share semantic "
                "context (they appear with words like 'employees', 'million', etc.). "
                "The embedding space has no mathematical structure - there's no way "
                "to represent greater-than or less-than relationships. '50 employees' "
                "and '5000 employees' are nearly equidistant from '500 employees'."
            ),
            AdversarialCategory.TEMPORAL_CONSTRAINT: (
                "Dates and time expressions are encoded as semantic tokens, not temporal "
                "values. 'January 2024' and 'January 2025' share most of their semantic "
                "content - both contain 'January' and a year. The embedding space lacks "
                "temporal structure, so it cannot distinguish between 'last week' and "
                "'last year' based on actual time distance."
            ),
            AdversarialCategory.MULTI_CONSTRAINT: (
                "Cosine similarity in embedding space measures overall semantic alignment, "
                "not constraint satisfaction. When you have query Q with constraints A, B, C, "
                "a document satisfying only A and B might have higher similarity than one "
                "satisfying all three if A and B contribute more to the semantic meaning. "
                "The embedding 'averages' the constraints rather than requiring conjunction."
            ),
            AdversarialCategory.POLYSEMY: (
                "Word embeddings represent words as single vectors, even when words have "
                "multiple meanings (senses). 'Bank' gets one embedding that blends financial "
                "institution and river bank. Contextualized embeddings (like BERT) help but "
                "still struggle with rare senses or ambiguous contexts. The search query "
                "context may be insufficient to disambiguate."
            ),
            AdversarialCategory.COMPOSITIONAL: (
                "Sentence embeddings often behave like weighted bag-of-words, losing "
                "compositional structure. 'A acquired B' and 'B acquired A' may have "
                "similar embeddings because they contain the same words. Understanding "
                "who-does-what-to-whom requires structural parsing that embedding "
                "similarity doesn't capture well."
            ),
            AdversarialCategory.ANTONYM_CONFUSION: (
                "This is the distributional semantics paradox: antonyms appear in similar "
                "contexts and thus get similar embeddings. 'Good' and 'bad' both appear "
                "after 'The movie was'. 'Increase' and 'decrease' both appear with 'sales', "
                "'revenue', 'costs'. The embedding space sees them as semantically related "
                "because they're contextually substitutable."
            ),
            AdversarialCategory.SPECIFICITY_GRADIENT: (
                "Embedding spaces have finite resolution. Very specific queries may not "
                "have corresponding specific regions in the space - they get mapped to "
                "nearby more general concepts. Conversely, very broad queries create "
                "diffuse regions that match many different concepts. Testing the "
                "specificity limits reveals the granularity of the embedding space."
            ),
            AdversarialCategory.CROSS_LINGUAL: (
                "Multilingual embeddings attempt to align semantic spaces across languages, "
                "but alignment is imperfect. Rare words, domain-specific terms, and cultural "
                "concepts may not align well. A French query might retrieve English documents "
                "that are topically related but miss French-specific nuances."
            ),
            AdversarialCategory.COUNTERFACTUAL: (
                "Search systems retrieve existing content, which is predominantly factual. "
                "A counterfactual query like 'what if X never happened' is semantically "
                "similar to factual content about X happening. The embedding doesn't "
                "distinguish between factual and counterfactual modes - it matches "
                "based on semantic content (X) not modality (counterfactual)."
            ),
            AdversarialCategory.BOOLEAN_LOGIC: (
                "Vector spaces have no native boolean operations. AND, OR, NOT are "
                "approximated through vector arithmetic but fail for complex expressions. "
                "'A AND B' might be represented as (A + B)/2, but '(A OR B) AND C NOT D' "
                "has no clean vector representation. The boolean structure is lost in "
                "the averaging."
            ),
            AdversarialCategory.ENTITY_DISAMBIGUATION: (
                "Entity embeddings are dominated by the most common referent. 'Michael Jordan' "
                "embeds as the basketball player because that's 99% of training occurrences. "
                "The professor, actor, or other Michael Jordans are drowned out. Adding "
                "disambiguating context helps but may not overcome the prior."
            ),
            AdversarialCategory.INSTRUCTION_FOLLOWING: (
                "Meta-instructions like 'academic papers only' are semantic modifiers that "
                "compete with the main query content for embedding space. 'Academic papers "
                "only: transformers' is dominated by 'transformers' - the instruction "
                "becomes a weak signal. Structured filters (like Exa's category parameter) "
                "work better than natural language instructions."
            ),
        }
        return explanations.get(metadata.category, metadata.failure_hypothesis)

    async def generate_all_categories(
        self, count_per_category: int = 10
    ) -> list[Query]:
        """Generate queries for all categories."""
        all_queries = []

        for category in AdversarialCategory:
            print(f"Generating {count_per_category} queries for {category.value}...")
            queries = self.generate_for_category(category, count_per_category)
            all_queries.extend(queries)
            print(f"  Generated {len(queries)} queries")

        return all_queries


def generate_query_set(
    name: str | None = None,
    count_per_category: int = 10,
    tiers: list[str] | None = None,
    categories: list[AdversarialCategory] | None = None,
    use_llm: bool = True,
) -> QuerySet:
    """Generate a complete query set using multiple tiers.

    Args:
        name: Name for the query set
        count_per_category: Number of queries per category for LLM generation
        tiers: Which tiers to use: ['seed', 'template', 'llm'] (default: all)
        categories: Which categories to generate (default: all)
        use_llm: Whether to use LLM generation (requires API key)

    Returns:
        QuerySet with generated queries
    """
    if tiers is None:
        tiers = ["seed", "template", "llm"]
    if categories is None:
        categories = list(AdversarialCategory)

    query_set = QuerySet(name=name)
    settings = get_settings()

    # Tier 1: Seeds
    if "seed" in tiers:
        print("Loading seed queries...")
        seeds = load_all_seeds()
        for seed in seeds:
            if seed.category in categories:
                query_set.add_query(seed)
        print(f"  Added {len([q for q in query_set.queries if q.tier == 'seed'])} seed queries")

    # Tier 2: Templates
    if "template" in tiers:
        print("Generating template-based queries...")
        for category in categories:
            from searchprobe.queries.templates import generate_for_category

            templates = generate_for_category(category, max_queries_per_template=3)
            for q in templates:
                query_set.add_query(q)
        print(
            f"  Added {len([q for q in query_set.queries if q.tier == 'template'])} template queries"
        )

    # Tier 3: LLM generation
    if "llm" in tiers and use_llm:
        if not settings.has_anthropic_configured():
            print("  Skipping LLM generation (no Anthropic credentials configured)")
        else:
            print("Generating LLM-based queries...")
            generator = QueryGenerator()

            for category in categories:
                # Get existing queries to avoid duplicates
                existing = [q.text for q in query_set.queries if q.category == category]

                try:
                    queries = generator.generate_for_category(
                        category, count_per_category, existing
                    )
                    for q in queries:
                        query_set.add_query(q)
                except Exception as e:
                    print(f"  Error generating for {category.value}: {e}")

            print(
                f"  Added {len([q for q in query_set.queries if q.tier == 'llm'])} LLM queries"
            )

    # Set config
    query_set.config = {
        "tiers": tiers,
        "categories": [c.value for c in categories],
        "count_per_category": count_per_category,
        "use_llm": use_llm,
    }

    return query_set
