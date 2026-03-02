"""Prompt templates for LLM-as-judge evaluation."""

from searchprobe.queries.taxonomy import AdversarialCategory

# Base system prompt for the judge
JUDGE_SYSTEM_PROMPT = """You are an expert search quality evaluator. Your task is to assess how well search results satisfy a given query.

You will be given:
1. A search query
2. The query's category (type of challenge it poses)
3. A list of search results (title, URL, snippet, and optionally content)
4. Specific evaluation dimensions to score

For each dimension, provide:
- A score from 0.0 to 1.0 (to 2 decimal places)
- A brief justification (1-2 sentences)

Be strict but fair. Consider:
- 0.0-0.2: Complete failure, results don't address the query at all
- 0.3-0.4: Poor, results are tangentially related but miss key requirements
- 0.5-0.6: Mediocre, some relevant results but significant gaps
- 0.7-0.8: Good, most results are relevant with minor issues
- 0.9-1.0: Excellent, results fully satisfy the query requirements

Always output valid JSON matching the required schema."""


# Category-specific evaluation guidance
CATEGORY_GUIDANCE: dict[AdversarialCategory, str] = {
    AdversarialCategory.NEGATION: """This query contains NEGATION (e.g., "NOT", "excluding", "without").
Pay special attention to whether results correctly EXCLUDE the negated concept.
If results include what was negated, score negation_respect very low.
Example: For "startups NOT in AI", any AI startup is a failure.""",
    AdversarialCategory.NUMERIC_PRECISION: """This query requires NUMERIC PRECISION (exact numbers, ranges, or quantities).
Check if results match the specific numeric requirements exactly.
Close but not exact (e.g., 48 employees when 50 was required) should score low.
Example: For "companies with exactly 50 employees", results with 45 or 55 employees fail.""",
    AdversarialCategory.TEMPORAL_CONSTRAINT: """This query has TEMPORAL CONSTRAINTS (dates, time periods, recency).
Verify results fall within the specified time period.
Results from wrong time periods should heavily penalize temporal_accuracy.
Example: For "news from third week of January 2025", February articles fail.""",
    AdversarialCategory.MULTI_CONSTRAINT: """This query has MULTIPLE CONSTRAINTS that must ALL be satisfied.
Each result should satisfy ALL requirements, not just some.
Partial matches (2 out of 3 constraints) should score moderately low.
Example: For "Series B startups in healthcare founded after 2020", all three criteria must match.""",
    AdversarialCategory.POLYSEMY: """This query involves POLYSEMY (words with multiple meanings).
Determine the intended meaning from context and check if results match it.
Results about the wrong sense (e.g., planet Mercury vs. element mercury) should fail.
Example: For "Mercury contamination in fish", results about the planet Mercury fail.""",
    AdversarialCategory.COMPOSITIONAL: """This query has COMPOSITIONAL structure where word order matters.
The relationship between entities is crucial (A acquired B ≠ B acquired A).
Results with reversed relationships should score very low.
Example: For "acquired by their former subsidiary", the direction of acquisition matters.""",
    AdversarialCategory.ANTONYM_CONFUSION: """This query uses terms with ANTONYMS that embeddings may confuse.
Check that results match the correct polarity (increase vs decrease, etc.).
Results about the opposite concept should fail.
Example: For "decrease employee turnover", results about increasing turnover fail.""",
    AdversarialCategory.SPECIFICITY_GRADIENT: """This query tests SPECIFICITY (narrow vs. broad matching).
For narrow queries, results should be precise, not just tangentially related.
For broad queries, coverage matters more than precision.
Example: For a very specific niche query, general overview articles fail.""",
    AdversarialCategory.CROSS_LINGUAL: """This query involves MULTIPLE LANGUAGES or cross-lingual concepts.
Results should be in the appropriate language(s) or handle the multilingual aspect.
Language mismatches or translation errors should be penalized.
Example: For mixed English-Spanish queries, results should handle both appropriately.""",
    AdversarialCategory.COUNTERFACTUAL: """This query is COUNTERFACTUAL (hypothetical scenarios).
Results should engage with the hypothetical, not just return factual content.
Factual results that ignore the counterfactual premise fail.
Example: For "what if Bitcoin was never invented", results about actual Bitcoin history fail.""",
    AdversarialCategory.BOOLEAN_LOGIC: """This query uses BOOLEAN LOGIC (AND, OR, NOT combinations).
Results must satisfy the logical structure of the query.
OR means any matching, AND means all matching, NOT means excluding.
Example: For "(AI OR ML) AND healthcare NOT insurance", insurance companies fail.""",
    AdversarialCategory.ENTITY_DISAMBIGUATION: """This query requires ENTITY DISAMBIGUATION (distinguishing similar names).
The disambiguating context must be respected.
Results about the wrong entity (e.g., wrong Michael Jordan) fail.
Example: For "Michael Jordan the professor", results about the basketball player fail.""",
    AdversarialCategory.INSTRUCTION_FOLLOWING: """This query contains META-INSTRUCTIONS about result format or type.
Results should follow the instruction (e.g., "academic papers only").
Results of wrong type/format despite being topically relevant should fail.
Example: For "academic papers only on climate change", news articles fail.""",
}


def build_evaluation_prompt(
    query: str,
    category: str,
    results: list[dict],
    dimensions: list[str],
    ground_truth: dict | None = None,
) -> str:
    """Build the evaluation prompt for the LLM judge.

    Args:
        query: The search query
        category: Adversarial category name
        results: List of search result dicts
        dimensions: Dimensions to evaluate
        ground_truth: Optional ground truth for calibration

    Returns:
        Formatted prompt string
    """
    # Get category-specific guidance
    try:
        cat_enum = AdversarialCategory(category)
        guidance = CATEGORY_GUIDANCE.get(cat_enum, "")
    except ValueError:
        guidance = ""

    # Format results
    results_text = ""
    for i, result in enumerate(results[:10], 1):  # Limit to top 10
        results_text += f"\n--- Result {i} ---\n"
        results_text += f"Title: {result.get('title', 'N/A')}\n"
        results_text += f"URL: {result.get('url', 'N/A')}\n"
        results_text += f"Snippet: {result.get('snippet', 'N/A')}\n"
        if result.get("content"):
            # Truncate content
            content = result["content"][:1000]
            if len(result["content"]) > 1000:
                content += "... [truncated]"
            results_text += f"Content: {content}\n"

    # Format dimensions
    dimensions_text = "\n".join(f"- {dim}" for dim in dimensions)

    # Build prompt
    prompt = f"""Evaluate these search results for the following query.

## Query
"{query}"

## Category
{category}

{f"## Category-Specific Guidance{chr(10)}{guidance}" if guidance else ""}

## Search Results
{results_text}

## Dimensions to Evaluate
{dimensions_text}

## Instructions
For each dimension listed above, provide:
1. A score from 0.0 to 1.0
2. A brief justification (1-2 sentences)

Also identify:
- failure_modes: List of specific ways the results failed (if any)
- best_result_index: Index (1-based) of the best result, or null if all fail
- overall_assessment: One sentence summary

Respond with valid JSON in this exact format:
{{
  "scores": {{
    "dimension_name": {{
      "score": 0.75,
      "justification": "Brief explanation"
    }}
  }},
  "failure_modes": ["failure1", "failure2"],
  "best_result_index": 1,
  "overall_assessment": "One sentence summary"
}}"""

    # Add ground truth if available (for calibration)
    if ground_truth:
        prompt += f"""

## Reference Information (for your calibration)
Expected relevant results should match: {ground_truth.get('should_match', 'N/A')}
Expected irrelevant results would match: {ground_truth.get('should_not_match', 'N/A')}
Expected entity disambiguation: {ground_truth.get('expected_entity', 'N/A')}"""

    return prompt


def build_batch_prompt(
    evaluations: list[dict],
) -> str:
    """Build a prompt for batch evaluation of multiple queries.

    Args:
        evaluations: List of dicts with query, category, results, dimensions

    Returns:
        Formatted prompt for batch evaluation
    """
    prompt = """Evaluate multiple search result sets. For each query, provide scores.

"""
    for i, eval_item in enumerate(evaluations, 1):
        prompt += f"\n{'='*50}\n"
        prompt += f"## Query Set {i}\n"
        prompt += build_evaluation_prompt(
            query=eval_item["query"],
            category=eval_item["category"],
            results=eval_item["results"],
            dimensions=eval_item["dimensions"],
        )

    prompt += """

Respond with a JSON array, one evaluation object per query set:
[
  { evaluation for query 1 },
  { evaluation for query 2 },
  ...
]"""

    return prompt
