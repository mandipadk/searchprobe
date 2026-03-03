"""Crossover operators for combining adversarial query features."""

import random

from searchprobe.adversarial.models import Individual


def clause_swap(parent_a: Individual, parent_b: Individual) -> Individual:
    """Swap clauses between two parent queries.

    Splits each parent at a random point and combines halves.
    """
    words_a = parent_a.query.split()
    words_b = parent_b.query.split()

    if len(words_a) < 2 or len(words_b) < 2:
        return parent_a

    # Find split points
    split_a = random.randint(1, len(words_a) - 1)
    split_b = random.randint(1, len(words_b) - 1)

    # Combine first half of A with second half of B
    new_words = words_a[:split_a] + words_b[split_b:]
    new_query = " ".join(new_words)

    return Individual(
        query=new_query,
        category=parent_a.category or parent_b.category,
        generation=max(parent_a.generation, parent_b.generation) + 1,
        parent_ids=[parent_a.id, parent_b.id],
        mutation_history=parent_a.mutation_history + ["clause_swap"],
    )


def constraint_merge(parent_a: Individual, parent_b: Individual) -> Individual:
    """Merge constraints from both parents.

    Combines both queries, attempting to merge their constraints.
    """
    # Simple strategy: concatenate with conjunction
    query_a = parent_a.query.rstrip(".")
    query_b = parent_b.query.rstrip(".")

    # Try to find a natural merge point
    merge_words = ["and also", "that additionally", "and furthermore"]
    merge = random.choice(merge_words)

    new_query = f"{query_a} {merge} {query_b}"

    # Truncate if too long
    words = new_query.split()
    if len(words) > 30:
        new_query = " ".join(words[:30])

    return Individual(
        query=new_query,
        category=parent_a.category or parent_b.category,
        generation=max(parent_a.generation, parent_b.generation) + 1,
        parent_ids=[parent_a.id, parent_b.id],
        mutation_history=parent_a.mutation_history + ["constraint_merge"],
    )


def template_blend(parent_a: Individual, parent_b: Individual) -> Individual:
    """Blend query templates by mixing structural elements.

    Takes the structure of one query and fills in content words from another.
    """
    words_a = parent_a.query.split()
    words_b = parent_b.query.split()

    # Function words (keep from parent_a's structure)
    function_words = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
        "and", "or", "but", "not", "no", "is", "are", "was", "were",
        "that", "which", "who", "what", "how", "when", "where", "why",
        "from", "by", "about", "between", "through", "during", "before",
        "after", "above", "below", "than", "only", "also",
    }

    # Get content words from parent_b
    content_words_b = [w for w in words_b if w.lower() not in function_words]

    if not content_words_b:
        return clause_swap(parent_a, parent_b)

    # Replace some content words in parent_a with words from parent_b
    new_words = words_a.copy()
    content_idx_a = [i for i, w in enumerate(words_a) if w.lower() not in function_words]

    if content_idx_a:
        # Replace 1-2 content words
        n_replace = min(2, len(content_idx_a), len(content_words_b))
        replace_indices = random.sample(content_idx_a, n_replace)
        replacements = random.sample(content_words_b, n_replace)

        for idx, replacement in zip(replace_indices, replacements):
            new_words[idx] = replacement

    return Individual(
        query=" ".join(new_words),
        category=parent_a.category or parent_b.category,
        generation=max(parent_a.generation, parent_b.generation) + 1,
        parent_ids=[parent_a.id, parent_b.id],
        mutation_history=parent_a.mutation_history + ["template_blend"],
    )


# Registry of crossover operators
CROSSOVER_OPERATORS = {
    "clause_swap": clause_swap,
    "constraint_merge": constraint_merge,
    "template_blend": template_blend,
}


def apply_random_crossover(parent_a: Individual, parent_b: Individual) -> Individual:
    """Apply a random crossover operator to two parents."""
    operator_name = random.choice(list(CROSSOVER_OPERATORS.keys()))
    operator = CROSSOVER_OPERATORS[operator_name]
    return operator(parent_a, parent_b)
