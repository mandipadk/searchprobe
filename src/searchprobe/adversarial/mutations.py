"""Mutation operators for evolving adversarial queries."""

import random
from typing import Any

from searchprobe.adversarial.models import Individual


def word_substitute(individual: Individual, **kwargs: Any) -> Individual:
    """Replace a word with a synonym/hypernym.

    Uses NLTK WordNet when available, falls back to simple synonyms.
    """
    words = individual.query.split()
    if len(words) < 2:
        return individual

    idx = random.randint(0, len(words) - 1)
    original_word = words[idx]
    synonym = _get_synonym(original_word)

    if synonym and synonym != original_word:
        words[idx] = synonym
        return _new_individual(
            individual,
            " ".join(words),
            f"word_substitute('{original_word}'→'{synonym}')",
        )

    return individual


def negation_toggle(individual: Individual, **kwargs: Any) -> Individual:
    """Insert or remove negation words."""
    negation_words = {"not", "no", "never", "without", "neither"}
    words = individual.query.split()

    # Check if query has negation
    has_negation = any(w.lower() in negation_words for w in words)

    if has_negation:
        # Remove negation
        new_words = [w for w in words if w.lower() not in negation_words]
        detail = "negation_remove"
    else:
        # Insert negation at a plausible position
        insert_pos = random.randint(0, len(words))
        neg_word = random.choice(["not", "never", "without"])
        new_words = words[:insert_pos] + [neg_word] + words[insert_pos:]
        detail = f"negation_insert('{neg_word}' at {insert_pos})"

    return _new_individual(individual, " ".join(new_words), detail)


def constraint_inject(individual: Individual, **kwargs: Any) -> Individual:
    """Add a constraint clause to the query."""
    constraints = [
        "founded after 2020",
        "with more than 100 employees",
        "based in Europe",
        "with revenue over $10 million",
        "open source only",
        "published in peer-reviewed journals",
        "with exactly 50 employees",
        "from the last 30 days",
        "in the healthcare sector",
        "written in Python",
        "with fewer than 10 employees",
        "valued at over $1 billion",
    ]

    constraint = random.choice(constraints)
    new_query = f"{individual.query} {constraint}"

    return _new_individual(
        individual, new_query, f"constraint_inject('{constraint}')"
    )


def specificity_shift(individual: Individual, **kwargs: Any) -> Individual:
    """Make the query more specific or more general."""
    direction = random.choice(["specific", "general"])

    if direction == "specific":
        specifiers = [
            "specifically",
            "exactly",
            "the most notable",
            "the first",
            "the largest",
            "in particular",
            "as of 2024",
        ]
        spec = random.choice(specifiers)
        new_query = f"{spec} {individual.query}"
        detail = f"specificity_increase('{spec}')"
    else:
        # Try to remove specificity markers
        words = individual.query.split()
        remove_words = {"specifically", "exactly", "particular", "precise", "first", "largest"}
        new_words = [w for w in words if w.lower() not in remove_words]
        if len(new_words) < len(words):
            new_query = " ".join(new_words)
            detail = "specificity_decrease"
        else:
            # Add generalization
            new_query = f"anything related to {individual.query}"
            detail = "specificity_decrease(generalized)"

    return _new_individual(individual, new_query, detail)


def category_blend(individual: Individual, partner: Individual | None = None, **kwargs: Any) -> Individual:
    """Combine features from two adversarial categories.

    Takes a query and blends in challenge patterns from a different category.
    """
    blend_patterns = [
        "NOT",
        "with exactly",
        "before 2020",
        "academic papers only about",
        "hypothetically, if",
        "in both French and English about",
    ]

    pattern = random.choice(blend_patterns)
    words = individual.query.split()

    # Insert pattern at a random position
    insert_pos = random.randint(0, len(words))
    new_words = words[:insert_pos] + pattern.split() + words[insert_pos:]

    return _new_individual(
        individual, " ".join(new_words), f"category_blend('{pattern}')"
    )


def tense_flip(individual: Individual, **kwargs: Any) -> Individual:
    """Change temporal framing of the query."""
    tense_pairs = {
        "will": "did",
        "future": "past",
        "upcoming": "previous",
        "next": "last",
        "new": "old",
        "current": "former",
        "latest": "earliest",
        "after": "before",
        "since": "until",
    }

    words = individual.query.split()
    changed = False
    for i, word in enumerate(words):
        lower = word.lower()
        if lower in tense_pairs:
            words[i] = tense_pairs[lower]
            changed = True
            break
        # Check reverse mapping
        for k, v in tense_pairs.items():
            if lower == v:
                words[i] = k
                changed = True
                break
        if changed:
            break

    if not changed:
        # Add temporal context
        temporal = random.choice(["historically", "in the past", "in the future", "originally"])
        words.insert(0, temporal)

    return _new_individual(individual, " ".join(words), "tense_flip")


def entity_swap(individual: Individual, **kwargs: Any) -> Individual:
    """Replace an entity with an ambiguous alternative."""
    entity_swaps = {
        "Google": "Alphabet",
        "Apple": "apple",
        "Amazon": "amazon",
        "Python": "python",
        "Java": "java",
        "Mercury": "mercury",
        "Jordan": "jordan",
        "Paris": "paris",
        "Cambridge": "cambridge",
        "Tesla": "tesla",
    }

    words = individual.query.split()
    for i, word in enumerate(words):
        if word in entity_swaps:
            words[i] = entity_swaps[word]
            return _new_individual(
                individual, " ".join(words),
                f"entity_swap('{word}'→'{entity_swaps[word]}')",
            )

    return individual


def _new_individual(parent: Individual, new_query: str, mutation: str) -> Individual:
    """Create a new individual from a parent with a mutation."""
    return Individual(
        query=new_query,
        category=parent.category,
        generation=parent.generation + 1,
        parent_ids=parent.parent_ids + [parent.id],
        mutation_history=parent.mutation_history + [mutation],
        metadata={**parent.metadata, "last_mutation": mutation},
    )


def _get_synonym(word: str) -> str | None:
    """Get a synonym for a word."""
    try:
        from nltk.corpus import wordnet

        synsets = wordnet.synsets(word)
        for synset in synsets:
            for lemma in synset.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():
                    return name
    except (ImportError, LookupError):
        pass

    # Simple fallback
    simple_syns = {
        "companies": "businesses",
        "startups": "ventures",
        "find": "discover",
        "search": "lookup",
        "best": "top",
        "large": "big",
        "small": "tiny",
    }
    return simple_syns.get(word.lower())


# Registry of all mutation operators
MUTATION_OPERATORS = {
    "word_substitute": word_substitute,
    "negation_toggle": negation_toggle,
    "constraint_inject": constraint_inject,
    "specificity_shift": specificity_shift,
    "category_blend": category_blend,
    "tense_flip": tense_flip,
    "entity_swap": entity_swap,
}


def apply_random_mutation(individual: Individual) -> Individual:
    """Apply a random mutation operator to an individual."""
    operator_name = random.choice(list(MUTATION_OPERATORS.keys()))
    operator = MUTATION_OPERATORS[operator_name]
    return operator(individual)
