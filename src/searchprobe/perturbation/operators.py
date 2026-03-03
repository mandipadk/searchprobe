"""Perturbation operators for systematic query modification."""

import random
from enum import Enum


class PerturbationType(str, Enum):
    """Types of perturbation operators."""

    WORD_DELETE = "word_delete"
    WORD_SWAP = "word_swap"
    NEGATION_INSERT = "negation_insert"
    NEGATION_REMOVE = "negation_remove"
    SYNONYM_REPLACE = "synonym_replace"


# Common negation words
NEGATION_WORDS = {"not", "no", "never", "without", "neither", "nor", "none", "nothing"}

# Simple synonym map for when NLTK is not available
SIMPLE_SYNONYMS: dict[str, list[str]] = {
    "big": ["large", "huge", "enormous"],
    "small": ["tiny", "little", "miniature"],
    "fast": ["quick", "rapid", "swift"],
    "slow": ["sluggish", "gradual", "leisurely"],
    "good": ["great", "excellent", "fine"],
    "bad": ["poor", "terrible", "awful"],
    "new": ["recent", "fresh", "modern"],
    "old": ["ancient", "vintage", "aged"],
    "start": ["begin", "launch", "initiate"],
    "end": ["finish", "conclude", "terminate"],
    "find": ["discover", "locate", "search"],
    "make": ["create", "build", "construct"],
    "use": ["utilize", "employ", "apply"],
    "show": ["display", "present", "demonstrate"],
    "important": ["significant", "crucial", "vital"],
    "increase": ["grow", "rise", "expand"],
    "decrease": ["reduce", "decline", "shrink"],
    "companies": ["businesses", "firms", "corporations"],
    "startup": ["venture", "enterprise", "company"],
}


def word_delete(query: str, max_variants: int = 5) -> list[tuple[str, str]]:
    """Generate variants by deleting one word at a time.

    Args:
        query: Original query
        max_variants: Maximum number of variants

    Returns:
        List of (perturbed_query, detail) tuples
    """
    words = query.split()
    if len(words) <= 1:
        return []

    variants = []
    for i, word in enumerate(words):
        new_words = words[:i] + words[i + 1:]
        perturbed = " ".join(new_words)
        variants.append((perturbed, f"deleted '{word}' at position {i}"))
        if len(variants) >= max_variants:
            break

    return variants


def word_swap(query: str, max_variants: int = 5) -> list[tuple[str, str]]:
    """Generate variants by swapping adjacent words.

    Args:
        query: Original query
        max_variants: Maximum number of variants

    Returns:
        List of (perturbed_query, detail) tuples
    """
    words = query.split()
    if len(words) <= 1:
        return []

    variants = []
    indices = list(range(len(words) - 1))
    random.shuffle(indices)

    for i in indices[:max_variants]:
        new_words = words.copy()
        new_words[i], new_words[i + 1] = new_words[i + 1], new_words[i]
        perturbed = " ".join(new_words)
        detail = f"swapped '{words[i]}' and '{words[i + 1]}'"
        variants.append((perturbed, detail))

    return variants


def negation_insert(query: str, max_variants: int = 3) -> list[tuple[str, str]]:
    """Generate variants by inserting negation words.

    Args:
        query: Original query
        max_variants: Maximum number of variants

    Returns:
        List of (perturbed_query, detail) tuples
    """
    words = query.split()
    variants = []

    # Try inserting "not" before key positions
    for i in range(min(len(words), max_variants)):
        new_words = words[:i] + ["not"] + words[i:]
        perturbed = " ".join(new_words)
        variants.append((perturbed, f"inserted 'not' before '{words[i]}'"))

    return variants[:max_variants]


def negation_remove(query: str, **kwargs: object) -> list[tuple[str, str]]:
    """Generate variants by removing negation words.

    Args:
        query: Original query

    Returns:
        List of (perturbed_query, detail) tuples
    """
    words = query.split()
    variants = []

    for i, word in enumerate(words):
        if word.lower() in NEGATION_WORDS:
            new_words = words[:i] + words[i + 1:]
            perturbed = " ".join(new_words)
            variants.append((perturbed, f"removed negation '{word}'"))

    return variants


def synonym_replace(query: str, max_variants: int = 5) -> list[tuple[str, str]]:
    """Generate variants by replacing words with synonyms.

    Uses NLTK WordNet when available, falls back to a simple synonym map.

    Args:
        query: Original query
        max_variants: Maximum number of variants

    Returns:
        List of (perturbed_query, detail) tuples
    """
    words = query.split()
    variants = []

    for i, word in enumerate(words):
        synonyms = _get_synonyms(word.lower())
        if synonyms:
            syn = synonyms[0]
            new_words = words.copy()
            new_words[i] = syn
            perturbed = " ".join(new_words)
            variants.append((perturbed, f"replaced '{word}' with synonym '{syn}'"))
            if len(variants) >= max_variants:
                break

    return variants


def _get_synonyms(word: str) -> list[str]:
    """Get synonyms for a word.

    Tries NLTK WordNet first, falls back to simple synonym map.

    Args:
        word: Word to find synonyms for

    Returns:
        List of synonym strings
    """
    # Try NLTK WordNet
    try:
        from nltk.corpus import wordnet

        synsets = wordnet.synsets(word)
        synonyms = set()
        for synset in synsets:
            for lemma in synset.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():
                    synonyms.add(name)
        if synonyms:
            return list(synonyms)[:5]
    except (ImportError, LookupError):
        pass

    # Fall back to simple map
    return SIMPLE_SYNONYMS.get(word, [])


# Registry of operators
OPERATORS: dict[PerturbationType, object] = {
    PerturbationType.WORD_DELETE: word_delete,
    PerturbationType.WORD_SWAP: word_swap,
    PerturbationType.NEGATION_INSERT: negation_insert,
    PerturbationType.NEGATION_REMOVE: negation_remove,
    PerturbationType.SYNONYM_REPLACE: synonym_replace,
}


def apply_perturbation(
    query: str,
    perturbation_type: PerturbationType,
    max_variants: int = 5,
) -> list[tuple[str, str]]:
    """Apply a perturbation operator to a query.

    Args:
        query: Original query
        perturbation_type: Type of perturbation
        max_variants: Maximum variants to generate

    Returns:
        List of (perturbed_query, detail) tuples
    """
    operator = OPERATORS[perturbation_type]
    return operator(query, max_variants=max_variants)
