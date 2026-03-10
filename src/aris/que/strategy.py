"""Search strategy selection based on predicted failure modes."""

from __future__ import annotations

from aris.que.models import StructuredQuery


# Failure mode -> strategy adjustments
_MODE_ADJUSTMENTS: dict[str, dict] = {
    "negation_collapse": {
        "retriever_weights": {"sparse": 0.05, "dense": -0.1},
        "verification_strategies": ["negation"],
    },
    "numeric_blindness": {
        "retriever_weights": {"structured": 0.15, "dense": -0.1},
        "verification_strategies": ["numeric"],
    },
    "temporal_insensitivity": {
        "retriever_weights": {"structured": 0.1, "sparse": 0.05},
        "verification_strategies": ["temporal"],
    },
    "semantic_conflation": {
        "retriever_weights": {"hyde": 0.1, "dense": -0.05},
        "verification_strategies": ["entity"],
    },
    "keyword_leakage": {
        "retriever_weights": {"hyde": 0.1, "sparse": -0.1},
        "verification_strategies": [],
    },
    "popularity_bias": {
        "retriever_weights": {},
        "verification_strategies": ["entity"],
    },
    "partial_constraint": {
        "retriever_weights": {"structured": 0.1},
        "verification_strategies": [],
        "use_decomposition": True,
    },
    "inverse_result": {
        "retriever_weights": {"sparse": 0.05},
        "verification_strategies": ["negation"],
    },
}

# Default weights
_DEFAULT_WEIGHTS = {"dense": 0.4, "sparse": 0.25, "structured": 0.2, "hyde": 0.15}


def select_strategy(query: StructuredQuery) -> StructuredQuery:
    """Select optimal retrieval strategy based on predicted failure modes.

    Mutates the query's retriever_weights, verification_strategies, and use_decomposition.
    """
    weights = dict(_DEFAULT_WEIGHTS)
    verification: set[str] = set()
    use_decomposition = False

    for mode in query.predicted_failure_modes:
        adjustments = _MODE_ADJUSTMENTS.get(mode, {})

        # Adjust weights
        for key, delta in adjustments.get("retriever_weights", {}).items():
            weights[key] = weights.get(key, 0) + delta

        # Add verification strategies
        verification.update(adjustments.get("verification_strategies", []))

        # Check decomposition
        if adjustments.get("use_decomposition"):
            use_decomposition = True

    # Normalize weights to sum to 1
    total = sum(max(0, v) for v in weights.values())
    if total > 0:
        weights = {k: max(0, v) / total for k, v in weights.items()}

    # Also decompose if many constraints
    if len(query.constraints) >= 3:
        use_decomposition = True

    query.retriever_weights = weights
    query.verification_strategies = sorted(verification)
    query.use_decomposition = use_decomposition

    return query
