"""Entity extraction and disambiguation context generation."""

from __future__ import annotations

import re

from aris.que.models import EntityReference

# Known ambiguous entities with disambiguation hints
_AMBIGUOUS_ENTITIES: dict[str, list[dict[str, str]]] = {
    "python": [
        {"entity_type": "language", "disambiguation": "Python programming language"},
        {"entity_type": "animal", "disambiguation": "python snake species"},
    ],
    "java": [
        {"entity_type": "language", "disambiguation": "Java programming language"},
        {"entity_type": "place", "disambiguation": "Java island in Indonesia"},
    ],
    "rust": [
        {"entity_type": "language", "disambiguation": "Rust programming language"},
        {"entity_type": "game", "disambiguation": "Rust video game"},
    ],
    "swift": [
        {"entity_type": "language", "disambiguation": "Swift programming language by Apple"},
        {"entity_type": "person", "disambiguation": "Taylor Swift musician"},
    ],
    "apple": [
        {"entity_type": "company", "disambiguation": "Apple Inc. technology company"},
        {"entity_type": "food", "disambiguation": "apple fruit"},
    ],
    "michael jordan": [
        {"entity_type": "person", "disambiguation": "Michael Jordan basketball player"},
        {"entity_type": "person", "disambiguation": "Michael I. Jordan machine learning professor at UC Berkeley"},
    ],
    "mercury": [
        {"entity_type": "planet", "disambiguation": "Mercury planet in the solar system"},
        {"entity_type": "element", "disambiguation": "Mercury chemical element"},
        {"entity_type": "person", "disambiguation": "Freddie Mercury musician"},
    ],
}

# Context words that help disambiguate
_CONTEXT_HINTS: dict[str, list[str]] = {
    "language": ["programming", "code", "library", "framework", "developer", "software", "IDE", "compiler"],
    "company": ["company", "corporation", "inc", "startup", "founded", "CEO", "stock", "revenue"],
    "person": ["born", "died", "player", "professor", "author", "singer", "actor"],
    "place": ["island", "city", "country", "state", "province", "located"],
    "food": ["recipe", "cook", "eat", "taste", "fruit", "vegetable"],
    "game": ["game", "play", "server", "multiplayer"],
    "planet": ["planet", "orbit", "space", "solar"],
    "element": ["element", "chemical", "periodic"],
}


def extract_entities(query: str) -> list[EntityReference]:
    """Extract entities from query with disambiguation context."""
    entities = []
    query_lower = query.lower()

    # Check for known ambiguous entities
    for entity_name, variants in _AMBIGUOUS_ENTITIES.items():
        if entity_name in query_lower:
            # Try to disambiguate using context words
            best_variant = variants[0]  # default to first
            best_score = 0

            for variant in variants:
                etype = variant["entity_type"]
                hints = _CONTEXT_HINTS.get(etype, [])
                score = sum(1 for h in hints if h in query_lower)
                if score > best_score:
                    best_score = score
                    best_variant = variant

            entities.append(EntityReference(
                text=entity_name,
                entity_type=best_variant["entity_type"],
                disambiguation=best_variant["disambiguation"],
            ))

    return entities
