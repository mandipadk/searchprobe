"""Small dependency-free language identification.

Good enough to decide "is this page German?" for language constraints; when
confidence is low the judging layer defers to the LLM instead of guessing.
"""

from __future__ import annotations

import re
from collections import Counter

# Frequent function words per language (lowercase).
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        [
            "the",
            "and",
            "of",
            "to",
            "in",
            "is",
            "that",
            "for",
            "it",
            "with",
            "as",
            "was",
            "on",
            "are",
            "be",
            "this",
            "by",
        ]
    ),
    "de": frozenset(
        [
            "der",
            "die",
            "und",
            "das",
            "ist",
            "nicht",
            "zu",
            "den",
            "mit",
            "von",
            "sich",
            "des",
            "ein",
            "eine",
            "auch",
            "auf",
        ]
    ),
    "fr": frozenset(
        [
            "le",
            "la",
            "les",
            "et",
            "des",
            "est",
            "une",
            "un",
            "pour",
            "dans",
            "que",
            "qui",
            "pas",
            "sur",
            "du",
            "au",
            "avec",
        ]
    ),
    "es": frozenset(
        [
            "el",
            "la",
            "los",
            "las",
            "y",
            "de",
            "que",
            "en",
            "es",
            "un",
            "una",
            "por",
            "con",
            "para",
            "del",
            "se",
            "no",
        ]
    ),
    "it": frozenset(
        [
            "il",
            "la",
            "di",
            "e",
            "che",
            "è",
            "un",
            "una",
            "per",
            "non",
            "con",
            "del",
            "della",
            "sono",
            "gli",
            "le",
        ]
    ),
    "pt": frozenset(
        [
            "o",
            "a",
            "os",
            "as",
            "e",
            "de",
            "que",
            "em",
            "um",
            "uma",
            "para",
            "com",
            "não",
            "do",
            "da",
            "por",
            "são",
        ]
    ),
    "nl": frozenset(
        [
            "de",
            "het",
            "een",
            "en",
            "van",
            "is",
            "dat",
            "op",
            "te",
            "in",
            "niet",
            "zijn",
            "voor",
            "met",
            "die",
        ]
    ),
    "sv": frozenset(
        [
            "och",
            "att",
            "det",
            "som",
            "en",
            "på",
            "är",
            "av",
            "för",
            "med",
            "till",
            "den",
            "inte",
            "har",
        ]
    ),
    "pl": frozenset(
        ["i", "w", "na", "nie", "z", "się", "do", "to", "że", "jest", "jak", "o", "po", "co", "ale"]
    ),
    "tr": frozenset(
        ["ve", "bir", "bu", "da", "de", "için", "ile", "çok", "gibi", "daha", "ne", "olarak"]
    ),
}

# Script-based detection for languages identifiable by character ranges.
_SCRIPTS: list[tuple[str, re.Pattern[str]]] = [
    ("ja", re.compile(r"[぀-ヿ]")),  # hiragana/katakana
    ("ko", re.compile(r"[가-힯]")),
    ("zh", re.compile(r"[一-鿿]")),
    ("ru", re.compile(r"[Ѐ-ӿ]")),
    ("ar", re.compile(r"[؀-ۿ]")),
    ("he", re.compile(r"[֐-׿]")),
    ("el", re.compile(r"[Ͱ-Ͽ]")),
    ("hi", re.compile(r"[ऀ-ॿ]")),
    ("th", re.compile(r"[฀-๿]")),
]

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(text: str, sample_chars: int = 5000) -> tuple[str | None, float]:
    """Return (ISO 639-1 code, confidence in [0, 1]) or (None, 0.0)."""
    sample = text[:sample_chars]
    if not sample.strip():
        return None, 0.0

    letters = [ch for ch in sample if ch.isalpha()]
    if letters:
        script_counts = {code: len(pattern.findall(sample)) for code, pattern in _SCRIPTS}
        # Japanese text contains kanji too; any kana means Japanese.
        if script_counts["ja"] > 0.05 * len(letters):
            return "ja", min(1.0, script_counts["ja"] / (0.2 * len(letters)))
        best_script = max(script_counts, key=script_counts.get)
        share = script_counts[best_script] / len(letters)
        if share > 0.3:
            return best_script, min(1.0, share / 0.6)

    words = [w.lower() for w in _WORD.findall(sample)]
    if len(words) < 5:
        return None, 0.0
    counts = Counter(words)
    scores = {lang: sum(counts[w] for w in stop) / len(words) for lang, stop in _STOPWORDS.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    if top < 0.05:
        return None, 0.0
    margin = (top - second) / top
    confidence = max(0.0, min(1.0, margin * min(1.0, len(words) / 50)))
    return best, confidence


def normalize_lang_code(code: str | None) -> str | None:
    """'en-US' -> 'en'."""
    if not code:
        return None
    return code.strip().lower().replace("_", "-").split("-")[0] or None
