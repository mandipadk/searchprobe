"""Stability metrics for measuring result set consistency under perturbation.

Implements Jaccard similarity and Rank-Biased Overlap (Webber et al. 2010).
"""

from searchprobe.perturbation.models import PerturbationAnalysis, SensitivityMap
from searchprobe.perturbation.operators import PerturbationType


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two result sets.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        set_a: First result set (URLs)
        set_b: Second result set (URLs)

    Returns:
        Jaccard similarity in [0, 1]
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


def rank_biased_overlap(
    list_a: list[str],
    list_b: list[str],
    p: float = 0.9,
) -> float:
    """Extrapolated Rank-Biased Overlap (Webber et al. 2010, eq. 32).

    RBO is a top-weighted similarity for (possibly uneven) ranked lists.
    The extrapolated form assumes the agreement observed at the deepest
    seen rank continues, so identical lists score exactly 1.0 and disjoint
    lists score 0.0 (the truncated form scores identical top-10 lists 0.65).

    Args:
        list_a: First ranked list (e.g. URLs in order). Duplicates are ignored.
        list_b: Second ranked list.
        p: Persistence in (0, 1); higher weights deeper ranks more.

    Returns:
        RBO_ext in [0, 1].
    """
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    a = list(dict.fromkeys(list_a))
    b = list(dict.fromkeys(list_b))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    s, n = len(short), len(long_)

    seen_short: set[str] = set()
    seen_long: set[str] = set()
    overlap = 0
    x: list[int] = [0] * (n + 1)  # x[d] = overlap at depth d
    for d in range(1, n + 1):
        item_long = long_[d - 1]
        if d <= s:
            item_short = short[d - 1]
            if item_short == item_long:
                overlap += 1
            else:
                overlap += int(item_short in seen_long) + int(item_long in seen_short)
            seen_short.add(item_short)
        else:
            overlap += int(item_long in seen_short)
        seen_long.add(item_long)
        x[d] = overlap

    x_s, x_l = x[s], x[n]
    total = sum(x[d] / d * p**d for d in range(1, n + 1))
    total += sum(x_s * (d - s) / (s * d) * p**d for d in range(s + 1, n + 1))
    rbo = (1 - p) / p * total + ((x_l - x_s) / n + x_s / s) * p**n
    return float(min(1.0, max(0.0, rbo)))


# Exact names only: a prefix match on "ref" would also drop content-selecting
# parameters such as ?reference=2.
_TRACKING_PARAMS = frozenset(
    {"ref", "ref_src", "ref_url", "referrer", "fbclid", "gclid", "msclkid", "mc_cid", "mc_eid"}
)


def normalize_url(url: str) -> str:
    """Normalize a URL for result-set comparison.

    Lowercases scheme/host, drops ``www.``, fragments, tracking parameters and
    trailing slashes, and treats http/https as the same document.
    """
    from urllib.parse import parse_qsl, urlencode, urlsplit

    parts = urlsplit(url.strip())
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    query = urlencode(
        [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not (k.lower().startswith("utm_") or k.lower() in _TRACKING_PARAMS)
        ]
    )
    path = parts.path.rstrip("/") or "/"
    return f"{host}{path}" + (f"?{query}" if query else "")


def compute_sensitivity_map(
    query: str,
    analyses: list[PerturbationAnalysis],
) -> SensitivityMap:
    """Compute word-level sensitivity from perturbation analyses.

    Each word's sensitivity score reflects how much removing/changing it
    disrupts the search results. Higher = more load-bearing.

    Args:
        query: Original query
        analyses: List of PerturbationAnalysis from word_delete perturbations

    Returns:
        SensitivityMap with per-word scores
    """
    words = query.split()
    word_scores: dict[str, float] = {}

    # Default: all words start at 0 sensitivity
    for word in words:
        word_scores[word] = 0.0

    # For word_delete perturbations, sensitivity = 1 - stability
    for analysis in analyses:
        if analysis.perturbation_type == PerturbationType.WORD_DELETE.value:
            detail = analysis.perturbation_detail
            # Extract deleted word from detail string
            if "deleted '" in detail:
                deleted_word = detail.split("deleted '")[1].split("'")[0]
                # Sensitivity = how much results changed (1 - jaccard)
                sensitivity = 1.0 - analysis.jaccard_similarity
                if deleted_word in word_scores:
                    word_scores[deleted_word] = max(word_scores[deleted_word], sensitivity)

    # Normalize to [0, 1]
    max_score = max(word_scores.values()) if word_scores else 1.0
    if max_score > 0:
        word_scores = {w: s / max_score for w, s in word_scores.items()}

    provider = analyses[0].provider if analyses else "unknown"
    category = analyses[0].category if analyses else "unknown"

    return SensitivityMap(
        query=query,
        word_scores=word_scores,
        provider=provider,
        category=category,
    )
