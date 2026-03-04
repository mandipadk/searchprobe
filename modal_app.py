"""SearchProbe Modal App — GPU-accelerated remote computation for embedding analysis.

Run with:
    modal run modal_app.py --command geometry --models "all-MiniLM-L6-v2,all-mpnet-base-v2"
    modal run modal_app.py --command geometry --categories "negation,antonym_confusion"
    modal run modal_app.py --command geometry --output geometry_results.json
"""

from __future__ import annotations

import json
import math
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Modal App & Image
# ---------------------------------------------------------------------------

app = modal.App("searchprobe")


def download_models():
    """Pre-download sentence-transformer and cross-encoder models into the image."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    for model_name in [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
    ]:
        SentenceTransformer(model_name)
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")


searchprobe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=3.0",
        "torch>=2.0",
        "scikit-learn>=1.4",
        "umap-learn>=0.5",
        "nltk>=3.8",
        "numpy>=1.26",
        "scipy>=1.12",
    )
    .run_function(download_models)
)

model_volume = modal.Volume.from_name("searchprobe-models", create_if_missing=True)


# ---------------------------------------------------------------------------
# Adversarial Pairs — embedded inline so Modal image doesn't need searchprobe
# ---------------------------------------------------------------------------

_ADVERSARIAL_PAIRS: dict[str, list[tuple[str, str, str]]] = {
    "negation": [
        ("companies in AI", "companies NOT in AI", "Negation should flip meaning"),
        ("startups that raised venture capital", "startups that never raised venture capital", ""),
        ("programming languages that are object-oriented", "programming languages that are NOT object-oriented", ""),
        ("foods containing gluten", "foods containing no gluten", ""),
        ("countries with a coastline", "countries without a coastline", ""),
        ("open source software", "proprietary software that is not open source", ""),
    ],
    "numeric_precision": [
        ("companies with 50 employees", "companies with 5000 employees", "50 vs 5000"),
        ("buildings taller than 500 meters", "buildings shorter than 50 meters", ""),
        ("startups that raised $10 million", "startups that raised $10 billion", ""),
        ("products priced at $10", "products priced at $10000", ""),
        ("cities with 100000 people", "cities with 10 million people", ""),
    ],
    "temporal_constraint": [
        ("news from January 2024", "news from January 2020", ""),
        ("tech layoffs in 2024", "tech layoffs in 2019", ""),
        ("events from last week", "events from last year", ""),
        ("startups founded in 2023", "startups founded in 2010", ""),
        ("articles published yesterday", "articles published a decade ago", ""),
    ],
    "multi_constraint": [
        ("Python libraries for computer vision released after 2023", "Python libraries for computer vision", ""),
        ("female CEOs of Fortune 500 healthcare companies", "CEOs of Fortune 500 companies", ""),
        ("open-source databases in Rust with ACID", "databases with ACID transactions", ""),
        ("remote Rust jobs paying over $200k", "software engineering jobs", ""),
        ("vegan restaurants in Tokyo with Michelin stars open on Mondays", "restaurants in Tokyo", ""),
    ],
    "polysemy": [
        ("Java programming performance", "Java island tourism", "Same word, different senses"),
        ("Python snake habitat", "Python programming tutorial", ""),
        ("Apple fruit nutrition", "Apple company stock price", ""),
        ("Mercury planet distance from sun", "Mercury element toxicity", ""),
        ("crane bird migration", "crane construction equipment", ""),
        ("bank of the river erosion", "bank financial services", ""),
    ],
    "compositional": [
        ("companies that acquired startups", "startups that acquired companies", "Same words, opposite meaning"),
        ("dog bites man", "man bites dog", ""),
        ("teachers who became students", "students who became teachers", ""),
        ("the cause of the effect", "the effect of the cause", ""),
        ("parents learning from children", "children learning from parents", ""),
    ],
    "antonym_confusion": [
        ("strategies to increase employee retention", "strategies to decrease employee retention", ""),
        ("methods to accelerate development", "methods to slow down development", ""),
        ("ways to simplify complexity", "ways to add complexity", ""),
        ("companies that succeeded", "companies that failed", ""),
        ("technologies that improve security", "technologies that weaken security", ""),
        ("reasons to buy stocks", "reasons to sell stocks", ""),
    ],
    "specificity_gradient": [
        ("machine learning", "the specific implementation of dropout regularization in ResNet-50 layer 3", ""),
        ("web development", "the exact CSS grid template used in Stripe's pricing page header", ""),
        ("databases", "the B+ tree rebalancing algorithm in PostgreSQL 16's btree AM", ""),
        ("artificial intelligence", "the attention head pruning strategy in GPT-4's 73rd transformer layer", ""),
        ("programming", "the exact memory layout of a Vec<T> in Rust 1.75 on ARM64", ""),
    ],
    "cross_lingual": [
        ("artificial intelligence research", "recherche en intelligence artificielle", ""),
        ("machine learning tutorial", "tutoriel d'apprentissage automatique", ""),
        ("climate change effects", "Auswirkungen des Klimawandels", ""),
        ("startup ecosystem", "ecosistema de startups", ""),
        ("database optimization", "\u30c7\u30fc\u30bf\u30d9\u30fc\u30b9\u306e\u6700\u9069\u5316", ""),
    ],
    "counterfactual": [
        ("the invention of the internet", "what if the internet was never invented", ""),
        ("Bitcoin's creation and growth", "what if Bitcoin had never been created", ""),
        ("history of social media", "a world without social media", ""),
        ("Apple's founding and success", "what if Apple had gone bankrupt in 1997", ""),
        ("the discovery of penicillin", "hypothetical timeline without antibiotics", ""),
    ],
    "boolean_logic": [
        ("machine learning AND healthcare", "machine learning AND healthcare AND NOT radiology", ""),
        ("Python AND web frameworks", "Python AND (FastAPI OR Django) AND NOT Flask", ""),
        ("renewable energy", "(solar OR wind) AND NOT residential AND commercial", ""),
        ("database performance", "(PostgreSQL OR MySQL) AND performance AND NOT MongoDB", ""),
        ("startup funding", "startup AND (acquired OR merged) AND 2024 AND NOT SPAC", ""),
    ],
    "entity_disambiguation": [
        ("Michael Jordan basketball career", "Michael Jordan professor machine learning Berkeley", ""),
        ("Paris France tourism", "Paris Texas community", ""),
        ("Amazon online shopping", "Amazon river ecosystem", ""),
        ("Apple iPhone features", "apple fruit varieties", ""),
        ("Cambridge University UK", "Cambridge Massachusetts MIT", ""),
        ("Jordan country Middle East", "Jordan basketball shoes Nike", ""),
    ],
    "instruction_following": [
        ("transformer architecture improvements", "academic papers only: transformer architecture improvements", ""),
        ("Kubernetes deployment", "find blog posts, not documentation, about Kubernetes deployment", ""),
        ("learning Rust programming", "personal experiences with learning Rust, not tutorials", ""),
        ("OpenAI news", "news articles only about OpenAI, no blog posts", ""),
        ("climate change research", "peer-reviewed studies only about climate change, no opinion pieces", ""),
    ],
}

_BASELINE_PAIRS: dict[str, list[tuple[str, str]]] = {
    "negation": [("companies in AI", "AI startups"), ("venture capital firms", "investment companies")],
    "numeric_precision": [("large companies", "big corporations"), ("startup funding", "venture capital investment")],
    "temporal_constraint": [("recent news", "latest updates"), ("tech trends", "technology developments")],
    "polysemy": [("programming languages", "coding languages"), ("river ecosystems", "freshwater habitats")],
    "antonym_confusion": [("employee retention", "keeping employees"), ("code quality", "software quality")],
}

_RANDOM_PAIRS: list[tuple[str, str]] = [
    ("quantum computing algorithms", "Italian pasta recipes"),
    ("medieval castle architecture", "Python web frameworks"),
    ("deep sea marine biology", "real estate investment"),
    ("ancient Egyptian hieroglyphics", "smartphone battery technology"),
    ("jazz music improvisation", "compiler optimization techniques"),
    ("volcanic eruption prediction", "fashion design trends"),
    ("ballet dance training", "database sharding strategies"),
    ("beekeeping for beginners", "spacecraft navigation systems"),
]

_CATEGORY_THRESHOLDS: dict[str, float] = {
    "negation": 0.85, "numeric_precision": 0.80, "temporal_constraint": 0.82,
    "multi_constraint": 0.75, "polysemy": 0.50, "compositional": 0.80,
    "antonym_confusion": 0.80, "specificity_gradient": 0.60, "cross_lingual": 0.70,
    "counterfactual": 0.80, "boolean_logic": 0.80, "entity_disambiguation": 0.50,
    "instruction_following": 0.85,
}

ALL_CATEGORIES = list(_ADVERSARIAL_PAIRS.keys())


# ---------------------------------------------------------------------------
# Inline Metric Functions
# ---------------------------------------------------------------------------

def _cosine_similarity(a, b):
    """Cosine similarity between two numpy vectors."""
    import numpy as np
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _angular_distance(a, b):
    """Normalized angular distance in [0, 1]."""
    sim = _cosine_similarity(a, b)
    sim = max(-1.0, min(1.0, sim))
    return float(math.acos(sim) / math.pi)


def _local_intrinsic_dimensionality(embeddings, k=10):
    """MLE estimate of local intrinsic dimensionality."""
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    n = embeddings.shape[0]
    if n < k + 1:
        k = max(1, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    distances = distances[:, 1:]
    distances = np.maximum(distances, 1e-10)
    lids = []
    for i in range(n):
        d_k = distances[i, -1]
        if d_k <= 0:
            continue
        log_ratios = np.log(distances[i] / d_k)
        mean_lr = np.mean(log_ratios)
        if mean_lr < 0:
            lids.append(-1.0 / mean_lr)
    return float(np.mean(lids)) if lids else 0.0


def _isotropy_score(embeddings):
    """Entropy-based isotropy score in [0, 1]."""
    import numpy as np
    if embeddings.shape[0] < 2:
        return 0.0
    centered = embeddings - np.mean(embeddings, axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)
    total = eigenvalues.sum()
    if total == 0:
        return 0.0
    ev_norm = eigenvalues / total
    nonzero = ev_norm[ev_norm > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    max_entropy = np.log(len(ev_norm))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _compute_vulnerability_score(adv_sim, baseline_sim, random_sim, category):
    """Compute vulnerability score for a category."""
    threshold = _CATEGORY_THRESHOLDS.get(category, 0.80)
    random_sim = random_sim if random_sim > 0 else 0.1
    exceedance = max(0.0, (adv_sim - threshold) / (1.0 - threshold))
    exceedance = min(1.0, exceedance)
    collapse = min(2.0, adv_sim / baseline_sim) / 2.0 if baseline_sim > 0 else 0.5
    if random_sim < 1.0:
        random_adjusted = max(0.0, (adv_sim - random_sim) / (1.0 - random_sim))
        random_adjusted = min(1.0, random_adjusted)
    else:
        random_adjusted = 1.0
    score = 0.4 * exceedance + 0.35 * collapse + 0.25 * random_adjusted
    return min(1.0, max(0.0, score))


def _compute_ndcg(relevance_scores, k=None):
    """Normalized Discounted Cumulative Gain."""
    if not relevance_scores:
        return 0.0
    if k is not None:
        relevance_scores = relevance_scores[:k]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores))
    ideal = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Modal Functions
# ---------------------------------------------------------------------------

@app.function(
    image=searchprobe_image,
    gpu="T4",
    volumes={"/models": model_volume},
    timeout=600,
)
def analyze_geometry(
    models: list[str] | None = None,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Run embedding geometry analysis on GPU.

    Args:
        models: Sentence-transformer model names (default: all-MiniLM-L6-v2, all-mpnet-base-v2)
        categories: Categories to analyze (default: all 13)

    Returns:
        Dict with vulnerability_matrix, profiles, and per-model/per-category details.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if models is None:
        models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    if categories is None:
        categories = ALL_CATEGORIES

    result: dict[str, Any] = {"models": models, "profiles": {}, "vulnerability_matrix": {}}

    for model_name in models:
        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name, device="cuda")
        model_profiles: dict[str, Any] = {}

        for category in categories:
            adv_pairs = _ADVERSARIAL_PAIRS.get(category, [])
            base_pairs = _BASELINE_PAIRS.get(category, [
                ("machine learning algorithms", "ML techniques"),
                ("web development", "building websites"),
            ])
            random_pairs = _RANDOM_PAIRS

            # Encode all texts
            adv_texts = []
            for a, b, _ in adv_pairs:
                adv_texts.extend([a, b])
            base_texts = []
            for a, b in base_pairs:
                base_texts.extend([a, b])
            rand_texts = []
            for a, b in random_pairs:
                rand_texts.extend([a, b])

            all_texts = adv_texts + base_texts + rand_texts
            embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)

            # Compute adversarial similarities
            adv_sims = []
            pair_details = []
            adv_embs = embeddings[:len(adv_texts)]
            for i, (a, b, desc) in enumerate(adv_pairs):
                ea, eb = adv_embs[2 * i], adv_embs[2 * i + 1]
                sim = _cosine_similarity(ea, eb)
                ang = _angular_distance(ea, eb)
                adv_sims.append(sim)
                pair_details.append({
                    "query_a": a, "query_b": b, "similarity": sim,
                    "angular_distance": ang, "description": desc,
                })

            # Compute baseline similarities
            base_embs = embeddings[len(adv_texts):len(adv_texts) + len(base_texts)]
            base_sims = []
            for i in range(len(base_pairs)):
                sim = _cosine_similarity(base_embs[2 * i], base_embs[2 * i + 1])
                base_sims.append(sim)

            # Compute random similarities
            rand_embs = embeddings[len(adv_texts) + len(base_texts):]
            rand_sims = []
            for i in range(len(random_pairs)):
                sim = _cosine_similarity(rand_embs[2 * i], rand_embs[2 * i + 1])
                rand_sims.append(sim)

            mean_adv = float(np.mean(adv_sims)) if adv_sims else 0.0
            mean_base = float(np.mean(base_sims)) if base_sims else 0.0
            mean_rand = float(np.mean(rand_sims)) if rand_sims else 0.0
            collapse_ratio = mean_adv / mean_base if mean_base > 0 else 1.0
            vulnerability = _compute_vulnerability_score(mean_adv, mean_base, mean_rand, category)

            # Geometric properties
            lid = _local_intrinsic_dimensionality(adv_embs) if len(adv_embs) >= 4 else 0.0
            iso = _isotropy_score(adv_embs) if len(adv_embs) >= 4 else 0.0

            model_profiles[category] = {
                "category": category,
                "model_name": model_name,
                "mean_adversarial_sim": mean_adv,
                "mean_baseline_sim": mean_base,
                "mean_random_sim": mean_rand,
                "collapse_ratio": collapse_ratio,
                "vulnerability_score": vulnerability,
                "intrinsic_dimensionality": lid,
                "isotropy_score": iso,
                "pair_details": pair_details,
                "adversarial_similarities": adv_sims,
                "baseline_similarities": base_sims,
                "random_similarities": rand_sims,
            }

        result["profiles"][model_name] = model_profiles
        result["vulnerability_matrix"][model_name] = {
            cat: p["vulnerability_score"] for cat, p in model_profiles.items()
        }

    return result


@app.function(
    image=searchprobe_image,
    gpu="T4",
    volumes={"/models": model_volume},
    timeout=300,
)
def validate_cross_encoder(
    query_text: str,
    results: list[dict[str, str]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
) -> dict[str, Any]:
    """Score search results using a cross-encoder and compute NDCG metrics.

    Args:
        query_text: The search query
        results: List of dicts with 'title', 'url', 'snippet', optional 'content'
        model_name: Cross-encoder model name

    Returns:
        Dict with scores, original_ndcg, reranked_ndcg, ndcg_improvement, kendall_tau.
    """
    import numpy as np
    from scipy import stats as scipy_stats
    from sentence_transformers import CrossEncoder

    if not results:
        return {
            "scores": [], "original_ndcg": 0.0, "reranked_ndcg": 0.0,
            "ndcg_improvement": 0.0, "kendall_tau": 0.0,
        }

    model = CrossEncoder(model_name, device="cuda")

    pairs = []
    for r in results:
        doc_text = r.get("content") or r.get("snippet", "")
        if len(doc_text) > 512:
            doc_text = doc_text[:512]
        pairs.append((query_text, doc_text))

    raw_scores = model.predict(pairs)
    if not isinstance(raw_scores, np.ndarray):
        raw_scores = np.array(raw_scores)

    scores = []
    for i, (r, score) in enumerate(zip(results, raw_scores)):
        scores.append({
            "document_title": r.get("title", ""),
            "document_url": r.get("url", ""),
            "original_rank": i,
            "cross_encoder_score": float(score),
        })

    sorted_by_score = sorted(range(len(scores)), key=lambda i: scores[i]["cross_encoder_score"], reverse=True)
    for new_rank, orig_idx in enumerate(sorted_by_score):
        scores[orig_idx]["reranked_position"] = new_rank

    original_relevance = [s["cross_encoder_score"] for s in scores]
    original_ndcg = _compute_ndcg(original_relevance)
    sorted_relevance = sorted(original_relevance, reverse=True)
    reranked_ndcg = _compute_ndcg(sorted_relevance)

    original_ranks = list(range(len(scores)))
    reranked_ranks = [s["reranked_position"] for s in scores]
    if len(scores) >= 2:
        tau, _ = scipy_stats.kendalltau(original_ranks, reranked_ranks)
        kendall_tau = float(tau) if not math.isnan(tau) else 0.0
    else:
        kendall_tau = 1.0

    return {
        "scores": scores,
        "original_ndcg": original_ndcg,
        "reranked_ndcg": reranked_ndcg,
        "ndcg_improvement": reranked_ndcg - original_ndcg,
        "kendall_tau": kendall_tau,
    }


@app.function(
    image=searchprobe_image,
    gpu="T4",
    volumes={"/models": model_volume},
    timeout=600,
)
def batch_validate(
    items: list[dict[str, Any]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
) -> list[dict[str, Any]]:
    """Batch cross-encoder validation.

    Args:
        items: List of dicts with query_text, results, category, provider
        model_name: Cross-encoder model name

    Returns:
        List of validation result dicts.
    """
    import numpy as np
    from scipy import stats as scipy_stats
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device="cuda")
    output = []

    for item in items:
        query_text = item["query_text"]
        results = item.get("results", [])

        if not results:
            output.append({
                "query_text": query_text, "category": item.get("category", ""),
                "provider": item.get("provider", ""), "scores": [],
                "original_ndcg": 0.0, "reranked_ndcg": 0.0,
                "ndcg_improvement": 0.0, "kendall_tau": 0.0,
            })
            continue

        pairs = []
        for r in results:
            doc_text = r.get("content") or r.get("snippet", "")
            if len(doc_text) > 512:
                doc_text = doc_text[:512]
            pairs.append((query_text, doc_text))

        raw_scores = model.predict(pairs)
        if not isinstance(raw_scores, np.ndarray):
            raw_scores = np.array(raw_scores)

        scores = []
        for i, (r, score) in enumerate(zip(results, raw_scores)):
            scores.append({
                "document_title": r.get("title", ""),
                "original_rank": i,
                "cross_encoder_score": float(score),
            })

        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i]["cross_encoder_score"], reverse=True)
        for new_rank, orig_idx in enumerate(sorted_idx):
            scores[orig_idx]["reranked_position"] = new_rank

        relevance = [s["cross_encoder_score"] for s in scores]
        orig_ndcg = _compute_ndcg(relevance)
        reranked_ndcg = _compute_ndcg(sorted(relevance, reverse=True))

        original_ranks = list(range(len(scores)))
        reranked_ranks = [s["reranked_position"] for s in scores]
        if len(scores) >= 2:
            tau, _ = scipy_stats.kendalltau(original_ranks, reranked_ranks)
            kendall_tau = float(tau) if not math.isnan(tau) else 0.0
        else:
            kendall_tau = 1.0

        output.append({
            "query_text": query_text,
            "category": item.get("category", ""),
            "provider": item.get("provider", ""),
            "scores": scores,
            "original_ndcg": orig_ndcg,
            "reranked_ndcg": reranked_ndcg,
            "ndcg_improvement": reranked_ndcg - orig_ndcg,
            "kendall_tau": kendall_tau,
        })

    return output


@app.function(
    image=searchprobe_image,
    gpu="T4",
    volumes={"/models": model_volume},
    timeout=300,
)
def encode_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> list[list[float]]:
    """Generate embeddings for a list of texts.

    Args:
        texts: Texts to encode
        model_name: Sentence-transformer model name

    Returns:
        List of embedding vectors (as nested lists of floats).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cuda")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


@app.function(
    image=searchprobe_image,
    gpu="T4",
    volumes={"/models": model_volume},
    timeout=300,
)
def compute_fitness_batch(
    queries: list[str],
    search_results: list[list[dict[str, str]]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
) -> list[dict[str, Any]]:
    """Compute adversarial fitness scores using cross-encoder NDCG gap.

    Args:
        queries: List of adversarial query strings
        search_results: List of result lists (one per query)
        model_name: Cross-encoder model name

    Returns:
        List of dicts with fitness, ndcg_improvement, original_ndcg, reranked_ndcg.
    """
    import numpy as np
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device="cuda")
    output = []

    for query, results in zip(queries, search_results):
        if not results:
            output.append({"fitness": 0.5, "ndcg_improvement": 0.0,
                           "original_ndcg": 0.0, "reranked_ndcg": 0.0})
            continue

        pairs = []
        for r in results:
            doc_text = r.get("content") or r.get("snippet", "")
            if len(doc_text) > 512:
                doc_text = doc_text[:512]
            pairs.append((query, doc_text))

        raw_scores = model.predict(pairs)
        if not isinstance(raw_scores, np.ndarray):
            raw_scores = np.array(raw_scores)

        relevance = [float(s) for s in raw_scores]
        orig_ndcg = _compute_ndcg(relevance)
        reranked_ndcg = _compute_ndcg(sorted(relevance, reverse=True))
        improvement = reranked_ndcg - orig_ndcg

        output.append({
            "fitness": min(1.0, improvement * 2),
            "ndcg_improvement": improvement,
            "original_ndcg": orig_ndcg,
            "reranked_ndcg": reranked_ndcg,
        })

    return output


# ---------------------------------------------------------------------------
# Local Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    command: str = "geometry",
    models: str = "all-MiniLM-L6-v2",
    categories: str = "",
    output: str = "results.json",
):
    """Run SearchProbe analysis on Modal GPUs.

    Commands: geometry, validate, encode

    Args:
        command: Analysis command to run
        models: Comma-separated model names
        categories: Comma-separated categories (empty = all)
        output: Output JSON file path
    """
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    cat_list = [c.strip() for c in categories.split(",") if c.strip()] or None

    if command == "geometry":
        print(f"Running geometry analysis with models={model_list}, categories={cat_list or 'all'}")
        result = analyze_geometry.remote(models=model_list, categories=cat_list)

        # Print summary
        vuln = result.get("vulnerability_matrix", {})
        for model_name, cats in vuln.items():
            print(f"\n{model_name}:")
            sorted_cats = sorted(cats.items(), key=lambda x: x[1], reverse=True)
            for cat, score in sorted_cats[:5]:
                severity = "CRITICAL" if score >= 0.8 else "HIGH" if score >= 0.6 else "MODERATE" if score >= 0.4 else "LOW"
                print(f"  {cat:30s} {score:.3f} ({severity})")

        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {output}")

    elif command == "encode":
        sample_texts = ["companies in AI", "companies NOT in AI", "AI startups"]
        print(f"Encoding {len(sample_texts)} texts with {model_list[0]}")
        embeddings = encode_texts.remote(texts=sample_texts, model_name=model_list[0])
        print(f"Got {len(embeddings)} embeddings, each {len(embeddings[0])} dimensions")

        with open(output, "w") as f:
            json.dump({"texts": sample_texts, "embeddings": embeddings}, f)
        print(f"Saved to {output}")

    else:
        print(f"Unknown command: {command}")
        print("Available commands: geometry, encode")
