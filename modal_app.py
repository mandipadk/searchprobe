"""SearchProbe on Modal GPUs: thin wrappers around library code (no duplicated logic).

Every function imports and calls ``searchprobe`` itself, so results on Modal are
identical to local results by construction.

Run with:
    modal deploy modal_app.py                      # for ModalRunner (from_name lookups)
    modal run modal_app.py --command probe --triplets docs/findings/triplets_v1.jsonl \
        --scorers "st:BAAI/bge-base-en-v1.5,st:Qwen/Qwen3-Embedding-0.6B,ce:BAAI/bge-reranker-v2-m3"
    modal run modal_app.py --command geometry --models "all-MiniLM-L6-v2"
"""

from __future__ import annotations

import json
from typing import Any

import modal

app = modal.App("searchprobe")

searchprobe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["analysis"])
    .env({"HF_HOME": "/models"})
    .add_local_python_source("searchprobe")
)
model_volume = modal.Volume.from_name("searchprobe-models", create_if_missing=True)
GPU_KW: dict[str, Any] = {
    "image": searchprobe_image,
    "gpu": "T4",
    "volumes": {"/models": model_volume},
}


@app.function(timeout=900, **GPU_KW)
def analyze_geometry(
    models: list[str] | None = None, categories: list[str] | None = None
) -> dict[str, Any]:
    """Query-pair geometry report (descriptive; see `probe_triplets` for retrieval evidence)."""
    from searchprobe.geometry.analyzer import EmbeddingGeometryAnalyzer

    report = EmbeddingGeometryAnalyzer(models=models, device="cuda").generate_report(categories)
    return {
        "vulnerability_matrix": report.get_vulnerability_matrix(),
        "profiles": {
            m: {c: p.to_dict() for c, p in cats.items()} for m, cats in report.profiles.items()
        },
    }


@app.function(timeout=1800, **GPU_KW)
def probe_triplets(triplets_jsonl: str, scorers: list[str]) -> dict[str, Any]:
    """Pairwise accuracy of each scorer on (query, d+, d-) triplets."""
    import tempfile

    from searchprobe.explain.probe import probe_markdown, run_probe, summarize_probe
    from searchprobe.explain.scorers import make_scorer
    from searchprobe.explain.triplets import load_triplets

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(triplets_jsonl)
    scores = run_probe(load_triplets(f.name), [make_scorer(s) for s in scorers])
    return {
        "markdown": probe_markdown(scores),
        "summary": [
            {
                "scorer": s.scorer,
                "group": s.group,
                "accuracy": s.accuracy.mean,
                "lower": s.accuracy.lower,
                "upper": s.accuracy.upper,
                "n": s.n,
            }
            for s in summarize_probe(scores)
        ],
    }


@app.function(timeout=300, **GPU_KW)
def validate_cross_encoder(
    query_text: str,
    results: list[dict[str, str]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
) -> dict[str, Any]:
    """Cross-encoder rescoring of one result list (library implementation)."""
    from searchprobe.validation.cross_encoder import CrossEncoderValidator

    v = CrossEncoderValidator(model_name, device="cuda").validate_search_results(
        "remote", query_text, "unknown", "remote", results
    )
    return v.to_dict()  # scores are records (title, url, ranks, score), as before


@app.function(timeout=900, **GPU_KW)
def batch_validate(
    items: list[dict[str, Any]], model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
):
    """Cross-encoder rescoring for many result lists in one container."""
    from searchprobe.validation.cross_encoder import CrossEncoderValidator

    validator = CrossEncoderValidator(model_name, device="cuda")
    out = []
    for item in items:
        v = validator.validate_search_results(
            item.get("query_id", ""),
            item["query_text"],
            item.get("category", "unknown"),
            item.get("provider", "unknown"),
            item.get("results", []),
        )
        out.append({**v.to_dict(), "query_text": item["query_text"]})
    return out


@app.function(timeout=300, **GPU_KW)
def encode_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[list[float]]:
    """Embeddings with the model-appropriate document prompt."""
    from sentence_transformers import SentenceTransformer

    from searchprobe.explain.scorers import prompts_for

    _, doc_prompt = prompts_for(model_name)
    model = SentenceTransformer(model_name, device="cuda", trust_remote_code=True)
    return model.encode([doc_prompt + t for t in texts], normalize_embeddings=True).tolist()


@app.function(timeout=300, **GPU_KW)
def compute_fitness_batch(
    queries: list[str],
    search_results: list[list[dict[str, str]]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
) -> list[dict[str, Any]]:
    """Cross-encoder gap per query (legacy evolution fitness)."""
    from searchprobe.validation.cross_encoder import CrossEncoderValidator

    validator = CrossEncoderValidator(model_name, device="cuda")
    out = []
    for q, results in zip(queries, search_results):
        v = validator.validate_search_results("evo", q, "unknown", "remote", results)
        out.append(
            {
                "query": q,
                "fitness": min(1.0, v.ndcg_improvement * 2),
                "ndcg_improvement": v.ndcg_improvement,
                "original_ndcg": v.original_ndcg,
                "reranked_ndcg": v.reranked_ndcg,
            }
        )
    return out


@app.local_entrypoint()
def main(
    command: str = "probe",
    models: str = "all-MiniLM-L6-v2",
    categories: str = "",
    triplets: str = "docs/findings/triplets_v1.jsonl",
    scorers: str = "bm25,st:BAAI/bge-base-en-v1.5,ce:BAAI/bge-reranker-base",
    output: str = "modal_results.json",
):
    """Commands: probe (triplet probe on GPU), geometry, encode."""
    if command == "probe":
        with open(triplets) as f:
            payload = f.read()
        result = probe_triplets.remote(payload, [s.strip() for s in scorers.split(",") if s])
        print(result["markdown"])
    elif command == "geometry":
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        cat_list = [c.strip() for c in categories.split(",") if c.strip()] or None
        result = analyze_geometry.remote(models=model_list, categories=cat_list)
    elif command == "encode":
        texts = ["companies in AI", "companies NOT in AI", "AI startups"]
        result = {"texts": texts, "embeddings": encode_texts.remote(texts, models.split(",")[0])}
    else:
        raise SystemExit(f"Unknown command {command!r}; use probe, geometry or encode")
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {output}")
