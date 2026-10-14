# Findings

Results produced with this repository on 2026-10-14. Each file states its inputs;
everything here can be regenerated with the commands shown.

## Triplet probe v1 (`probe_open_models.md`, `probe_bm25.md`)

**Question.** For a Wikidata-backed case such as *"European countries that are not members of
the European Union"*, does a retriever score the Wikipedia summary of a correct answer
(Norway) above the summary of a violating near-miss (France)?

**Data.** `triplets_v1.jsonl`: 132 triplets from the answer/negative sets of
`data/cases/v1/wikidata.jsonl` (`searchprobe explain triplets --no-judgments --per-case 4`).

**Command.**
```bash
searchprobe explain probe --triplets docs/findings/triplets_v1.jsonl \
  --scorers "bm25,st:sentence-transformers/all-MiniLM-L6-v2,st:BAAI/bge-small-en-v1.5,\
st:intfloat/e5-base-v2,st:Qwen/Qwen3-Embedding-0.6B,ce:cross-encoder/ms-marco-MiniLM-L6-v2,\
ce:BAAI/bge-reranker-base"
```

**What it shows.**
- Every retriever is close to chance at preferring the satisfying document: 55-62% overall
  pairwise accuracy (95% CIs roughly ±8 points), so no model is clearly better than another
  at this.
- BM25 is *below* chance (43%) and scores 0% on "members of X but not of Y" set
  differences: the violating documents mention both organizations, which is exactly what
  lexical matching rewards.
- Negation: the BGE reranker reaches 70% [57%, 83%]; the 0.6B Qwen3 embedder is at 44%.
  Numeric ranges sit at 39-66% across models, consistent with numbers being read as tokens.
- Entity disambiguation (namesake cities) is 100% for every model, but those triplets are
  easy (the query names the state), so this does not show robustness to entity confusion.

**Caveats.** Small n per category (wide CIs; temporal n = 5); summaries of different
entities are not minimal pairs, so this measures whether the operator information changes the
ranking, not pure negation sensitivity. Judged-pool triplets (real engine results) are added
with `searchprobe explain triplets` after a `searchprobe judge` run.
