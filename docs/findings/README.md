# Findings

Results produced with this repository on 2026-10-14. Each file states its inputs;
everything here can be regenerated with the commands shown.

## Triplet probe v1 (`probe_open_models.md`, `probe_bm25.md`)

**Question.** For a Wikidata-backed case such as *"European countries that are not members of
the European Union"*, does a retriever score the Wikipedia summary of a correct answer
(Norway) above the summary of a violating near-miss (France)?

**Data.** `triplets_v1.jsonl`: 203 triplets from the answer/negative sets of
`data/cases/v1/wikidata.jsonl` (`searchprobe explain triplets --no-judgments --per-case 4`).

**Command.**
```bash
searchprobe explain probe --triplets docs/findings/triplets_v1.jsonl \
  --scorers "bm25,st:sentence-transformers/all-MiniLM-L6-v2,st:BAAI/bge-small-en-v1.5,\
st:intfloat/e5-base-v2,st:Qwen/Qwen3-Embedding-0.6B,ce:cross-encoder/ms-marco-MiniLM-L6-v2,\
ce:BAAI/bge-reranker-base"
```

**What it shows.**
- No retriever prefers the satisfying document more often than chance: 48-53% overall
  pairwise accuracy (95% CIs about ±7 points). Bi-encoders and cross-encoders are
  indistinguishable here.
- On "members of X but not of Y" set differences (boolean_logic) every model is *below*
  chance (16-34%, n = 32): the violating document mentions both organizations, and all
  models reward that.
- Negation ("European countries that are not members of the EU"): 46-55% for neural
  models, 33% for BM25 (n = 94).
- BM25 is below chance overall (40%), as expected for a scorer that rewards term overlap
  with the query's excluded concept.
- Numeric ranges and thresholds: 48-65%; multi-constraint cases: 57-74%; temporal cases:
  45-82% (n = 11). These are the only groups where some models are above chance, and the
  CIs are wide.

**Correction.** An earlier version of this file (commit `f9a6a37`, 132 triplets) reported a
70% negation accuracy for the BGE reranker. That dataset had Wikidata membership errors
(former members such as the UK counted as EU members, a mislabeled organization id,
Denmark and the Netherlands missing). On the corrected and larger dataset the effect does
not replicate.

**Caveats.** Small n in some categories (temporal n = 11; entity and compositional
n = 1-2, so those rows say nothing); summaries of different
entities are not minimal pairs, so this measures whether the operator information changes the
ranking, not pure negation sensitivity. Judged-pool triplets (real engine results) are added
with `searchprobe explain triplets` after a `searchprobe judge` run.
