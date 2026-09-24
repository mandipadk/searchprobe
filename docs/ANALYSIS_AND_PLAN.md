# SearchProbe — Deep Analysis & 10x Plan

*Written 2026-09-24 against commit `ad021d2`. Every claim below was checked against the code at that commit; the ones marked **(verified)** were also reproduced by running code.*

---

## 0. TL;DR

**What the project is really for:** producing *trustworthy evidence* of where and why neural (embedding-based) search engines fail on queries whose meaning depends on logical structure (negation, numbers, time, conjunction, composition, entity identity, instructions), and ideally showing how to fix it.

**Where it is today:** a wide scaffold built in roughly 8 days (Mar 2–10, 2026): 13 categories, 4 providers, an LLM judge, 4 "research modules", a DAG session layer, a dashboard, reports, Modal/Colab, and a second package (Aris) meant to be the fix. The breadth is impressive. But the **measurement chain, the part that turns a query into a number, is not yet valid.** The numbers it produces today cannot support the conclusions the README draws:

1. **There is effectively no ground truth.** 720 of 740 template queries have none, the other 20 have an *empty* exclusion list, LLM-generated queries only get free-text "manual" hints, and there are 22 hand-written seeds. Where ground truth does exist, **neither the judge nor the verifier can read it**, because of key and schema mismatches (verified). The verifier isn't called from any scoring path anyway, and zero-result searches are never stored or scored.
2. **The unit of judgment is wrong.** One LLM call scores an entire top-10 list with a single number per dimension. That gives no per-document relevance, so no nDCG, no precision@k, no recall and no pooling: none of the standard IR measures are possible.
3. **Providers are judged on different evidence.** Exa and Tavily send the judge up to 1,000 characters of page text; Brave and SerpAPI send a ~160-character snippet. That confounds provider quality with how verbose each API is.
4. **Exa's modes are merged at evaluation time.** The `evaluations` table has no `search_mode` column, so `exa:auto` and `exa:neural` end up in one bucket. That removes the one comparison the project was built around: neural vs. keyword/auto.
5. **Every research module measures something other than what it claims.** Geometry compares query-to-query cosine over 5 pairs per category using 2021-era models. Cross-encoder "validation" is circular and its nDCG math breaks on logits (verified: nDCG = −2.27). Perturbation treats stability as good even for meaning-changing edits, and its RBO gives 0.65 for *identical* lists (verified). Evolution's default fitness is a keyword heuristic that never calls a search engine.
6. **Engineering bugs break guarantees the code advertises:** a $2 budget spent $20 (verified); the "concurrent" judge runs serially (verified); ruff reports 182 errors, so CI lint is red.

**The 10x move:** stop adding surface area and rebuild the core as a **measurement instrument**:

- **verifiable test cases**: structured constraints, minimal-pair twins, and known positive and negative documents, generated from structured data rather than LLM imagination;
- **per-document, per-constraint judging** on uniformly fetched and snapshotted content, producing standard IR metrics with paired statistics and a calibrated judge;
- **metamorphic testing**: q vs. not-q result overlap against a measured noise floor. It needs no ground truth, is cheap and objective, and tests the hypotheses directly;
- **mechanistic explanations tied to actual retrieval**: (query, positive doc, violating doc) triplets on modern open embedders and rerankers, linked per case to provider failures;
- **adversarial discovery with validity constraints**: differential fitness plus diversity, feeding discovered failures back into the dataset as regression tests;
- **a fix layer (Aris, consolidated)**: a constraint-aware verify-and-rerank wrapper around any provider, measured by the same instrument.

The sections below give the evidence, then the plan.

---

## 1. The core problem, restated precisely

"Neural search fails on negation" is not a finding until four questions are answered:

| Question | What it requires |
|---|---|
| **Did it fail?** | For each returned document, a *verifiable* judgment of whether it satisfies each constraint in the query. |
| **How badly, compared with what?** | Rank-aware metrics (violation rate@k, nDCG@k, P@k), paired comparisons over the *same* queries, CIs, a baseline (keyword engine, provider noise floor). |
| **Why?** | A mechanistic test that predicts the failure *per case*: the retriever's representation can't separate a satisfying document from a violating near-duplicate. |
| **Can it be fixed, at what cost?** | An intervention measured on the same instrument (quality gained vs. latency and $). |

The current code has a component with the right *name* for each row, but none of them computes the quantity the row requires. Everything in the plan is organized around making each row true.

---

## 2. How the system works today (actual data flow)

```
generate ─► queries (seed 22 | template 740 | LLM N)  ──►  SQLite queries.ground_truth (mostly empty/"manual")
run      ─► for each query × provider(× exa mode): provider.search() ─► search_results (JSON blob of top-k)
evaluate ─► for each search_result: ONE judge call over the whole top-10 ─► evaluations (no search_mode!)
report / dashboard ─► mean of weighted_score by provider & category (+ CIs)

research (independent, not linked per-query to the above):
  geometry  ─► cos(query_a, query_b) on hard-coded pairs, 2 MiniLM/MPNet models
  validate  ─► rerank provider top-10 with ms-marco MiniLM CE, "nDCG improvement"
  perturb   ─► word delete/swap/synonym, Jaccard/RBO between original and perturbed results
  evolve    ─► GA over query strings, fitness = heuristic | 1 − judge | CE gap
session (intelligence/) ─► DAG that chains the above and "correlates" them
aris/     ─► separate engine prototype (see §3.7)
```

---

## 3. Findings

Severity: 🔴 invalidates results · 🟠 materially biases results · 🟡 quality/engineering.

### 3.1 Ground truth: the foundation is missing

- 🔴 **Templates have no usable ground truth (verified).** `generate_all(max_per_template=100)` produces 740 queries: **720 have `ground_truth=None`**, and 20 have `must_not_contain` with an **empty keyword list** (`queries/templates.py:42`: `must_not_contain_keywords=[]`).
- 🔴 **Some templates contradict themselves (verified).** 27 cross-product queries such as *"buildings between 200 and 150 meters tall"* (`templates.py`, the `{min_val}/{max_val}` slots are sampled independently). Units are also mixed without checks, e.g. "between 500 and 600 stories".
- 🔴 **LLM-generated queries only get free-text hints** (`queries/generator.py:144-147`: `GroundTruth(strategy="manual", manual_judgment=...)`), which nothing ever checks.
- 🟠 **The seeds are few and sometimes wrong.** There are 22 in total, and 5 categories have just one (`queries/seeds.py`). Examples:
  - *"Programming languages NOT object-oriented"* uses `must_contain_keywords=["C", ...]`. Under substring matching, "C" matches almost any text.
  - `must_not_contain=["Python", ...]` wrongly fails a page that says "unlike Python, Haskell is purely functional".
  - *"Companies NOT in AI"* uses `must_not_contain=["AI", ...]`. Under substring matching this fires on "s**ai**d", "m**ai**ntain", "em**ai**l". **Verified:** a plumbing company page fails the check.
  - Temporal seeds hard-code 2025 dates, while templates use "the last 30 days" and "yesterday", which can't have fixed ground truth at all.
- 🔴 **The judge can't see the ground truth.** `evaluation/prompts.py:170-175` reads `ground_truth.get('should_match')` and `.get('should_not_match')`. Those keys don't exist on `GroundTruth` (`queries/models.py`), so only `expected_entity` ever reaches the judge.
- 🔴 **The verifier can't read the ground truth either (verified).** `GroundTruthEngine.validate` (`intelligence/ground_truth.py:244-273`) expects `{"must_not_contain": {"terms": [...]}}`. Stored ground truth looks like `{"strategy": "must_not_contain", "must_not_contain_keywords": [...]}`, so passing it in returns **zero outcomes**, and `compute_objective_score` then returns 0.0.
- 🟠 **The verifier is naive even when fed correctly.**
  - Pure substring matching.
  - `NumericRangeStrategy` passes a document if **any** number anywhere in it falls in range: years, prices and footnotes all count (`ground_truth.py:115-149`).
  - `date_range`, `language_match` and `type_match` are declared in the schema (`queries/models.py:18-27`) but have **no implementation**.
  - Snippet-only providers present far less text, so they are systematically *less* likely to trip a `must_not_contain`. That biases negation scores in favor of Brave and SerpAPI.

### 3.2 The LLM judge: wrong unit, weak protocol

- 🔴 **One score for a whole list.** `SearchJudge.evaluate` (`evaluation/judge.py:112-218`) sends the top 10 in a single prompt and gets back one score per dimension. Consequences:
  - no per-document labels, so no nDCG/P@k/MRR, no pooling across providers, no recall;
  - the score is position-insensitive: a perfect #1 and a perfect #10 count the same;
  - judgments can't be reused across providers that return the same URL, so cost scales with providers × modes rather than with unique documents.
- 🟠 **Uneven evidence (see §3.4).** The judge sees page text for some providers and snippets for others.
- 🟠 **Configuration is ignored.** `Settings.judge_model` and `judge_temperature` (`config/settings.py:49-50`) are never passed (`SearchJudge(settings)` uses the hard-coded default `model="claude-sonnet-4-6"` and no temperature). `EvaluationResult.to_dict()` doesn't include the judge model, so the `evaluations.judge_model` column is always NULL. Results can't be traced to the judge that produced them.
- 🟠 **Scores lack structure.**
  - Free-form JSON is parsed with regex, and a parse failure silently becomes `failure_modes=["parse_error"]` with score 0.0.
  - `failure_modes` are free text, so frequency counts fragment ("negation_ignored" vs "ignores negation").
  - Tool use / structured outputs, or at least schema validation, would fix both.
- 🟠 **The judge is never checked for reliability.** There is no human-labeled calibration set, no inter-judge agreement, no repeat-sampling variance, and no position-bias control (results are always shown in provider order). The published LLM-relevance-judging literature (Thomas et al. 2023; UMBRELA, 2024) makes calibration against human labels the price of admission.
- 🟡 **Blocking calls in async code (verified).** `evaluate` is `async` but calls the synchronous `anthropic.Anthropic().messages.create`. **Five evaluations with `max_concurrent=5` took 1.50 s instead of 0.30 s**, so the "concurrent evaluation" commit delivers no concurrency.

### 3.3 Storage and statistics: comparisons are distorted

- 🔴 **Exa modes merge (verified by schema).** `evaluations` has no `search_mode` column (`storage/database.py:71-83`), and `add_evaluation` drops the field (`:438-470`). Everything reading evaluations back (`get_evaluations`, `materialize_aggregate_scores`, the dashboard) merges all Exa modes into "exa". `skip_evaluated` also matches on (run, query, provider), so evaluating one Exa mode marks the others as done on the next incremental run.
- 🟠 **Failures count as zero quality.** Search errors, empty results and judge parse errors are all stored with `weighted_score=0.0` (`judge.py:113-127, 202-218`) and averaged in. An API outage lowers a provider's "quality".
- 🟠 **Paired data is analyzed as unpaired.** Every provider runs the same queries, yet the CLI calls `compare_providers(..., paired=False)` (`cli/evaluate.py:289-290`), which throws away most of the statistical power. The unpaired branch also **invents** a CI of `mean_diff ± 0.1` (`statistics.py:289-295`). The paired branch pairs by list order (`zip`), not by `query_id`.
- 🟡 **No multiple-comparison correction in practice.** `benjamini_hochberg` exists but isn't used on the 13 categories × provider-pair comparisons. t-intervals on bounded, often bimodal scores (lots of 0s) are clipped to [0, 1]; bootstrap exists but isn't the default.
- 🟡 **No noise floor.** Nothing measures how much results for the *same* query vary between two calls, so every stability and difference number has an unknown noise baseline.

### 3.4 Providers: uneven evidence, mislabeled modes

- 🟠 **Evidence asymmetry.** Exa returns page `text` (`providers/exa.py:59-66`) and Tavily returns `raw_content` (`tavily.py:57-59`). Brave (`brave.py:82-86`) and SerpAPI (`serpapi.py:83-89`) use the *snippet* as "content". The judge truncates to 1,000 characters, so its evidence ranges from a full page to one sentence depending on provider.
- 🟠 **Invented scores.** Brave and SerpAPI "scores" are fabricated from position (`1.0 - i/n*0.5`). Anything that consumes `score` treats them as real.
- 🟠 **Mislabeled mode.** Exa `"deep"` is silently sent as `"auto"` (`exa.py:57`), so a "deep" row in any table is actually auto. A keyword or keyword-like mode isn't exercised, even though the README frames the project as neural-vs-keyword.
- 🟡 **Blocking SDKs in the default executor.** Exa, Tavily and SerpAPI SDKs are synchronous and run through `run_in_executor(None, …)` on the default pool. That's fine, but perturbation and evolution call the **raw** provider with no rate limiter, retry or cost tracking (`cli/perturb.py`, `cli/evolve.py`).

### 3.5 Pipeline and engineering

- 🔴 **The budget isn't enforced (verified).** `BenchmarkRunner.run` checks the budget *before* the first `await` in every task, and `asyncio.gather` starts all tasks at once, so every task passes the check before any cost is recorded. **A 20-query run with a $2 budget and $1/query spent $20.** Evolution never enforces its `budget_limit` at all, and `total_cost` is hard-coded to 0.0 (`adversarial/optimizer.py:242`).
- 🔴 **Zero-result searches are dropped** (`runner.py:181`), and any other task exception is swallowed without a row (see §3.7).
- 🟡 If the budget *is* exceeded, the whole gather raises and `complete_run` is never called, leaving a half-finished run.
- 🟡 **The CI lint job is red:** `ruff check src/searchprobe` reports **182 errors**. Unit and integration tests pass (200 tests), but none cover geometry metrics, RBO, nDCG or the judge prompt contract, which is where the verified bugs are.
- 🟡 Two databases (`storage/database.py` and `storage/async_db.py`), two configs (`config/settings.py` and `core/config.py`), and a 700-line `modal_app.py` that re-implements library logic.

### 3.6 Research modules: each measures the wrong quantity

**Embedding geometry** (`geometry/`)
- 🔴 **Wrong quantity.** It measures cos(query_a, query_b). Retrieval failure is about cos(query, satisfying doc) vs. cos(query, violating doc). Two similar queries don't imply a retrieval failure, and dissimilar queries don't rule one out. The hypothesis "negation collapses" has to be tested as *"does the embedder prefer the non-AI company's page to the AI company's page for 'companies not in AI'?"*
- 🔴 **Tiny samples, no uncertainty.** 5–6 adversarial pairs and **2 baseline pairs** per category, with no CIs.
- 🔴 **The LID and isotropy numbers are artifacts of sample size.** Each category embeds 10–12 texts. The covariance of n points has at most n−1 nonzero eigenvalues, so the normalized entropy "isotropy" is bounded by log(n−1)/log(d), about **0.40** for n=11, d=384, whatever the geometry. MLE-LID with k=10 on 11 points is similarly degenerate. The differences between categories reflect how many pairs each has.
- 🟠 **Uncalibrated thresholds.** The "calibrated thresholds" (`vulnerability.py:10-24`) and the weights 0.4/0.35/0.25 are hand-picked. Absolute cosine levels vary a lot between models (anisotropy), so a fixed 0.85 threshold means different things per model.
- 🟠 **Models unrelated to the providers.** `all-MiniLM-L6-v2` and `all-mpnet-base-v2` (2021, symmetric, no query/passage prompts) say little about modern retrieval embedders or about Exa. The README claim that *"ANY embedding-based retrieval system will fail"* doesn't follow from this evidence.
- 🟠 **Pair semantics are inconsistent.** For polysemy and entity disambiguation, the "adversarial" pairs already include the disambiguator ("Amazon online shopping" vs "Amazon river ecosystem"), so *low* similarity is the correct behavior, yet those categories are scored with the same "higher sim = more vulnerable" logic.

**Cross-encoder validation** (`validation/`)
- 🔴 **Circular.** The "optimal" ranking is the cross-encoder's own ranking, so `reranked_ndcg` is 1.0 by construction and "improvement" means "how much the provider disagrees with a 2021 MS MARCO MiniLM". That is agreement between two rankers, not a quality gap. It also can't detect the real embedding failure: relevant documents missing from the top 10 entirely (recall).
- 🔴 **Broken nDCG math (verified).** The model config sets `sbert_ce_default_activation_function: Identity`, so `predict` returns raw logits. `_compute_ndcg` uses them directly as gains: `[-3, 5, -8, 2]` gives **nDCG = −2.27** and `[-2, -1]` gives **1.16**.
- 🟠 MS MARCO MiniLM cross-encoders are known to be poor on negation (NevIR, 2023), so they're a weak oracle for exactly these categories. Documents are also truncated to 512 characters, which for Brave/SerpAPI means the snippet.

**Perturbation** (`perturbation/`)
- 🔴 **It measures the opposite of the hypothesis.** The adversarial hypothesis is that engines are *insensitive* to meaning-changing edits. If results for "companies in AI" and "companies NOT in AI" overlap heavily, that **is** the negation failure. The engine averages Jaccard over all operators and reports it as "stability" (higher is better). Meaning-preserving edits (synonyms, typos) should be invariant; meaning-changing edits (negation, antonyms, numbers, dates, argument swaps) should change the results. This is CheckList's INV/DIR distinction (Ribeiro et al. 2020), and it's the missing idea.
- 🔴 **Broken RBO (verified).** Truncated RBO without extrapolation gives **0.651 for two identical top-10 lists**.
- 🟠 **Incomplete sensitivity maps.** `word_delete` stops after `max_variants=5`, so words past position 5 are never deleted and show as sensitivity 0 ("not load-bearing"). Repeated words collide in the dict. Scores are max-normalized, so they're only relative.
- 🟠 **Unusable synonyms, irreproducible swaps.** `synonym_replace` takes `list(set(...))[0]` from WordNet: non-deterministic order, no part-of-speech or sense filter (e.g. WordNet's "ai" is the three-toed sloth). `word_swap` uses unseeded `random`.
- 🟡 The default operators exclude the negation operators, the ones that matter most. There is no URL normalization and no noise-floor re-run.

**Evolutionary optimizer** (`adversarial/`)
- 🔴 **The default fitness never queries a search engine.** `embedding_sim` is a keyword heuristic (+0.2 for "not", +0.15 for digits, +0.15 for length >10, +0.1 × mutation depth…; `fitness.py:144-191`). Evolution under it drifts toward long, negation-stuffed word salad, whatever any engine actually does.
- 🔴 **The `cross_encoder` mode always returns 0.0.** The CLI never passes a `cross_encoder` to `FitnessEvaluator` (`cli/evolve.py:122-126`), and `_evaluate_cross_encoder` returns 0.0 when it is None.
- 🔴 **The `llm_judge` fitness rewards unanswerable queries.** "1 − judge score" can't tell *"the engine failed"* from *"no such document exists / the query is nonsense"*. Without a validity and answerability constraint, the optimizer finds impossible queries rather than failure modes. A single noisy judge sample plus elitism also produces winner's-curse elites.
- 🟠 **Seed categories are assigned at random.** `random.choice(target_categories)` (`optimizer.py:62-66`): a negation seed can be judged with numeric-precision weights. With no categories given, the category is "" and default weights apply.
- 🟠 **Broken bandit.**
  - The "improvement" reward is measured against a *freshly constructed* child whose fitness is 0.0, so reward is absolute fitness, not improvement.
  - Bandit counts only update after each generation, and `select()` returns the first untried operator, so **each of the first K generations uses a single operator for all offspring**.
  - There's no deduplication or fitness cache, and the "diversity-aware survivor selection" in the docstring isn't implemented.

### 3.7 Intelligence / session / reporting layer
- 🔴 **The ground-truth verifier is dead code.** `GroundTruthEngine` is imported only by its unit test. No CLI, judge, runner or adapter calls it. Combined with §3.1, **no ground truth influences any score anywhere** except `expected_entity` in the judge prompt.
- 🔴 **Zero-result searches disappear.** The runner stores a response only `if response.success or response.error` (`pipeline/runner.py:181`), and `success` requires ≥1 result. An empty result set with no error is never stored or judged, which is survivorship bias in favor of engines that return nothing.
- 🔴 **The session "DAG" is mostly for show** (`intelligence/session.py`).
  - The edges carry no data: perturbation and validation read `context.metadata["queries"]`, which nothing sets, so both fall back to 2 hard-coded example queries per category.
  - Geometry guidance for evolution never applies, because every shipped profile sets `categories`.
  - The `ResilientProvider` is built and then discarded (adapters get `provider.provider`, the raw provider), so there's no retry, rate limiting or cost tracking.
  - Only `providers[0]` is used. Most profile fields (`exa_modes`, `judge_model`, `budget_limit`, …) are never read.
  - Nothing is persisted to the database. `SignalBus` has no subscribers.
  - The "parallel" stages run synchronous CPU work inside `async`, so they block each other.
- 🔴 **The correlation statistics are noise** (`intelligence/correlation.py`).
  - Spearman correlations run across 3–13 category points; p-values are computed and thrown away, and |ρ| > 0.5 counts as "notable" across 10 dimension pairs.
  - Simulated false-positive rate for independent variables: **42% at n=4** (the size of `full_test.toml`), 7.8% at n=13 per pair.
  - `vulnerability_score` is overwritten per model (the last model wins). `evaluation_score` is never populated. `embedding_gap` inherits the broken nDCG. The risk score can go negative.
- 🟠 **Failure taxonomy is substring heuristics over LLM free text** (`intelligence/taxonomy.py`): "not" → negation_collapse, "time" matches "sometimes", and the category boost alone assigns a mode. It isn't used by the report or dashboard anyway; they count raw strings, so "unique failure types" ≈ total.
- 🟠 **Resilience layer.**
  - `ResilientProvider` retries non-transient errors, e.g. it retried a 401 three times.
  - Retries happen *after* the single rate-limit token is taken (3 HTTP calls per token), while holding the runner semaphore.
  - HALF_OPEN admits every caller, not one.
  - `acquire(tokens > burst)` loops forever.
  - Exa classifies any error containing "rate" (gene**rate**, accu**rate**) as a rate limit.
- 🟠 **The dashboard and report show wrong or pooled numbers.**
  - DB-backed views key providers as `exa:None` or `exa:default`, pooling all modes.
  - The Provider Comparison page matches `exa:default` against `exa:None`, so its head-to-head view is always empty.
  - The Category Deep Dive page crashes for every category (`metadata.get` on a pydantic model, `dashboard/pages/category_deep_dive.py:37-42`).
  - The Validation page shows category "unknown".
  - The geometry heatmap ignores the selected run and shows the *oldest* value per cell.
  - Materialized aggregates use an unclipped normal CI (e.g. [−0.05, 1.05] at n=2).
  - The "failure rate" can exceed 100%.
  - Charts fill missing cells with 0.
- 🟡 **CLI:** `searchprobe report list-runs` runs the report generator first (the callback ignores `invoked_subcommand`). The judge is called without `temperature`, so the API default of 1.0 applies despite `judge_temperature=0.0` in the settings.
- 🟡 **Duplication that has already drifted.**
  - `modal_app.py` copies pairs, thresholds and scoring from the library, but defines baseline pairs for only 5 of 13 categories, so Modal vulnerability scores differ from local ones.
  - `AsyncDatabase` is never imported and would crash on a fresh DB, since it has no schema creation.
  - The notebook's perturbation cell passes `{"query": …}` where the library reads `["text"]`.

### 3.8 Aris (the "fix" engine)
Aris is ~4.6k lines in 76 files from a single commit, with **no tests**. The pipeline *shape* is right: understand → retrieve with a positive query → verify each constraint → rerank. The logic that actually executes, though, is about 400 lines of regex heuristics and a fixed-weight blend, and on the benchmark's own queries those heuristics are often wrong.

- 🔴 **It returns nothing out of the box.** The DuckDuckGo source imports `AsyncDDGS` (`aris/sources/duckduckgo.py:19`), which isn't exported by the pinned `duckduckgo-search>=7.0`. The error is swallowed and zero results come back. Without chromadb installed, every search path crashes (`index/dense_store.py`). The parser reads only `ARIS_ANTHROPIC_API_KEY` and passes `api_key=""` explicitly, so users who have only `ANTHROPIC_API_KEY` silently fall back to regex parsing.
- 🔴 **Negation parsing mangles scope (spot-checked).** `detect_negations("Python web frameworks NOT Django NOT Flask")` → `['Django NOT Flask']`, a single term that never matches anything. `"libraries that do not use PyTorch"` → the positive query becomes `"Python libraries that do"`. `generate_positive_query` uses `str.replace`, so negating "R" deletes every capital R.
- 🔴 **The negation filter confuses mentioning a term with having the property.** For "…NOT relational" it drops *"MongoDB is a non-relational database"*, exactly the document you want. For "NOT Java" it drops Scala pages that mention Java.
- 🔴 **Numeric and temporal extraction is fragile.**
  - "over 1000", "under $1000", "10k", "10 million" all extract nothing.
  - `"what is this about, really"` raises `ValueError` and crashes the whole search (spot-checked).
  - "after 2023" becomes ≥ 2023-01-01, off by one year.
  - The verifiers accept *any* number or year anywhere in a ~200-character snippet: "founded in 2015 … 50 employees" satisfies "> 1000 employees".
  - `published_date` is never populated, so the date-first logic never runs.
- 🔴 **Entity handling is wrong on its own benchmark queries.** A 7-entry substring dictionary maps "javascript" → java and "trust fund" → rust. "Michael Jordan machine learning papers" resolves to the **basketball player**. The entity verifier can never return *violated*.
- 🟠 **Much of the pipeline is dead or inert:** HyDE (never instantiated), decomposition (local index only), structured/BM25 retrieval (always empty), boolean logic (`verify_boolean_and` imported but never called; there is no OR/precedence parser), `verification_strategies` (computed, never read), the document cache, and ingestion.
  - The agent loop usually runs all 3 iterations, re-issuing nearly the same web query and tripling cost.
  - In fallback mode the loop can put negated terms *back into* the query.
- 🟠 **The Aris "benchmark" measures nothing.** It runs 16 hand-written queries against Aris alone, and "success" means *returned ≥1 result*. The comparison provider path imports a function that doesn't exist, and Aris isn't registered in searchprobe's `ProviderRegistry`, so the real judge can't evaluate it.
- 🟡 **Security:** the unauthenticated `/api/v1/index` route fetches arbitrary URLs (SSRF), and CORS is set to `*` with credentials.

**Worth keeping:** the pipeline shape, LLM-produced structured constraints (with a real schema), three-valued verification (satisfied/violated/unknown), preferring `published_date`, RRF/MMR, and source retry and circuit breaking. **Minimal core that would move the needle:** the constraint-aware verify-and-rerank wrapper in Phase 6: an LLM parser as the single source of constraints, fetched full text, a per-constraint LLM verifier with evidence, and evaluation through searchprobe's own judge on the same cases.

---

## 4. What "10x" means here

10x doesn't mean more categories, providers, dashboards or modules; the project already has plenty of each. It means **every number the tool prints is something you'd defend in a paper or a vendor meeting**, and the tool can go from *"this query failed"* to *"this is the mechanism, here is the fix, and here is how much it helps"*.

Design principles for v3:

1. **The dataset is the product.** Test cases are verifiable *by construction*: typed constraints, known answer sets where possible, minimal-pair twins.
2. **Judge documents, not lists, and judge constraints, not vibes.** Every (case, document) pair gets per-constraint verdicts with evidence quotes and a graded relevance label.
3. **Same evidence for every provider.** The harness fetches and extracts page content itself, once per URL, as a content-addressed snapshot.
4. **Paired, cached, versioned, reproducible.** Judgments are keyed by (case, content hash, judge, prompt version). Runs carry a manifest (git SHA, dataset version, judge, SDK versions).
5. **Every metric has a baseline and a CI:** the provider's noise floor, a keyword baseline, and a paired bootstrap over cases.
6. **Explanations must predict per-case failures,** not correlate 13 category means.
7. **Discovery produces regression tests.**
8. **Fixes are measured on the same instrument,** as just another "system".

## 5. Target core architecture

```
TestCase v2
  text, category, operators[]
  constraints[]  (typed)   Exclude(concept|entity) · NumericCmp(attr, op, value, unit)
                           DateRange(published|event, lo, hi) · EntityIs(wikidata QID)
                           Relation(subj, pred, obj, directed) · DocType · Language · HasAll(...)
  twins{op → case_id}      minimal pairs: negate · antonym flip · number ×10 · date ±1y · argument swap
  answer_set / negative_set (optional, e.g. Wikidata QIDs)   known_pos/neg_urls (optional)
  provenance (generator, SPARQL/template/LLM, version) · validation (gold|auto|none)

System      = provider + mode (+ optional wrapper such as Aris)       ← the unit of comparison
Retrieval   System.search(case) → ResultList(url, rank, native score if real)
Evidence    fetch(url) → Snapshot(sha256, main text, lang, published_date)   one extractor, cached
Judgment    (case, snapshot) → per-constraint {satisfied | violated | unaddressed, quote} + relevance 0–3
            deterministic checkers first (numbers, dates, language, QID membership) → LLM for the rest
Metrics     violation@k · all-constraints-satisfied@k · nDCG@10 (graded, pooled) · P@k · first-hit rank
            metamorphic: Operator Blindness Index vs. noise floor
Stats       paired bootstrap over cases · BH across categories · effect sizes · power
Explain     (q, d+, d−) triplet probes on open embedders/rerankers → per-case failure prediction
Discover    structured generation + differential fitness + MAP-Elites → new TestCases
Intervene   constraint-aware verify & rerank wrapper → evaluated as a System
```

## 6. The plan

Phases are ordered by dependency: each one makes the next one meaningful. Rough effort assumes one focused developer working with Claude.

### Phase 0 — Make the current numbers honest (1–3 days)
Small, surgical PRs, each with a regression test that reproduces the bug first:

| # | Fix | Where |
|---|---|---|
| 0.1 | Add `search_mode` to `evaluations` (migration), include it in `skip_evaluated`, and group all stats/aggregates/dashboard by a single `system_key = provider:mode` helper | `storage/database.py`, `async_db.py`, `evaluation/statistics.py`, dashboard |
| 0.2 | Add a `status` field (ok / search_error / no_results / judge_error). **Store zero-result responses** (they are real failures). Exclude errors from quality means and report error rates separately | `judge.py`, `database.py`, stats |
| 0.3 | Judge: use `AsyncAnthropic` (or `asyncio.to_thread`); honor `settings.judge_model/temperature`; store `judge_model` + `prompt_version`; structured output via tool use + schema validation; pass real ground-truth fields | `evaluation/judge.py`, `prompts.py`, `config/__init__.py` |
| 0.4 | Budget: check-and-reserve the estimated cost under a lock *before* dispatch; on exhaustion cancel remaining tasks and still `complete_run(status="partial")`. Same ledger for perturb/evolve | `pipeline/runner.py`, `cost_tracker.py`, `cli/perturb.py`, `cli/evolve.py` |
| 0.5 | Stats: pair by `query_id`; delete the fake ±0.1 CI; paired bootstrap as default; BH across category×pair tests | `evaluation/statistics.py`, `cli/evaluate.py` |
| 0.6 | RBO → extrapolated RBO (Webber et al. 2010, eq. 32); cross-encoder scores → sigmoid before any gain math | `perturbation/stability.py`, `validation/cross_encoder.py` |
| 0.7 | Evolution: pass the cross-encoder; carry the seed's true category; bandit reward = child − parent fitness; update counts per selection; fitness cache; enforce budget | `adversarial/*`, `cli/evolve.py` |
| 0.8 | Perturbation: delete every word; seeded RNG; URL normalization; include negation operators by default | `perturbation/*` |
| 0.9 | Templates: constrain `min<max`, drop or fill empty ground truth. Seeds: fix the wrong keywords ("C", "AI") | `queries/templates.py`, `seeds.py` |
| 0.9b | Dashboard/report: key everything by `system_key`; fix the Category Deep Dive crash, the Validation page join, geometry run filtering, and `report list-runs`. Retry only transient errors; take a rate-limit token per attempt | `dashboard/`, `reporting/`, `cli/report.py`, `providers/resilient.py` |
| 0.10 | `ruff --fix` + manual fixes → CI green; add unit tests for RBO, nDCG, geometry metric bounds, and the judge prompt↔ground-truth contract | repo-wide |

**Done when** existing CLI flows produce per-mode, error-separated, paired statistics and every fix has a test.

### Phase 1 — The measurement core (≈1–2 weeks) ← biggest single lever
1. **Evidence layer** (`evidence/`): fetch each unique URL once; extract main text (trafilatura), language, and published date (htmldate); store a content-addressed snapshot; record fetch failures explicitly. Pick query-focused passages (BM25 windows) instead of blind first-1,000-character truncation, so long pages are judged on their relevant parts. Every system's documents go through the same path.
2. **Per-document constraint judging** (`judging/`): cheap deterministic checkers where the constraint is mechanical (numeric with unit/scale normalization, dates, language, QID membership), then an LLM with a structured schema, one verdict per constraint plus an evidence quote, plus UMBRELA-style 0–3 relevance. Cache judgments by (case, sha256, judge, prompt_version). Cost then scales with **unique documents**, not providers × modes.
3. **Metrics** (`metrics/`): **violation@k** (the headline metric for negation/numeric/temporal: what fraction of the top-k *violates* the constraint), all-constraints-satisfied@k, nDCG@10 over pooled graded labels, P@k, rank of first fully-satisfying doc. Stored per (case, system).
4. **Statistics**: paired bootstrap CIs on per-case differences, BH across categories, n and minimum detectable effect in every table.
5. **Judge calibration** (`calibrate` command): a stratified sample of about 200 (case, doc) pairs, human-labeled through a tiny labeling UI or CLI; Cohen's κ per constraint type per judge configuration. Choose the judge on evidence and print κ in every report.

**Done when** a report shows violation@10 and nDCG@10 per system × category with paired CIs, judge κ ≥ 0.6 on the calibration set, and re-evaluating from cached snapshots reproduces the same numbers exactly.

### Phase 2 — Verifiable test cases: the dataset (≈1–2 weeks, then ongoing)
1. **TestCase v2 schema** (above) with a v1→v2 converter.
2. **Generators backed by structured knowledge, so the answer set is known:**
   - *Wikidata SPARQL:* exclusion ("landlocked countries that have a navy", "programming languages not influenced by C"), numeric ranges (population, height, founding year, employees), temporal (events or launches in a date window; publication-date constraints checked against extracted `published_date`), 3-way conjunctions, same-label entity disambiguation (Paris, Texas vs. Paris, France via QIDs), directed relations ("acquired by" / "founded by" with argument swap).
   - *QUEST-style* Wikipedia category set operations (intersection and difference) for boolean and negation queries (Malaviya et al. 2023).
   - A result is **satisfying** if the page is about an entity in the answer set (Wikipedia URL → QID, or entity linking) and **violating** if it's in the negative set (e.g. AI companies for "not in AI").
3. **LLM generation v2** must emit typed constraints, a twin and a verification plan. It's then auto-validated (parses, non-contradictory, answerable: some system or a reference search surfaces at least one satisfying doc), otherwise discarded.
4. **Twins for every case** (negate / antonym / ×10 / ±1y / argument swap), which Phase 3 depends on.
5. **Human review CLI** to promote cases to *gold*; versioned JSONL under `data/cases/vN/`; a frozen **Core** split for comparisons over time and a rotating **Fresh** split for time-sensitive queries.

**Done when** there are ≥50 verifiable cases per core category (negation, numeric, temporal, multi-constraint, entity, compositional, boolean), each with at least one twin.

### Phase 3 — Metamorphic testing: no ground truth needed (≈3–5 days)
Replace `perturbation/` with `metamorphic/` using explicit relations:
- **INV** (meaning-preserving: validated paraphrase, typo, synonym): results *should not* change much.
- **DIR** (operator flip: the case's twin): results *should* change.
- **Noise floor:** run each query twice (and again later) to measure the provider's own variance.
- **Operator Blindness Index:** OBI = (RBOext(q, q′) − RBO_unrelated) / (RBO_noise − RBO_unrelated). OBI ≈ 1 means the engine ignores the operator entirely; ≈ 0 means the results differ as much as for unrelated queries.
- **Judged DIR:** do results for q′ satisfy q′'s constraints better than results for q do? That confirms the change went in the right direction, not just that something changed.

This yields the most direct evidence for the project's thesis cheaply: *"adding NOT changes Exa-neural's top-10 by X% while its noise floor is Y%"*, per category, with CIs, and without an LLM judge in the loop.

**Done when** OBI has CIs per system × operator and has been checked against judged violation@k.

### Phase 4 — Explanations that predict failures (≈1 week)
- **Triplet probes:** for each case, (q, d+, d−), where d− is a *violating near-duplicate* (same topic, operator violated), taken from the judged pool or created as a minimal counterfactual edit of d+. Measure **pairwise accuracy P[s(q,d+) > s(q,d−)]** and the margin distribution across: modern open embedders with their proper query/passage prompts (e.g. BGE-M3, multilingual-E5, GTE, Nomic, Jina v3, Qwen3-Embedding), API embedders, BM25, and rerankers (BGE-reranker-v2, an LLM reranker).
- **Per-case linkage:** does triplet failure on open models *predict* provider violation on the same case (AUC / logistic regression, n in the hundreds)? This answers "why" empirically and replaces the n=13 category correlation.
- Keep query–query geometry only as a secondary visualization; drop LID/isotropy or compute them on adequately sized samples. For multi-constraint/boolean, reproduce a small version of the dimensionality limits in *On the Theoretical Limitations of Embedding-Based Retrieval* (Weller et al. 2025, the LIMIT dataset).

**Done when** there's per-category triplet accuracy with CIs for ≥6 retrievers, plus the per-case prediction AUC.

### Phase 5 — Adversarial discovery that finds real failures (≈1 week)
- Search a **structured** space (generator × entity × operator × phrasing) with LLM mutation operators that preserve validity ("rephrase, keep constraints", "add a constraint from the property set", "swap in a rarer entity").
- **Differential fitness:** the target system fails (violation@k or OBI) *and* the query is valid (constraints parse, reads naturally) *and* it's answerable (non-empty answer set, or a reference system retrieves a satisfying doc).
- A **MAP-Elites** archive over (category × difficulty × entity popularity) for diversity; embedding deduplication; fitness cache; re-evaluate elites to counter noise; the shared budget ledger.
- Discovered failures are promoted into TestCases with twins, so they become a regression suite.

**Done when** ≥80% of discovered cases pass human validity review and they fail more often than seed cases on held-out systems.

### Phase 6 — The fix: Aris consolidated into a measurable wrapper (≈1–2 weeks)
`ConstraintAwareSearch(system)`:
1. Parse the query into the *same* typed constraints as TestCase v2 (shared code).
2. Plan retrieval: a positive-only rewrite (strip the negation) plus provider-native filters (date ranges, include/exclude text, category) and over-retrieval (k≈30–50) across rewrites.
3. Verify each candidate with the Phase 1 judging layer (checkers plus a small model).
4. Rerank: all constraints satisfied first, then relevance.
5. Return results with per-constraint evidence.

Evaluate `exa+aris` and `brave+aris` as systems on the same benchmark, and report quality gain per category against added latency and $. Park the rest of Aris (API server, agent loop, local index) until the wrapper shows significant gains.

**Done when** violation@10 drops significantly on negation/numeric/temporal for at least 2 base providers, with cost and latency reported.

### Phase 7 — Consolidation (alongside the above)
- One storage layer with versioned migrations and one config. Delete the duplicate DB and config modules. `modal_app.py` and the notebook call library functions only.
- Core entities: Case · System · Run(manifest) · ResultList · Snapshot · Judgment · Metric.
- Report: one **report card** per run with headline metrics and CIs, per-category tables, and, most persuasive of all, **concrete failure exhibits** (query → violating document → quoted evidence).

## 7. What to cut or freeze now
- The **vulnerability score**, isotropy/LID per category, and the dashboard pages built on them: they give confident-looking numbers with no validity.
- The **heuristic "embedding_sim" fitness**: it optimizes query shape, not engine behavior.
- **Position-derived "scores"** for Brave/SerpAPI: use `None` and rank only.
- **Cross-encoder "embedding gap"** as currently defined: replace it with rerankers evaluated against judged labels in Phase 4.
- Aris subsystems not on the Phase 6 path.

## 8. Suggested first PRs (in order)
1. Phase 0.10 (CI green) + 0.1 (search_mode) + 0.2 (status), which unblock trustworthy comparisons.
2. Phase 0.3–0.6, the judge, budget, stats and metric bugs, each with a failing test first.
3. Phase 1.1 (evidence layer) + TestCase v2 schema: the foundation for everything else.
4. Phase 3 (metamorphic OBI): the quickest route to a real headline finding, runnable on existing seeds once twins exist.

## 9. Decisions I need from you
1. **Instrument or product first?** The plan builds the measuring instrument before the fix, because the fix can't be shown to work without it. If Aris-as-product matters more, Phase 6 can start after Phase 1.
2. **Spend envelope** per full benchmark run (search API + judge tokens). This sets dataset size and judge choice.
3. **Provider set.** Keep the current four, or add others (e.g. Perplexity's search API, Parallel, Linkup, You.com)? Exa's current API modes should be re-verified when the adapter is fixed.
4. **Human labeling.** Are you willing to label ~200 pairs for judge calibration? It's the single highest-trust step.

---

## 10. References that inform the plan
- Weller, Lawrie, Van Durme. *NevIR: Negation in Neural Information Retrieval* (EACL 2024). Pairwise negation test: most bi-encoders and sparse models perform at or below random, and cross-encoders do best. https://aclanthology.org/2024.eacl-long.139/
- Weller et al. *On the Theoretical Limitations of Embedding-Based Retrieval* (ICLR 2026), LIMIT dataset. https://arxiv.org/abs/2508.21038
- Malaviya et al. *QUEST: A Retrieval Dataset of Entity-Seeking Queries with Implicit Set Operations* (ACL 2023). Wikipedia-category set operations as verifiable negation/conjunction ground truth.
- Upadhyay et al. *UMBRELA: the open-source reproduction of the Bing relevance assessor* (2024); Thomas et al. *Large language models can accurately predict searcher preferences* (2023). https://arxiv.org/abs/2406.06519
- Ribeiro et al. *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList* (ACL 2020). INV/DIR metamorphic tests.
- Webber, Moffat, Zobel. *A Similarity Measure for Indefinite Rankings* (TOIS 2010). Extrapolated RBO.

## Appendix: how this analysis was produced
- I read all core modules directly: queries, evaluation, providers, pipeline, storage, geometry, validation, perturbation, adversarial.
- Two independent read-throughs covered `src/aris/` (all 76 files, with import and probe scripts) and the intelligence/session/reporting/dashboard layer.
- Headline claims were re-checked against source and, where marked, reproduced by running code in a local venv.
- Test suite: 200 passed. `ruff check src/searchprobe`: 182 errors.
