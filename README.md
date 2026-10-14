# SearchProbe

**A measurement instrument for where, and why, neural search fails on queries whose meaning
depends on logic: negation, numbers, dates, conjunctions, set differences, entity identity
and instructions.**

"Neural search fails on negation" is a claim; SearchProbe turns it into evidence:

| Question | How SearchProbe answers it |
|---|---|
| **Did it fail?** | Every returned document is fetched and judged **per constraint** (*is this an AI company?*), with deterministic checkers for what is mechanical (language, publication date, membership of a Wikidata answer set) and an LLM judge for the rest. |
| **How badly, compared with what?** | Rank-aware metrics (**violation@k**, all-constraints-satisfied@k, pooled nDCG@10, MRR) with bootstrap CIs, paired by case, Benjamini-Hochberg corrected. Each engine's own run-to-run noise is measured. |
| **Without ground truth?** | **Metamorphic tests**: each case has a minimal-pair *twin* (negation removed, number ×10, year shifted). The **Operator Blindness Index** measures how much an engine's results change when the meaning changes, against its own noise floor. |
| **Why?** | **Triplet probes**: does an open retriever score a satisfying document above a violating one? Linked *per case* to provider failures (ROC AUC). |
| **Can it be fixed?** | **Aris** wraps any provider with constraint parsing, over-retrieval, per-document verification and reranking, and is scored as another system on the same cases. |

## Install

```bash
git clone https://github.com/mandipadk/searchprobe.git && cd searchprobe
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core + tests
pip install -e ".[analysis]"       # optional: embedding/reranker probes (torch, sentence-transformers)
cp .env.example .env               # add provider keys and an Anthropic key (or Vertex AI)
```

## Quick start

```bash
# One command: cases -> search -> judge -> scorecard -> metamorphic tests
searchprobe research --profile profiles/research/negation.toml
# => reports/negation/{scorecard.md, scorecard.json, metamorphic.md, manifest.json}
```

Or step by step:

```bash
searchprobe cases list                                   # the dataset (321 cases)
searchprobe cases import -c negation,boolean_logic       # load into the DB as a query set
searchprobe run --providers exa,brave --exa-modes neural,auto
searchprobe judge --budget 5 --out reports/scorecard.md  # fetch pages, judge each document
searchprobe scorecard --format markdown                  # recompute from cache (no LLM calls)
searchprobe metamorphic -s exa:neural,exa:auto,brave -c negation --out reports/obi.md
searchprobe dashboard                                    # 🎯 Scorecard page + legacy views
```

Is Aris better than its base engine?

```bash
searchprobe research --profile profiles/research/aris_ablation.toml   # exa:neural vs aris+exa:neural
aris wrap "European countries that are not members of the European Union" --base exa
```

## The dataset

`data/cases/v1/` holds 321 typed, versioned test cases: 250 base cases plus 71 minimal-pair
twins ([details](data/cases/README.md)):

- **Wikidata** (186): answer *and* negative sets from SPARQL. Examples: European non-members of
  the EU (France is the violating near-miss), countries with a population between 1 and 2
  million (hard negatives just outside the range), Nobel laureates of a given year, US
  namesake cities vs. the famous original (Paris, Texas vs. Paris), river tributaries. Twins
  get their own answer sets from the same data. Hard-coded Wikidata ids are checked against
  their labels, and organization memberships against known member counts, before generation.
- **Structured** (105): language constraints checked by a language detector, publication
  windows checked from extracted page dates, document-type instructions.
- **Seeds** (30): the original hand-written queries converted to typed constraints.

Grow it with `searchprobe cases generate wikidata|structured|llm` (LLM cases need typed
constraints and can be filtered for answerability), harden it with `searchprobe discover`
(MAP-Elites with differential fitness: the target must fail **and** a reference engine must
succeed), and promote reviewed cases with `searchprobe cases review`.

## First findings

From `docs/findings/` (203 Wikidata triplets, open models on CPU):

- None of the tested retrievers (MiniLM, BGE-small, E5-base, Qwen3-Embedding-0.6B, MS MARCO
  MiniLM and BGE rerankers) prefers the satisfying document more often than chance: **48-53%**
  overall. On "members of X but not of Y" every model is **below** chance (16-34%), because
  the violating documents mention both organizations. BM25 is at **40%** overall.

Provider-level scorecards need provider keys; run `searchprobe research` to produce yours.

## Commands

| Command | Purpose |
|---|---|
| `cases list/validate/import/twins/review` | Inspect, check, load and curate test cases |
| `cases generate wikidata/structured/llm` | Build verifiable cases |
| `run` | Search every case with every system (budget enforced by reservation) |
| `judge` / `scorecard` | Per-document judging; metrics with CIs; failure exhibits |
| `calibrate label/report` | Label a stratified sample by hand; Cohen's κ of judge vs. human |
| `metamorphic` | Operator Blindness Index (DIR twins, INV typos/paraphrases, noise floor) |
| `explain triplets/probe` | Triplet probes on open retrievers; per-case failure prediction |
| `discover` | Breed harder, still-answerable variants of cases |
| `research` | Run a full profile end to end |
| `evaluate`, `report`, `geometry`, `validate`, `perturb`, `evolve`, `session` | Legacy list-level tools (kept, corrected, superseded by the above) |

## Architecture

```
src/searchprobe/
├── cases/         TestCase v2 schema, twins, converters, generators (wikidata, structured, llm)
├── evidence/      fetch once per URL, main-text/date/language extraction, snapshots, passages
├── judging/       deterministic checkers, per-document LLM judge, judgment cache, run engine
├── metrics/       violation@k, nDCG@k (pooled), scorecards, Markdown rendering
├── calibration/   human labels, stratified sampling, kappa
├── metamorphic/   variants, OBI, engine, reports
├── explain/       triplets, scorers (BM25, bi-/cross-encoders, OpenAI), probes, AUC
├── discovery/     case mutations, differential fitness, MAP-Elites
├── pipeline/      benchmark runner, cost tracker (reservations), rate limiter, research runs
├── providers/     Exa, Tavily, Brave, SerpAPI, resilient wrapper, registry (incl. aris+<base>)
├── llm.py         structured-output Claude calls, refusal handling, cost accounting
├── manifest.py    run manifests (git SHA, versions, judge)
└── (legacy)       evaluation/ (list-level judge), geometry/, validation/, perturbation/,
                   adversarial/, intelligence/, reporting/, dashboard/
src/aris/          wrapper.py (constraint-aware verify-and-rerank); other modules parked
```

## Methodology notes and limitations

- **Judge reliability is measured, not assumed.** Run `searchprobe calibrate label` on ~200
  pairs; `calibrate report` gives κ per constraint type. Treat judge-based numbers as
  provisional below κ = 0.6.
- **Uniform evidence.** The judge sees the same extracted page text for every system, never
  provider-specific snippets (unless a page cannot be fetched, which is flagged).
- **Infrastructure failures are not quality.** Search and judge errors are reported as error
  rates and excluded from quality means; empty result lists count as failures.
- **Wikidata is incomplete.** Documents outside a known answer set are judged on their content.
- **Independence.** Aris's verifier model is configured separately from the evaluation judge.
- **Cost.** Every stage has a budget enforced before dispatch; judgments are cached by
  (case, content hash, judge, prompt version), so re-runs and offline scorecards are free.

## Development

```bash
pytest                      # ~300 tests, no network or API keys needed
ruff check src && ruff format --check src
```

The analysis behind this redesign, and its plan, is in
[docs/ANALYSIS_AND_PLAN.md](docs/ANALYSIS_AND_PLAN.md).

## License

MIT
