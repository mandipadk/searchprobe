# SearchProbe

**Adversarial benchmark and research framework for neural search engines.**

SearchProbe goes beyond testing *that* search fails — it measures *why* by analyzing embedding space geometry, validating with cross-encoders, and using evolutionary optimization to discover failure modes automatically.

## Why SearchProbe?

Neural/embedding-based search engines (like [Exa.ai](https://exa.ai)) represent a paradigm shift from traditional keyword search. But embeddings have specific, predictable failure modes:

- **Negation blindness**: "companies NOT in AI" returns AI companies
- **Numeric imprecision**: "exactly 50 employees" matches any company size
- **Temporal drift**: "January 2024" and "January 2025" look similar in vector space
- **Antonym confusion**: "increase" and "decrease" are embedding neighbors

SearchProbe generates adversarial queries that exploit these weaknesses, runs them against multiple search providers, and uses LLM judges to evaluate result quality. The research modules then explain *why* failures occur at the geometric level.

## Features

### Benchmarking
- **13 adversarial query categories** with failure hypotheses grounded in embedding theory
- **Multi-provider benchmarking**: Exa, Tavily, Brave Search, Google (via SerpAPI)
- **3-tier query generation**: hand-curated seeds, parameterized templates, LLM-generated
- **LLM-as-judge evaluation** with category-specific dimension weighting
- **Statistical rigor**: confidence intervals, bootstrap CIs, Benjamini-Hochberg correction

### Research Modules
- **Embedding Geometry Analyzer** — Measures cosine similarity collapse, intrinsic dimensionality, and isotropy to explain *why* embeddings fail on each adversarial category
- **Cross-Encoder Validation** — Quantifies the "embedding gap" by re-scoring results with a cross-encoder, computing NDCG improvement potential per category
- **Perturbation Analysis Engine** — Systematically perturbs queries and measures result stability, producing word-level sensitivity maps
- **Adversarial Query Optimizer** — Evolutionary optimization that breeds worst-case queries, discovering failure modes humans wouldn't think of

### Visualization & Reporting
- **Interactive dashboard** (Streamlit) with 8 pages including geometry, robustness, and validation views
- **Publication-quality charts**: vulnerability heatmaps, t-SNE/UMAP projections, 3D embedding explorers, sensitivity maps
- **HTML reports** with radar charts and failure taxonomies

## Installation

```bash
# Clone the repository
git clone https://github.com/mandipadk/searchprobe.git
cd searchprobe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Install analysis dependencies (for geometry, validation, perturbation modules)
pip install -e ".[analysis]"

# Copy environment template and add your API keys
cp .env.example .env
```

The `analysis` extra installs sentence-transformers, PyTorch, scikit-learn, UMAP, and NLTK. Models auto-download on first use (~100-400MB each).

## Quick Start

### Benchmarking Workflow

```bash
# Generate adversarial queries
searchprobe generate --count 20

# Run benchmark against providers
searchprobe run --providers exa,tavily --exa-modes auto,neural

# Evaluate results with LLM judge
searchprobe evaluate --run-id latest

# Generate reports
searchprobe report --format both

# Launch interactive dashboard
searchprobe dashboard
```

### Research Workflow

```bash
# Analyze embedding geometry — understand WHY search fails
searchprobe geometry --models all-MiniLM-L6-v2,all-mpnet-base-v2

# Validate with cross-encoder — measure the embedding gap
searchprobe validate --run-id latest

# Perturbation analysis — find load-bearing words
searchprobe perturb --run-id latest --operators word_delete,word_swap

# Evolve adversarial queries — breed worst-case failures
searchprobe evolve --provider exa --generations 20 --population 30
```

## Architecture

```
src/searchprobe/
├── adversarial/      # Evolutionary query optimizer
│   ├── crossover.py  #   Crossover operators (clause swap, constraint merge)
│   ├── fitness.py    #   Fitness evaluation (LLM judge, cross-encoder, embedding)
│   ├── mutations.py  #   Mutation operators (negation toggle, entity swap, etc.)
│   └── optimizer.py  #   Evolutionary loop with tournament selection
├── cli/              # CLI commands (generate, run, evaluate, geometry, etc.)
├── config/           # Environment-based configuration
├── dashboard/        # Streamlit dashboard (8 pages)
├── evaluation/       # LLM-as-judge + statistical analysis
├── geometry/         # Embedding space geometry analysis
│   ├── analyzer.py   #   Core analyzer using sentence-transformers
│   ├── metrics.py    #   Cosine sim, intrinsic dimensionality, isotropy
│   ├── pairs.py      #   70+ curated adversarial embedding pairs
│   └── vulnerability.py  # Calibrated vulnerability scoring
├── perturbation/     # Systematic robustness testing
│   ├── engine.py     #   Perturbation + search + stability measurement
│   ├── operators.py  #   Word delete/swap, negation, synonym replace
│   └── stability.py  #   Jaccard, Rank-Biased Overlap, sensitivity maps
├── pipeline/         # Benchmark orchestration, rate limiting, cost tracking
├── providers/        # Search provider adapters (Exa, Tavily, Brave, SerpAPI)
├── queries/          # Query generation (seeds, templates, LLM)
├── reporting/        # Chart generation (Plotly) and HTML reports
├── storage/          # SQLite database with WAL mode
├── utils/            # Shared parsing and logging utilities
└── validation/       # Cross-encoder validation and gap analysis
```

## Configuration

Set your API keys in `.env`:

```env
SEARCHPROBE_EXA_API_KEY=your-exa-key
SEARCHPROBE_TAVILY_API_KEY=your-tavily-key
SEARCHPROBE_BRAVE_API_KEY=your-brave-key
SEARCHPROBE_SERPAPI_API_KEY=your-serpapi-key
SEARCHPROBE_ANTHROPIC_API_KEY=your-anthropic-key
```

## Adversarial Categories

| Category | Failure Hypothesis |
|----------|-------------------|
| Negation | Embeddings collapse "NOT X" to "X" |
| Numeric Precision | Numbers are tokens, not values |
| Temporal Constraint | Dates are weakly encoded |
| Multi-Constraint | Similarity averages constraints |
| Polysemy | Word senses blend together |
| Compositional | Word order is partially lost |
| Antonym Confusion | Antonyms are distributional neighbors |
| Specificity Gradient | Resolution limits in embedding space |
| Cross-Lingual | Multilingual alignment gaps |
| Counterfactual | Returns factual instead of hypothetical |
| Boolean Logic | No native AND/OR/NOT operators |
| Entity Disambiguation | Named entities conflate |
| Instruction Following | Meta-instructions ignored |

## Key Research Concepts

### Embedding Geometry Analysis
If `cos(embed("companies in AI"), embed("companies NOT in AI")) = 0.96`, then **any** embedding-based retrieval system will fail on negation — it's a fundamental geometric limitation, not a provider bug. SearchProbe measures:

- **Adversarial Collapse Ratio** — `adversarial_sim / baseline_sim` (>1 means adversarial pairs are more similar than they should be)
- **Local Intrinsic Dimensionality** — MLE estimate per Amsaleg et al. 2015 (lower = queries clustered in subspace)
- **Isotropy Score** — How uniformly distributed embeddings are (per Mu & Viswanath 2018)
- **Vulnerability Score** — Composite metric per category, 0.0 (robust) to 1.0 (catastrophic collapse)

### Cross-Encoder Validation
Cross-encoders jointly encode (query, document) pairs at O(n) cost, producing much more accurate scores than bi-encoders. By re-scoring results, we quantify the exact "embedding gap":
- **NDCG improvement potential** per category
- **Kendall's tau** rank correlation between provider ranking and optimal ranking

### Perturbation Analysis
Systematic query modification reveals which words are "load-bearing" for retrieval, producing sensitivity maps analogous to attention visualization in NLP.

## Development

```bash
# Run tests
pytest

# Run only unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Type checking
mypy src/searchprobe

# Linting
ruff check src/searchprobe

# Format code
ruff format src/searchprobe
```

## License

MIT

## Acknowledgments

This project was built to demonstrate understanding of neural search systems, particularly [Exa.ai](https://exa.ai)'s embedding-based search architecture. The evaluation methodology follows Exa's published approach using LLM-as-judge with pointwise scoring.
