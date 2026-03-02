# SearchProbe

**Adversarial benchmark framework for neural search engines.**

SearchProbe systematically discovers failure modes in embedding-based search engines through adversarial query generation, multi-provider benchmarking, and LLM-based evaluation.

## Why SearchProbe?

Neural/embedding-based search engines (like [Exa.ai](https://exa.ai)) represent a paradigm shift from traditional keyword search. But embeddings have specific, predictable failure modes:

- **Negation blindness**: "companies NOT in AI" returns AI companies
- **Numeric imprecision**: "exactly 50 employees" matches any company size
- **Temporal drift**: "January 2024" and "January 2025" look similar in vector space
- **Antonym confusion**: "increase" and "decrease" are embedding neighbors

SearchProbe generates adversarial queries that exploit these weaknesses, runs them against multiple search providers, and uses LLM judges to evaluate result quality.

## Features

- **13 adversarial query categories** with failure hypotheses grounded in embedding theory
- **Multi-provider benchmarking**: Exa, Tavily, Brave Search, Google (via SerpAPI)
- **LLM-as-judge evaluation** following Exa's own published methodology
- **Statistical rigor**: confidence intervals, significance tests
- **Interactive dashboard** for exploring results
- **Polished reports** with radar charts and failure taxonomies

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

# Copy environment template and add your API keys
cp .env.example .env
```

## Quick Start

```bash
# Test with a single query
searchprobe run --query "companies that are NOT in AI" --providers exa

# Generate adversarial queries
searchprobe generate --count 20

# Run full benchmark
searchprobe run --providers exa,tavily --exa-modes auto,neural,deep

# Evaluate results
searchprobe evaluate --run-id latest

# Generate reports
searchprobe report --format both

# Launch interactive dashboard
searchprobe dashboard
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

## Development

```bash
# Run tests
pytest

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
