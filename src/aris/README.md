# Aris

Aris is SearchProbe's answer to the failures it measures: a **constraint-aware
verify-and-rerank layer** that wraps any search provider.

## Primary path: `aris.wrapper.ConstraintAwareSearch`

1. Parse the query into typed constraints (the same schema as SearchProbe test
   cases) and retrieval rewrites, the first being the query's positive core with
   negations removed.
2. Over-retrieve from the base provider with the original query and rewrites;
   fuse with reciprocal rank fusion.
3. Fetch each candidate page once (shared evidence layer).
4. Verify each candidate per constraint: deterministic checkers, then an LLM verifier.
5. Drop violating documents; rank fully satisfying ones first, then by relevance.

It is a SearchProbe provider, so it is measured on the same cases with the same
judge as everything else:

```bash
searchprobe run --providers exa,aris+exa --exa-modes neural
searchprobe judge --budget 5 --out reports/aris_vs_exa.md
aris wrap "European countries that are not members of the European Union" --base exa
```

The verifier model (`SEARCHPROBE_ARIS_VERIFIER_MODEL`) should differ from the
evaluation judge (`SEARCHPROBE_JUDGE_MODEL`) so that a wrapped system is not graded
by the model that filtered it.

## Parked modules

`agent/`, `api/`, `index/`, `retrieval/` (HyDE, local dense/sparse/structured
stores), `que/` regex parsers, `verification/` regex verifiers and `benchmark/`
predate the wrapper. Known crash bugs in them have been fixed, but they are not on the
evaluated path and remain parked until the wrapper shows measured gains. The legacy
`aris benchmark` command only checks that results come back.
