# Triplet Probe Report

Pairwise accuracy: how often a scorer ranks the document that satisfies the query above one that violates it (0.5 = chance). Brackets: 95% bootstrap CIs.

| Category | bm25 |
|---|---|
| boolean_logic | 0% [0%, 0%] (n=8) |
| compositional | 100% [100%, 100%] (n=1) |
| entity_disambiguation | 100% [100%, 100%] (n=6) |
| multi_constraint | 55% [35%, 75%] (n=20) |
| negation | 39% [26%, 52%] (n=54) |
| numeric_precision | 42% [26%, 58%] (n=38) |
| temporal_constraint | 40% [0%, 80%] (n=5) |
| **all** | 43% [35%, 52%] (n=132) |
