# Triplet Probe Report

Pairwise accuracy: how often a scorer ranks the document that satisfies the query above one that violates it (0.5 = chance). Brackets: 95% bootstrap CIs.

| Category | bm25 |
|---|---|
| boolean_logic | 19% [6%, 34%] (n=32) |
| compositional | 100% [100%, 100%] (n=1) |
| entity_disambiguation | 100% [100%, 100%] (n=2) |
| multi_constraint | 61% [39%, 78%] (n=23) |
| negation | 33% [23%, 43%] (n=94) |
| numeric_precision | 57% [42%, 72%] (n=40) |
| temporal_constraint | 45% [18%, 73%] (n=11) |
| **all** | 40% [33%, 48%] (n=203) |
