# SearchProbe test cases (TestCase v2)

Each line of a `.jsonl` file is one `searchprobe.cases.models.TestCase`: a query, typed
constraints that define a correct result, and (where available) known answer and negative
entity sets and minimal-pair twins.

| File | Source | How correctness is known |
|---|---|---|
| `v1/wikidata.jsonl` | `searchprobe cases generate wikidata` | Answer and negative sets from Wikidata SPARQL (recorded in `provenance.source`), with attribute values for numeric constraints. Twins have their own answer sets. |
| `v1/structured.jsonl` | `searchprobe cases generate structured` | Language constraints (decided by the language detector) and publication-date windows (decided from the page's extracted publication date); doc-type constraints are LLM-judged. |
| `v1/seeds.jsonl` | `searchprobe cases convert-seeds` | The original hand-written v1 seeds converted to typed constraints, plus rule-based twins. |

## Using it

```bash
searchprobe cases list                      # summary by category
searchprobe cases validate                  # schema + integrity (twins resolve, ids unique)
searchprobe cases import -c negation        # load into the DB as a query set
searchprobe run --providers exa,brave --exa-modes neural,auto
searchprobe judge --budget 5 --out reports/scorecard.md
searchprobe metamorphic --systems exa:neural,exa:auto,brave -c negation --out reports/obi.md
```

## Caveats

- Wikidata is incomplete and changes over time. `answer_set_complete` is only `true` for
  small, curated sets (e.g. EU membership, Nobel laureates of a year). For other cases a
  document about an entity outside the answer set is judged on its content, not penalized
  for being absent from the set.
- Wikidata membership data has errors. Generation stops unless every organization's current
  member count matches `EXPECTED_MEMBER_COUNTS` in `generators/wikidata.py`; known errors
  are corrected in `MEMBERSHIP_ADD` / `MEMBERSHIP_REMOVE` (as of 2026-10: Slovakia missing
  from the Council of Europe, the UK from the Commonwealth, Australia listed in ASEAN,
  Angola still in OPEC). Ended memberships (the UK in the EU) are excluded.
- Numeric values are Wikidata's best-ranked statements at generation time; populations in
  particular move. Hard negatives are entities just outside the range.
- Cases marked `validation: "auto"` passed schema checks but have not been reviewed by a
  person. Promote cases with `searchprobe cases review <file>`; gold cases are never
  overwritten by regeneration.
