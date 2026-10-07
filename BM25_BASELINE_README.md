# BM25 Pilot Baseline

## Status

The BM25 pilot baseline completed successfully on the full Amazon Electronics
knowledge base. The run used all 4,450,749 chunks and did not reduce the corpus
or reuse cached query results.

This is a **silver-label pilot**, intended to validate the retrieval pipeline
and establish an initial comparison point. It is not a substitute for the
manually judged query set required for final project claims.

## Evaluation design

The pilot contains nine queries:

- Three short-title known-item queries. Every chunk with the target
  `parent_asin` is relevant.
- Six natural-language aspect queries covering battery life, sound quality,
  screen/display, connectivity, build quality, and price/value. Relevance is
  derived from the corpus's existing keyword-based `aspect` label.

Each query retrieves the top 100 chunks. The top 20 are saved for inspection.
Metrics are calculated at ranks 1, 5, and 10.

## Aggregate results

| Suite | Queries | P@1 | P@5 | P@10 | Success@10 | MRR@10 | nDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aspect silver labels | 6 | 0.833 | 0.900 | 0.850 | 1.000 | 0.875 | 0.851 | 0.000034 |
| Known-item short title | 3 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.404 |

Aspect recall is extremely small because each aspect has between 136,744 and
460,441 relevant chunks in the full corpus. Precision and ranking metrics are
more informative for that suite.

## Latency

| Measurement | Seconds |
|---|---:|
| Index cold load | 143.35 |
| Mean query latency | 93.65 |
| Median query latency | 94.91 |
| P95 query latency | 110.68 |
| Total scoring time for nine queries | 842.82 |

Hardware used for this run:

- CPU: 12th Gen Intel Core i7-12700H, 14 cores / 20 logical processors
- RAM: 15.68 GB
- Python: 3.10.11

The supplied `rank_bm25` representation is much larger than available free
RAM. Query execution therefore incurs substantial Windows paging. The local
retriever uses the exact stored BM25 statistics and formula but visits every
document dictionary once per query, rather than once per query token. A unit
comparison confirmed that its scores match `rank_bm25.get_scores()`.

The measured latency is acceptable for an offline baseline, but not for an
interactive system. A production-oriented experiment should use a persistent
inverted-index engine rather than scanning Python dictionaries.

## Error analysis

### Known-item failure

The query `Allstate 1-Year Tablet Accidental Protection` did not retrieve its
target ASIN in the top 10. The terms are generic and overlap with many similar
protection-plan products. The other two known-item queries placed only chunks
from the correct product in the top 10.

### Aspect-label mismatch

For `monitor with a bright sharp screen`, the top two chunks were labeled
`design/appearance`, while the expected label was `screen/display`. A relevant
`screen/display` result first appeared at rank 4. This may reflect both BM25's
lexical behavior and the limitations of the keyword-derived aspect taxonomy.

## Reproduction

Run from the `electronics_kb` directory:

```powershell
.\.venv\Scripts\python.exe scripts\run_bm25_baseline.py
```

The default run creates:

- `evaluation/bm25_pilot_queries.jsonl`
- `results/bm25_pilot_results.jsonl`
- `results/bm25_pilot_metrics.json`
- `results/bm25_pilot_metrics.csv`

The number of pilot queries and retrieval depth can be changed with command-line
arguments. Run `python scripts/run_bm25_baseline.py --help` for all options.

## Limitations and next evaluation step

- Short-title queries have direct lexical overlap with product-name prefixes in
  the indexed chunks.
- Aspect labels were created by keyword rules, which favors lexical retrieval
  and can contain classification errors.
- Nine queries are enough for a pipeline pilot, not for statistical conclusions.
- The final evaluation should use manually authored natural-language queries,
  independently judged relevance labels, and the same saved metric definitions.
