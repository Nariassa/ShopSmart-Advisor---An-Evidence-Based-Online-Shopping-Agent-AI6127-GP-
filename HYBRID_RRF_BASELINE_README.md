# Hybrid BM25 + Dense RRF Pilot Baseline

## Status

The third ordered baseline completed successfully by combining the saved BM25
and dense rankings with Reciprocal Rank Fusion (RRF).

Configuration:

- Corpus: 4,450,749 Amazon Electronics chunks
- Queries: the same nine-query silver-label pilot used by both source runs
- Sources: BM25 and `BAAI/bge-base-en-v1.5` dense retrieval
- Candidate depth: top 20 saved results from each source
- RRF constant: `k = 60`
- Metric depth: ranks 1, 5, and 10

The RRF score for a document is:

```text
RRF(document) = sum(1 / (60 + source_rank))
```

## Aggregate results

| Suite | Queries | P@1 | P@5 | P@10 | Success@10 | MRR@10 | nDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aspect silver labels | 6 | 0.833 | 0.800 | 0.817 | 1.000 | 0.917 | 0.815 | 0.000033 |
| Known-item short title | 3 | 0.667 | 0.667 | 0.567 | 0.667 | 0.667 | 0.596 | 0.357 |

## Three-baseline comparison

| Metric | BM25 | Dense | Hybrid RRF | Best |
|---|---:|---:|---:|---|
| Aspect P@5 | **0.900** | 0.700 | 0.800 | BM25 |
| Aspect P@10 | **0.850** | 0.783 | 0.817 | BM25 |
| Aspect MRR@10 | 0.875 | 0.778 | **0.917** | Hybrid RRF |
| Aspect nDCG@10 | **0.851** | 0.756 | 0.815 | BM25 |
| Known-item Success@10 | 0.667 | 0.667 | 0.667 | Tie |
| Known-item MRR@10 | **0.667** | 0.500 | **0.667** | BM25 / Hybrid |
| Known-item nDCG@10 | **0.667** | 0.398 | 0.596 | BM25 |

Hybrid RRF produced the best aspect MRR@10. It moved the first
`screen/display` result for the difficult display query to rank 2, compared
with rank 4 for BM25 and rank 6 for dense retrieval. Its overall precision and
nDCG remained between the two source systems.

## Candidate diversity

Only 2.89 results on average in each fused top 20 were present in both source
lists. Three aspect queries had no shared result in their fused top 20. This
shows that BM25 and dense retrieval contribute substantially different
candidates, even when the silver-label aggregate gain is modest.

## Latency

| Measurement | Seconds |
|---|---:|
| Mean RRF computation | 0.000115 |
| Mean reconstructed sequential latency | 94.22 |
| Mean estimated parallel latency | 93.65 |
| Combined source cold initialization | 397.44 |

RRF computation is negligible. End-to-end latency remains dominated by the
current Python BM25 implementation. The sequential and parallel values are
reconstructed from the saved source runs; the sources were not loaded together
in one process. Parallel execution may exceed the practical memory budget of
the 15.68 GB test machine.

## Error analysis

- The generic Allstate protection-plan query still failed because neither
  source retrieved the target product in its saved candidate list.
- Hybrid RRF improved the display query's first relevant rank by combining the
  two rankings, but taxonomy noise remained prominent.
- Product queries benefited when both systems retrieved the same ASIN. The
  first two product queries ranked a correct chunk first; the third failed in
  all three baselines.
- For aspect queries with little or no source overlap, RRF interleaved strong
  candidates from both systems instead of producing large score gains from
  agreement.

## Reproduction

Run from the `electronics_kb` directory after the BM25 and dense result files
exist:

```powershell
.\.venv\Scripts\python.exe scripts\run_hybrid_rrf_baseline.py
```

The run creates:

- `results/hybrid_rrf_pilot_results.jsonl`
- `results/hybrid_rrf_pilot_metrics.json`
- `results/hybrid_rrf_pilot_metrics.csv`

## Limitations and next step

- This is an offline replay of saved source rankings, not a simultaneous live
  retrieval run.
- Only the top 20 saved candidates per source were available. A later formal
  run should evaluate candidate depths such as 50 and 100.
- The nine-query silver-label pilot is too small for final statistical claims.
- Keyword-derived aspect labels can favor BM25 and under-credit semantically
  useful dense results.
- The next ordered baseline is cross-encoder reranking over the hybrid
  candidates, preferably after the manual evaluation set is ready.
