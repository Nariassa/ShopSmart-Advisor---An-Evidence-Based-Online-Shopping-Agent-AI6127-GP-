# Retrieval Baseline Suite — Completion Summary

## Completion status

All retrieval baselines specified by the project proposal are complete:

1. BM25 lexical retrieval
2. Dense retrieval with `BAAI/bge-base-en-v1.5`
3. Hybrid BM25 + Dense retrieval with Reciprocal Rank Fusion
4. Hybrid retrieval followed by cross-encoder reranking

Every run used the same 4,450,749-chunk corpus and the same nine-query pilot
evaluation set. Each run stores 20 unique ranked results per query.

## Aggregate comparison

| Baseline | Aspect P@5 | Aspect P@10 | Aspect MRR@10 | Aspect nDCG@10 | Known-item Success@10 | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | **0.900** | **0.850** | 0.875 | 0.851 | 0.667 | 93.65 s |
| Dense | 0.700 | 0.783 | 0.778 | 0.756 | 0.667 | **0.574 s** |
| Hybrid RRF | 0.800 | 0.817 | **0.917** | 0.815 | 0.667 | 94.22 s |
| Cross-Encoder | **0.900** | 0.833 | **0.917** | **0.851** | 0.667 | 94.66 s |

## Main findings

- BM25 produced the strongest silver-label precision, but its Python index is
  too slow for interactive use on the 15.68 GB test machine.
- Dense retrieval was approximately 163 times faster than BM25 after index
  initialization, but scored lower against keyword-derived aspect labels.
- Hybrid RRF produced the best first-relevant ranking and showed strong source
  diversity: the saved BM25 and dense top-20 lists overlapped by only 2.89
  results per query on average.
- Cross-encoder reranking restored BM25-level P@5 and nDCG while retaining the
  hybrid system's best MRR. Its mean reranking overhead was 0.442 seconds for
  20 candidates.
- The generic protection-plan product failed in every baseline because the
  target was absent from the retrieved candidate pools.

## Artifact map

| Baseline | Report | Metrics | Results | Runner |
|---|---|---|---|---|
| BM25 | `BM25_BASELINE_README.md` | `results/bm25_pilot_metrics.json` | `results/bm25_pilot_results.jsonl` | `scripts/run_bm25_baseline.py` |
| Dense | `DENSE_BASELINE_README.md` | `results/dense_pilot_metrics.json` | `results/dense_pilot_results.jsonl` | `scripts/run_dense_baseline.py` |
| Hybrid RRF | `HYBRID_RRF_BASELINE_README.md` | `results/hybrid_rrf_pilot_metrics.json` | `results/hybrid_rrf_pilot_results.jsonl` | `scripts/run_hybrid_rrf_baseline.py` |
| Cross-Encoder | `CROSS_ENCODER_BASELINE_README.md` | `results/cross_encoder_pilot_metrics.json` | `results/cross_encoder_pilot_results.jsonl` | `scripts/run_cross_encoder_baseline.py` |

The compact comparison is `results/baseline_comparison.csv`. The
machine-readable completion record is `results/baseline_suite_manifest.json`.

## Verification

Run the suite-level integrity check from the project directory:

```powershell
.\.venv\Scripts\python.exe scripts\verify_baseline_suite.py
```

The verifier checks corpus size, query identity and order, result depth,
duplicate result IDs, metric status, required files, and comparison-table order.

## Interpretation limits

- The pilot contains only nine queries.
- Relevance is derived from product ASINs and keyword-generated aspect labels,
  not independent human judgments.
- The aspect labels favor lexical matching and may under-credit semantic
  results.
- Hybrid and reranker end-to-end latency is reconstructed from saved source
  runs rather than measured in one memory-constrained process.
- Candidate depth is limited to the 20 results saved by the source runs.

The baseline suite is complete for pipeline validation. The four frozen systems
have also been evaluated on all 20 formal queries using 791 human-confirmed
graded judgments. Dense retrieval is the strongest formal system, with P@10 of
0.910, MRR@10 of 0.900, and nDCG@10 of 0.743. See
`FORMAL_EVALUATION_README.md` for the formal results and interpretation.
