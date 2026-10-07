# Cross-Encoder Reranking Pilot Baseline

## Status

The fourth ordered baseline completed successfully by reranking the top 20
Hybrid RRF candidates with `cross-encoder/ms-marco-MiniLM-L6-v2`.

Configuration:

- Corpus: 4,450,749 Amazon Electronics chunks
- Queries: the same nine-query silver-label pilot used by all prior baselines
- First-stage candidates: top 20 from Hybrid BM25 + Dense RRF
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Maximum pair length: 512 tokens
- Batch size: 16 on CPU
- Metric depth: ranks 1, 5, and 10

## Aggregate results

| Suite | Queries | P@1 | P@5 | P@10 | Success@10 | MRR@10 | nDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aspect silver labels | 6 | 0.833 | 0.900 | 0.833 | 1.000 | 0.917 | 0.851 | 0.000033 |
| Known-item short title | 3 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.404 |

## Four-baseline comparison

| Metric | BM25 | Dense | Hybrid RRF | Cross-Encoder | Best |
|---|---:|---:|---:|---:|---|
| Aspect P@5 | **0.900** | 0.700 | 0.800 | **0.900** | BM25 / Cross-Encoder |
| Aspect P@10 | **0.850** | 0.783 | 0.817 | 0.833 | BM25 |
| Aspect MRR@10 | 0.875 | 0.778 | **0.917** | **0.917** | Hybrid / Cross-Encoder |
| Aspect nDCG@10 | 0.851 | 0.756 | 0.815 | **0.851** | Effectively BM25 / Cross-Encoder |
| Known-item Success@10 | 0.667 | 0.667 | 0.667 | 0.667 | Tie |
| Known-item MRR@10 | **0.667** | 0.500 | **0.667** | **0.667** | Three-way tie |
| Known-item nDCG@10 | **0.667** | 0.398 | 0.596 | **0.667** | BM25 / Cross-Encoder |

The cross-encoder recovered the strong BM25 precision and nDCG while retaining
the hybrid system's best MRR. The cross-encoder nDCG exceeds BM25 by only
0.000015, which is not meaningful on six silver-label aspect queries and should
be treated as a tie.

## Latency

| Measurement | Seconds |
|---|---:|
| Candidate text lookup | 1.21 |
| Cached reranker model load | 0.266 |
| Reranker warm-up | 0.021 |
| Mean reranking latency for 20 candidates | 0.442 |
| Mean reconstructed sequential latency | 94.66 |
| Mean estimated parallel latency | 94.09 |
| Combined retrieval and reranker cold initialization | 397.70 |

Reranking overhead is small. End-to-end latency remains dominated by the
current Python BM25 scan. Latency was reconstructed from the saved Hybrid RRF
run plus directly measured reranking time.

## Error analysis

- Reranking restored all 10 top results for each of the first two known-item
  queries to the correct product. The generic protection-plan query still
  failed because the target was absent from the top 20 hybrid candidates.
- The display query improved from one relevant result in the hybrid top 5 to
  three after reranking. Its first result remained labeled `design/appearance`,
  while relevant `screen/display` results moved to ranks 2 and 3.
- The build-quality and price/value queries achieved perfect P@5 after
  reranking.
- Sound-quality P@5 decreased from 1.0 to 0.8, showing that reranking can also
  move silver-label non-relevant chunks upward.
- Candidate recall remains a hard ceiling: a reranker cannot recover a target
  that neither first-stage retriever supplied.

## Reproduction

Run from the `electronics_kb` directory after the Hybrid RRF result exists:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
.\.venv\Scripts\python.exe scripts\run_cross_encoder_baseline.py
```

The model is cached under `model_cache/cross_encoder/` on this machine. The run
creates:

- `results/cross_encoder_pilot_results.jsonl`
- `results/cross_encoder_pilot_metrics.json`
- `results/cross_encoder_pilot_metrics.csv`

## Limitations and next step

- The pilot contains only nine queries and uses keyword-derived silver labels.
- The reranker was trained on MS MARCO, not Amazon Electronics judgments.
- Only 20 hybrid candidates per query were available for reranking.
- End-to-end timing is reconstructed from separate retrieval and reranking runs.
- The next project step should be a manually judged query set and qrels,
  followed by a formal rerun and tuning of candidate depth, RRF parameters, and
  reranker depth.
