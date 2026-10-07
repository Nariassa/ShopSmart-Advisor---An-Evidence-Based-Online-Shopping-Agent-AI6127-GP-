# Dense Retrieval Pilot Baseline

## Status

The dense-retrieval pilot completed successfully on the full 4,450,749-chunk
Amazon Electronics Chroma index. It used the same nine queries, relevance
definitions, retrieval depth, and metric code as the BM25 pilot.

The embedding model was `BAAI/bge-base-en-v1.5`, matching the model used to
create the supplied dense index. This remains a **silver-label pilot**, not a
replacement for manually judged queries and qrels.

## Aggregate results

| Suite | Queries | P@1 | P@5 | P@10 | Success@10 | MRR@10 | nDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aspect silver labels | 6 | 0.667 | 0.700 | 0.783 | 1.000 | 0.778 | 0.756 | 0.000032 |
| Known-item short title | 3 | 0.333 | 0.400 | 0.400 | 0.667 | 0.500 | 0.398 | 0.321 |

Aspect recall is not informative in this pilot because each aspect has a very
large relevant pool. Precision, MRR, success, and nDCG are more useful.

## Latency

| Measurement | Seconds |
|---|---:|
| Model load | 1.38 |
| Chroma collection open | 0.14 |
| Full index initialization and count | 252.57 |
| Total measured cold initialization | 254.09 |
| Index warm-up query | 1.44 |
| Mean query encoding | 0.038 |
| Mean vector search | 0.536 |
| Mean total query latency | 0.574 |
| Median total query latency | 0.618 |
| P95 total query latency | 0.717 |

The first collection-wide operation loads the large HNSW index and is slow on
the 15.68 GB RAM test machine. Once the index is resident, all nine measured
queries completed in less than one second each.

## Comparison with BM25

| Metric | BM25 | Dense | Difference |
|---|---:|---:|---:|
| Aspect P@5 | **0.900** | 0.700 | -0.200 |
| Aspect P@10 | **0.850** | 0.783 | -0.067 |
| Aspect MRR@10 | **0.875** | 0.778 | -0.097 |
| Aspect nDCG@10 | **0.851** | 0.756 | -0.095 |
| Known-item Success@10 | 0.667 | 0.667 | 0.000 |
| Known-item MRR@10 | **0.667** | 0.500 | -0.167 |
| Mean warm query latency | 93.65 s | **0.574 s** | Dense is about 163x faster |

BM25 performs better on the silver relevance labels, while dense retrieval is
dramatically faster after initialization. The quality comparison is biased in
BM25's favor because the aspect labels themselves were created with keyword
rules. A semantically useful dense result can be counted as non-relevant when
its keyword-derived aspect label differs from the target label.

## Error analysis

### Known-item retrieval

- The Dell query placed the correct specification chunk at rank 2, but only
  three of the top 10 chunks belonged to the target product.
- The Broadcom query placed the correct product at rank 1 and retrieved nine
  relevant chunks in the top 10.
- The generic Allstate protection-plan query failed for both BM25 and dense
  retrieval because many near-duplicate plan products have highly similar
  names and descriptions.

### Aspect retrieval

- Battery, sound, and connectivity retrieval performed strongly.
- For `monitor with a bright sharp screen`, the first five dense results were
  labeled `design/appearance`; the first `screen/display` result was rank 6.
- For `durable sturdy build quality`, the first result was labeled
  `price/value`, followed by relevant `build quality` chunks. These cases may
  indicate taxonomy noise rather than purely retrieval errors.

## Reproduction

Run from the `electronics_kb` directory:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
.\.venv\Scripts\python.exe scripts\run_dense_baseline.py
```

The run creates:

- `results/dense_pilot_results.jsonl`
- `results/dense_pilot_metrics.json`
- `results/dense_pilot_metrics.csv`

The shared query set is `evaluation/bm25_pilot_queries.jsonl`; the filename is
retained because BM25 created the original pilot set. Both baselines evaluate
exactly the same query records.

## Limitations and next step

- The pilot has only nine queries.
- Short-title queries overlap directly with indexed product-name prefixes.
- Keyword-derived aspect labels can under-credit semantic retrieval.
- The next ordered experiment is hybrid BM25 + dense retrieval with Reciprocal
  Rank Fusion, after a manually judged evaluation set is prepared or approved.
