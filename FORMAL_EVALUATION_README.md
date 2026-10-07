# Formal Retrieval Evaluation

## Status

Formal evaluation is complete for all four frozen retrieval systems. The evaluation uses 791 graded query--chunk judgments across 20 queries. A project member reviewed every pair and confirmed the labels on October 7, 2026.

The qrels use three grades: 0 for not relevant, 1 for partially relevant, and 2 for highly relevant. Binary Precision, Recall, Success, and MRR treat labels 1 and 2 as relevant. Graded nDCG uses gains of `2^label - 1`.

## Main results

| System | P@1 | P@5 | P@10 | MRR@10 | nDCG@10 | Pool Recall@20 | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.450 | 0.580 | 0.590 | 0.696 | 0.474 | 0.379 | 100.342 s |
| Dense | **0.850** | **0.910** | **0.910** | **0.900** | **0.743** | **0.635** | **0.395 s** |
| Hybrid RRF | 0.750 | 0.720 | 0.750 | 0.863 | 0.627 | 0.517 | 100.737 s |
| Cross-Encoder | 0.800 | 0.850 | 0.830 | 0.892 | 0.710 | 0.517 | 101.110 s |

Dense retrieval ranks first on the primary graded metric, nDCG@10, and also has the strongest P@5, P@10, MRR@10, and pool-relative Recall@20. It is the recommended current retrieval configuration for the generation pipeline.

Cross-encoder reranking improves the Hybrid RRF candidate order: nDCG@10 rises from 0.627 to 0.710 and P@5 rises from 0.720 to 0.850. However, it reranks the same 20 hybrid candidates, so it cannot improve the candidate-set ceiling. Hybrid and Cross-Encoder therefore have the same P@20 and pool-relative Recall@20. A useful follow-up is to rerank a larger dense or hybrid candidate set rather than only the current hybrid top 20.

BM25 performs substantially worse on these natural-language queries and dominates the latency of the sequential hybrid pipeline. The current implementation should not be used as the default interactive path without optimization.

## Query-level error analysis

Dense has the highest nDCG@10 on 12 of the 20 queries. Cross-Encoder leads on five queries, BM25 on two, and Hybrid RRF on one. Cross-Encoder exceeds Dense on eight queries, including wireless earbuds, call headphones, portable speakers, travel laptops, tablets, camera bags, and warranty support. This indicates that pairwise reranking remains useful even though the current hybrid candidate set lowers its aggregate ceiling.

The largest Dense failure is Q020, the broad warranty-support query: Dense returns no relevant item in its top 10, while BM25 and Cross-Encoder recover relevant support and replacement evidence. Dense also has lower graded ranking quality on several multi-constraint product queries even when its binary P@10 remains high. The next controlled experiment should therefore rerank the saved Dense top 20 directly. This requires no new annotation because every Dense top-20 candidate is already in the judged pool. If candidate depth is increased beyond 20, the newly retrieved pairs must be annotated before they are included in a formal comparison.

## Interpretation

The formal findings differ from the nine-query pilot because the pilot used keyword-derived silver labels, which favored lexical matches. The formal evaluation uses human-confirmed query-specific relevance judgments and is the appropriate result set for the project report.

Recall is pool-relative: its denominator contains relevant candidates in the union of the four systems' top-20 results, not every relevant chunk in the 4.45-million-chunk corpus. Precision, MRR, and nDCG are the primary comparisons. The qrels were confirmed by one human reviewer; a second independent annotator was not used, so inter-annotator agreement is not reported.

Hybrid and Cross-Encoder latency totals include saved component timings and are not fresh parallel end-to-end measurements. The Cross-Encoder adds approximately 0.37 seconds to rerank 20 candidates, while the current BM25 stage accounts for most of the total time.

## Outputs

- `evaluation/qrels_human_validated_v1.txt`: current graded qrels in TREC format.
- `evaluation/human_validated_annotations_v1.jsonl`: human-confirmed labels, reasons, confidence, stable identifiers, and text hashes.
- `evaluation/human_validation_manifest.json`: validation scope, counts, and source hashes.
- `results/formal_human_validated_metrics.json`: complete aggregate and per-query metrics.
- `results/formal_human_validated_comparison.csv`: compact system comparison.
- `results/formal_human_validated_per_query.csv`: per-query metrics for error analysis.
- `results/formal_human_evaluation_manifest.json`: evaluation integrity record and output hashes.

## Reproduction

Run from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\finalize_human_validated_qrels.py
.\.venv\Scripts\python.exe scripts\run_formal_evaluation.py
```

The evaluator checks query coverage, rank order, result depth, duplicate chunk IDs, qrels coverage, and label ranges before calculating metrics.
