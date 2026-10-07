# Formal Retrieval and Qrels Pool

## Status

All four frozen retrieval configurations have been run on the 20-query formal
evaluation set. Each system saved 20 unique candidates per query. Human-validated
formal evaluation is complete; see `FORMAL_EVALUATION_README.md`.

## Frozen Configuration

1. BM25 with lowercase whitespace tokenization
2. Dense retrieval with `BAAI/bge-base-en-v1.5`
3. Reciprocal Rank Fusion with `rrf_k=60` over the BM25 and dense top 20
4. `cross-encoder/ms-marco-MiniLM-L6-v2` reranking of the hybrid top 20

All runs used the same 4,450,749-chunk corpus and the same frozen query file.
The query file SHA-256 digest is
`3107885e0dd3579f111308dc94a9c78e4aef32b0353c90b11b214f9aee749321`.

## Latency Summary

| System | Mean per query | Median | P95 | Additional note |
|---|---:|---:|---:|---|
| BM25 | 100.342 s | 102.151 s | 109.299 s | Cold index load: 131.673 s |
| Dense | 0.395 s | 0.379 s | 0.593 s | Model plus collection open: 2.891 s |
| Hybrid RRF | 100.737 s | 102.589 s | 109.892 s | Sequential reconstructed latency |
| Cross-Encoder | 101.110 s | 102.802 s | 110.301 s | Mean reranking overhead: 0.373 s |

Hybrid and cross-encoder totals include the measured source-run latencies. They
are not fresh end-to-end measurements in one shared process.

## Candidate Pool

Pooling the top 20 from all four systems produced 791 unique query--chunk pairs:

- 37 to 40 candidates per query
- 39.55 candidates per query on average
- 0.45 shared BM25/dense candidates per query on average
- Every hybrid candidate is contained in the BM25/dense source union
- Every cross-encoder candidate is contained in the hybrid candidate set

The low BM25/dense overlap shows that the two retrieval methods contribute
substantially different evidence. The annotation pool therefore keeps the full
union rather than sampling one system preferentially.

## Artifacts

- `evaluation/formal_queries_v1.jsonl`: frozen 20-query set
- `results/formal_bm25_results.jsonl`: BM25 rankings
- `results/formal_dense_results.jsonl`: dense rankings
- `results/formal_hybrid_rrf_results.jsonl`: RRF rankings
- `results/formal_cross_encoder_results.jsonl`: reranked rankings
- `results/formal_retrieval_manifest.json`: run integrity manifest
- `evaluation/formal_annotation_pool.jsonl`: blinded pooled candidates
- `evaluation/formal_annotation_pool_manifest.json`: pooling statistics
- `evaluation/Amazon_Electronics_Final_Qrels_Double_Annotation.xlsx`: final
  double-annotation workbook

## Reproduction

Run the stages in order from the project directory:

```powershell
.\.venv\Scripts\python.exe scripts\run_formal_retrieval.py --stage bm25
.\.venv\Scripts\python.exe scripts\run_formal_retrieval.py --stage dense
.\.venv\Scripts\python.exe scripts\run_formal_retrieval.py --stage hybrid
.\.venv\Scripts\python.exe scripts\run_formal_retrieval.py --stage rerank
.\.venv\Scripts\python.exe scripts\run_formal_retrieval.py --stage verify
.\.venv\Scripts\python.exe scripts\build_formal_annotation_pool.py
```

The BM25 and dense stages save each completed query and can resume from a valid
prefix of the formal query set.

## Evaluation result

A project member reviewed and confirmed all 791 labels. Dense retrieval ranks
first with P@10 of 0.910, MRR@10 of 0.900, and nDCG@10 of 0.743. The next
retrieval experiment should test cross-encoder reranking over a larger dense or
hybrid candidate set. The generation pipeline can use Dense as the current
default while that experiment is pending.
