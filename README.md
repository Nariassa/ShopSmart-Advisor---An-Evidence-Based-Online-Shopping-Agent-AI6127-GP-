# AI6127 Amazon Electronics Retrieval Evaluation

This repository contains the retrieval and evaluation work for an Amazon
Electronics product-question-answering project. It provides reproducible
implementations of four retrieval configurations, a frozen formal query set,
human-confirmed graded relevance judgments, evaluation scripts, and the final
comparison used for downstream system design.

The large knowledge-base artifacts are deliberately not stored in Git. They
can be downloaded or rebuilt by following [Local setup](LOCAL_SETUP.md).

## Project scope

The retrieval work covers the following tasks:

1. Validate the supplied Amazon Electronics knowledge base and local indexes.
2. Implement BM25 lexical retrieval.
3. Implement dense retrieval with `BAAI/bge-base-en-v1.5` and Chroma.
4. Implement hybrid retrieval with Reciprocal Rank Fusion (RRF).
5. Rerank hybrid candidates with
   `cross-encoder/ms-marco-MiniLM-L6-v2`.
6. Build a blinded candidate pool for manual relevance assessment.
7. Evaluate all frozen systems with the same human-confirmed qrels.
8. Recommend a retrieval configuration for the generation pipeline.

## Main result

Dense retrieval is the strongest current configuration on the formal
evaluation set and is recommended for downstream generation.

| System | P@1 | P@5 | P@10 | MRR@10 | nDCG@10 | Pool Recall@20 | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.450 | 0.580 | 0.590 | 0.696 | 0.474 | 0.379 | 100.342 s |
| **Dense** | **0.850** | **0.910** | **0.910** | **0.900** | **0.743** | **0.635** | **0.395 s** |
| Hybrid RRF | 0.750 | 0.720 | 0.750 | 0.863 | 0.627 | 0.517 | 100.737 s |
| Cross-Encoder | 0.800 | 0.850 | 0.830 | 0.892 | 0.710 | 0.517 | 101.110 s |

Cross-encoder reranking improves the Hybrid RRF ordering, increasing
nDCG@10 from 0.627 to 0.710. It does not exceed Dense because it reranks only
the Hybrid top-20 candidate set. A useful follow-up experiment is to rerank the
saved Dense top 20 directly; all required candidates are already judged.

The latency figures for Hybrid and Cross-Encoder include saved component
timings. They are not fresh parallel end-to-end measurements. The current BM25
implementation dominates their total latency.

## Data and evaluation design

The knowledge base contains 4,450,749 chunks derived from Amazon Electronics
product specifications and reviews. Each chunk includes a stable `chunk_id`,
parent ASIN, chunk type, aspect, text, rating, and category where available.

The formal test set contains 20 natural-language queries. Each of the four
systems retrieved 20 candidates per query. Pooling and deduplicating these
rankings produced 791 query-chunk pairs. A project member reviewed every pair
and confirmed a three-level relevance label:

- `0`: not relevant;
- `1`: partially relevant;
- `2`: highly relevant or directly answers the query.

The final label distribution is 209 grade-0, 181 grade-1, and 401 grade-2
judgments. Binary Precision, Recall, Success, and MRR treat grades 1 and 2 as
relevant. Graded nDCG uses gains of `2^label - 1`.

Recall is pool-relative: its denominator is the set of relevant candidates in
the union of the four top-20 rankings, not all relevant chunks in the full
4.45-million-chunk corpus. Precision, MRR, and nDCG are therefore the primary
comparison metrics.

The qrels were confirmed by one human reviewer. No independent second
annotator was used, so inter-annotator agreement is not reported.

## Repository layout

```text
.
|-- src/retrieval/                  Reusable retrieval components
|   |-- bm25_retriever.py
|   |-- dense_retriever.py
|   |-- hybrid_retriever.py
|   `-- cross_encoder_reranker.py
|-- scripts/                        Experiment and evaluation entry points
|-- tests/                          Unit tests for retrieval and metrics
|-- evaluation/
|   |-- formal_queries_v1.jsonl     Frozen 20-query test set
|   |-- formal_annotation_pool.jsonl
|   |-- human_validated_annotations_v1.jsonl
|   |-- qrels_human_validated_v1.txt
|   `-- Amazon_Electronics_Final_Qrels_Double_Annotation.xlsx
|-- results/                        Frozen runs, metrics, and manifests
|-- requirements.txt
|-- LOCAL_SETUP.md                  Full local data and index setup
|-- FORMAL_RETRIEVAL_README.md      Retrieval-run protocol
|-- FORMAL_EVALUATION_README.md     Metric definitions and error analysis
`-- HANDOFF_README.md               Downstream integration guidance
```

## Requirements

- Python 3.10 or a compatible recent Python 3 release;
- enough local storage for the external corpus and indexes;
- CPU execution is supported; no GPU is required;
- internet access is needed for the first model download unless models are
  already cached.

Create a virtual environment and install the dependencies:

```powershell
py -3.10 -m virtualenv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux or macOS, replace the final command with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Knowledge-base setup

The following artifacts are intentionally excluded from Git because of their
size:

- `chunks.parquet` (the 4,450,749-chunk corpus);
- `bm25_index.pkl`;
- `embedding_shards/`;
- `chroma_db/` or `chroma_db.zip`;
- local Hugging Face model caches.

Download the shared artifacts:

```powershell
.\.venv\Scripts\python.exe download_kb.py
```

If only the embedding shards are available, build the local Chroma index on
CPU:

```powershell
.\.venv\Scripts\python.exe build_chroma_local.py
```

See [LOCAL_SETUP.md](LOCAL_SETUP.md) for the verified inventory, environment
variables, rebuild procedure, and data caveats.

## Using the recommended dense retriever

Run the following from the repository root after `chroma_db/` has been
prepared:

```python
from src.retrieval.dense_retriever import DenseRetriever

retriever = DenseRetriever(
    chroma_path="chroma_db",
    cache_folder="model_cache/sentence_transformers",
).load()

results, timing = retriever.search(
    "wireless earbuds with long battery life",
    top_k=10,
)

for result in results:
    print(result.rank, result.chunk_id, result.score, result.metadata)

print(f"Query latency: {timing.total_seconds:.3f} seconds")
```

`electronics_kb.py` remains available for compatibility with the original
hybrid interface. Formal evaluation, however, supports Dense as the current
default for the downstream generation pipeline.

## Reproducing the formal retrieval runs

The stages are intentionally separate and resumable. Run them in order:

```powershell
.\.venv\Scripts\python.exe scripts\run_formal_retrieval.py --stage bm25
.\.venv\Scripts\python.exe scripts\run_formal_retrieval.py --stage dense
.\.venv\Scripts\python.exe scripts\run_formal_retrieval.py --stage hybrid
.\.venv\Scripts\python.exe scripts\run_formal_retrieval.py --stage rerank
.\.venv\Scripts\python.exe scripts\run_formal_retrieval.py --stage verify
.\.venv\Scripts\python.exe scripts\build_formal_annotation_pool.py
```

The frozen configuration is:

- BM25 tokenization: lowercase whitespace split;
- dense model: `BAAI/bge-base-en-v1.5`;
- RRF constant: `k = 60`;
- candidate depth: 20 per source;
- reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`;
- reranking depth: 20.

The repository already contains the final human-confirmed qrels. Recalculate
the formal metrics with:

```powershell
.\.venv\Scripts\python.exe scripts\run_formal_evaluation.py
```

The evaluator validates query coverage, rank order, result depth, duplicate
chunk IDs, qrels coverage, and label ranges before writing aggregate and
per-query results.

## Tests

Run the complete test suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The current suite contains nine tests covering BM25 retrieval, dense retrieval,
RRF fusion, cross-encoder reranking, and graded evaluation metrics.

## Key outputs

- `results/formal_human_validated_comparison.csv`: compact system comparison;
- `results/formal_human_validated_metrics.json`: aggregate and per-query
  metrics;
- `results/formal_human_validated_per_query.csv`: query-level error analysis;
- `results/formal_human_evaluation_manifest.json`: hashes and integrity
  metadata;
- `evaluation/qrels_human_validated_v1.txt`: final TREC-format qrels;
- `evaluation/human_validated_annotations_v1.jsonl`: label ledger with stable
  IDs, confidence, reasons, and text hashes;
- `evaluation/Amazon_Electronics_Final_Qrels_Double_Annotation.xlsx`: final
  review workbook.

For detailed interpretation, see
[FORMAL_EVALUATION_README.md](FORMAL_EVALUATION_README.md).

## Limitations

- The formal query set has 20 queries, so conclusions should be treated as
  project-level evidence rather than a broad benchmark claim.
- Pool-relative Recall@20 does not estimate recall over the full corpus.
- The qrels use one confirmed human reviewer and do not support an
  inter-annotator-agreement statistic.
- Prices were omitted from specification chunks because source coverage was
  poor.
- Review aspects were assigned with a validated keyword taxonomy rather than a
  full-corpus neural classifier.
- Chunk text is capped at 2,000 characters, while the dense model has a
  512-token input limit.
- BM25 latency is high in the current implementation and should be optimized
  before use in an interactive path.

## Downstream handoff

The generation and product-comparison stage should consume the Dense top-k
results as its default evidence set. Preserve `chunk_id`, `parent_asin`,
`chunk_type`, `aspect`, `text`, and retrieval score so that generated claims can
be traced back to source chunks.

If reranking is added, first rerank the saved Dense top 20 with the existing
cross-encoder. This comparison needs no additional judgments. Any experiment
that retrieves beyond the frozen top-20 pools must add the new query-chunk
pairs to the annotation pool before reporting formal metrics.
