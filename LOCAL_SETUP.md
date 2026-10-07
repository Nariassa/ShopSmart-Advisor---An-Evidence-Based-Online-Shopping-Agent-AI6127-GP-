# Amazon Electronics KB — Local setup

This folder contains the local dataset and indexes for the AI6127 retrieval work.

## Dataset inventory

- `chunks.parquet`: chunked Amazon Electronics product specifications and reviews.
- `bm25_index.pkl`: prebuilt BM25 keyword index.
- `embedding_shards/`: 45 embedding/id shard pairs (4,450,749 vectors, 768 dimensions, float16).
- `chroma_db/`: prebuilt Chroma dense-vector index.
- `electronics_kb.py`: hybrid BM25 + dense retrieval with Reciprocal Rank Fusion (RRF).

The corpus has 4,450,749 chunks. Its columns are `chunk_id`, `parent_asin`,
`chunk_type`, `aspect`, `text`, `rating`, and `category`.

## Environment

From PowerShell in this directory:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "from electronics_kb import search; print(search('battery life on wireless earbuds', top_k=3))"
```

To recreate the environment:

```powershell
py -3.10 -m pip install virtualenv
py -3.10 -m virtualenv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The first query may download `BAAI/bge-base-en-v1.5` and needs substantially
more RAM than later queries because the Parquet corpus, BM25 index, embedding
model, and Chroma index are loaded lazily.

The model is already cached under `model_cache/` on this machine. To keep a
fresh installation's cache on D:, set these before the first query:

```powershell
$env:HF_HOME="$PWD\model_cache"
$env:SENTENCE_TRANSFORMERS_HOME="$PWD\model_cache\sentence_transformers"
```

## Validation completed

- Parquet rows: **4,450,749**.
- Embedding shards: **45**, totalling **4,450,749 x 768** float16 vectors.
- BM25 `chunk_ids`: **4,450,749**; BM25 `corpus_size`: **4,450,749**.
- Chroma collection: `electronics_kb`; count: **4,450,749**; distance: cosine.
- Dense smoke query `battery life on wireless earbuds` returned three matching
  `battery life` review chunks with valid ASINs and metadata.

## Rebuild and recovery

- `download_kb.py` and `download_manifest.json` provide resumable downloads of
  the source artifacts and all 45 shard pairs.
- `build_chroma_local.py` reconstructs `chroma_db/` from the Parquet file and
  embedding shards without a GPU. This is slower than unpacking the supplied
  prebuilt index but is useful as a reproducible fallback.
- The `chroma_db_incomplete_*` directories are retained partial local builds;
  they are not used by `electronics_kb.py`.

## BM25 pilot baseline

The first BM25 baseline is documented in `BM25_BASELINE_README.md`. Its reusable
retriever is in `src/retrieval/bm25_retriever.py`, and the experiment runner is
`scripts/run_bm25_baseline.py`.

## Dense pilot baseline

The dense baseline is documented in `DENSE_BASELINE_README.md`. Its reusable
retriever is in `src/retrieval/dense_retriever.py`, and the experiment runner is
`scripts/run_dense_baseline.py`. A direct aggregate comparison is stored in
`results/baseline_comparison.csv`.

## Hybrid RRF pilot baseline

The hybrid baseline is documented in `HYBRID_RRF_BASELINE_README.md`. Its RRF
implementation is in `src/retrieval/hybrid_retriever.py`, and the offline replay
runner is `scripts/run_hybrid_rrf_baseline.py`. It fuses the top 20 saved BM25
and dense candidates per query with `rrf_k=60`.

## Cross-encoder reranking pilot baseline

The reranking baseline is documented in `CROSS_ENCODER_BASELINE_README.md`. Its
implementation is `src/retrieval/cross_encoder_reranker.py`, and the runner is
`scripts/run_cross_encoder_baseline.py`. It reranks the top 20 Hybrid RRF
candidates with `cross-encoder/ms-marco-MiniLM-L6-v2` on CPU.

## Completed baseline suite

All four proposal baselines are complete. See `BASELINE_SUITE_README.md` for the
consolidated comparison and run `scripts/verify_baseline_suite.py` to validate
all artifacts and regenerate `results/baseline_suite_manifest.json`.

## Important data caveats

- Price is omitted from specification chunks because it was mostly missing in
  the source data.
- Review aspect labels are keyword-based, not model-classified.
- Chunk text is capped at 2,000 characters; the embedding model has a 512-token
  limit.
- RRF scores are ranking scores within one query, not probabilities and not
  directly comparable across queries.
