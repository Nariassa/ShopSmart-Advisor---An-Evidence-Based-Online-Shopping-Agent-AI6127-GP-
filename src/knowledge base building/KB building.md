# ShopSmart Advisor: Knowledge Base Construction

This README documents how the ShopSmart knowledge base (KB) was built, from the raw Amazon Reviews 2023 Electronics data to the BM25 and dense (ChromaDB) indexes. The steps run in order in the Colab notebook `KB building` (steps 1 to 5). Counts below are copied from the notebook outputs.

## 1. Overview

| Item | Value |
|---|---|
| Source | Amazon Reviews 2023 (McAuley Lab, Hugging Face), Electronics category (reviews + item metadata) |
| Products with metadata | 1,610,012 |
| Electronics products | 789,923 |
| Products kept (complete records) | 198,592 |
| Reviews (raw, after filters, after cleaning) | 1,803,756, then 1,783,584, then 1,770,864 |
| Chunks | 4,450,749 (198,592 spec + 4,252,157 review) |
| Sparse index | BM25 (`rank_bm25`, in-process), `bm25_index.pkl` |
| Dense index | `BAAI/bge-base-en-v1.5` embeddings in ChromaDB (HNSW, cosine) |

## 2. Pipeline

```
raw reviews + metadata (Parquet, queried with DuckDB)
  -> product subset (Computers, Camera & Photo, Cell Phones & Accessories, Portable Audio & Accessories, Apple products)
  -> review subset and filtering
  -> cleaning (language ID, PII masking, normalisation)       -> reviews_clean.parquet
  -> chunking (1 spec chunk per product + aspect-split review chunks) -> chunks.parquet
  -> indexing: BM25 (bm25_index.pkl) + BGE embeddings -> ChromaDB
  -> corpus statistics                                         -> corpus_stats.json
```

### Step 1. Collection and product subset
- Raw data is read as Parquet with DuckDB, so the full files never have to be loaded into memory.
- Products are restricted to Electronics (789,923 of 1,610,012) and then to those with the fields the chunker needs (title, features, description, details, average rating, main category). This leaves 198,592 products.

### Step 2. Review subset and cleaning
- Reviews are joined to the kept products (1,803,756 reviews), filtered (1,783,584), then cleaned (1,770,864).
- Cleaning steps:
  - **Language ID:** fastText `lid.176`; non-English reviews are dropped.
  - **PII masking:** regex masking of emails, phone numbers and URLs.
  - **Normalisation:** contractions expanded and a curated slang map applied.
- Output: `reviews_clean.parquet`.

### Step 3. Chunking
Two kinds of chunks are built and combined into `chunks.parquet`.

| chunk_type | How it is built | Count |
|---|---|---|
| `spec` | One per product. `chunk_id = <parent_asin>_spec`, `aspect = "specifications"`. Text is the context prefix plus `Features: ... | Specifications: ... | Description: ...` | 198,592 |
| `review` | Each review is split into sentences (spaCy sentencizer). Sentences are grouped by aspect, giving one chunk per (review, aspect). `chunk_id = <parent_asin>_review_<n>` | 4,252,157 |

- **Context injection:** every chunk text starts with `[Product: <title> | ASIN: <parent_asin>] `, so a chunk is still interpretable on its own.
- **Metadata per chunk:** `chunk_id`, `parent_asin`, `chunk_type`, `aspect`, `text`, `rating` (product average rating), `category` (main category). The shared `parent_asin` links spec and review chunks of the same product.
- **Aspect assignment is keyword-based, not zero-shot.** `ASPECT_KEYWORDS` maps 10 aspects to keyword lists. `assign_aspect_keyword(sentence)` returns the aspect with the most keyword hits, or `general` if none match.
- **Why not zero-shot:** a zero-shot BART-MNLI tagger was tried first but ran at about 4.3 sentences/s. For about 10.5M sentences that is roughly 675 hours, so it was replaced by keywords.
- Review chunking is slow (single process): the log shows about 100,000 chunks per 12 minutes, so the full run took many hours on Colab. Save `chunks.parquet` as soon as it finishes.

Aspect distribution of all 4,450,749 chunks:

| Aspect | Chunks |
|---|---|
| general (no keyword hit) | 1,534,726 |
| price/value | 460,441 |
| connectivity | 313,514 |
| build quality | 312,062 |
| design/appearance | 302,166 |
| battery life | 285,714 |
| ease of use | 276,291 |
| specifications/performance | 227,266 |
| customer service | 217,752 |
| specifications (spec chunks) | 198,592 |
| screen/display | 185,481 |
| sound quality | 136,744 |

### Step 4. Indexing
**Sparse (BM25).**
- `rank_bm25.BM25Okapi` over the lower-cased, whitespace-tokenised chunk text (all 4,450,749 chunks).
- Persisted as `bm25_index.pkl` (`{"bm25": ..., "chunk_ids": [...]}`).
- Elasticsearch was the original plan; it was replaced by this in-process index because of an access issue.

**Dense (ChromaDB).**
- Model: `BAAI/bge-base-en-v1.5` (GPU, L4). Collection uses `hnsw:space = cosine`.
- Chunk text is capped at `MAX_CHARS = 2000` before embedding (about 400 to 500 tokens, under the 512-token limit).
- Chunk length is highly skewed (characters): mean 420.6, median 273, 75th percentile 392, max 37,026. 281,560 chunks are over 1,000 characters.
- **Speed-ups that made the run feasible:**
  1. Truncate to 2,000 characters.
  2. Sort chunks by length before batching, so padding is minimal.
  3. Raise the encode batch size from 128 to 256.
  - Measured throughput went from 72 to 312 chunks/s, and the estimated total from about 1,037 min to about 237 min.
- Stored metadata per vector: `parent_asin`, `chunk_type`, `aspect`, `category`, `rating`.
- **Resumable sharded embedding.** A Colab runtime reset wiped the local Chroma index, so embedding was redone in 45 shards of 100,000 chunks:
  - Deterministic order: stable sort by (`text_len`, `chunk_id`), so shard boundaries are identical on every run.
  - Each shard is saved to Drive as float16, L2-normalised `shard_XXXX_emb.npy` plus `shard_XXXX_ids.npy`. Re-running skips finished shards.
  - Chroma is then rebuilt on local disk from the saved shards (no GPU needed).

**Hybrid retrieval.** Dense and BM25 rankings are combined at query time with Reciprocal Rank Fusion (`RRF_K = 60`), then optionally re-ranked with a cross-encoder. Shared settings (paths, `EMBED_MODEL_NAME`, `RRF_K`, `TOP_K_DEFAULT`) live in `config.py`.

### Step 5. Corpus statistics
Corpus and token-distribution statistics are produced by the stats cell and saved to `corpus_stats.json` (see the report, section 1.6).

## 3. Outputs

| File | Description |
|---|---|
| `reviews_clean.parquet` | Cleaned review table |
| `chunks.parquet` | 4,450,749 chunks with metadata |
| `bm25_index.pkl` | BM25 index plus chunk id list |
| `chroma_db/` | Persistent ChromaDB collection (HNSW, cosine) |
| `embedding_shards/` | 45 float16 embedding shards + id arrays |
| `electronics_kb_bundle.zip` | `chunks.parquet`, `bm25_index.pkl`, `config.py`, `chroma_db/` |
| `corpus_stats.json` | Corpus and token statistics |

## 4. How to re-run

1. Open the notebook in Colab with a GPU runtime (L4 was used).
2. Install: `pip install duckdb pyarrow pandas fasttext spacy rank_bm25 chromadb sentence-transformers`.
3. Run steps 1 to 3 in order, saving `reviews_clean.parquet` and `chunks.parquet` to Drive.
4. Run the BM25 cell, then the sharded embedding script (safe to restart), then build Chroma from the shards.
5. Copy `chunks.parquet`, `bm25_index.pkl`, `config.py` and `chroma_db/` into the app folder.

Notes: Drive uploads of `chunks.parquet` must be complete; a truncated copy fails with "Parquet magic bytes not found in footer".

## 5. Known limitations

- **Keyword aspects are coarse.** 1,534,726 chunks (34.5% of all chunks, and about 36.1% of review chunks) fall into `general`. A sentence with no keyword match, or mixed topics, may be tagged imperfectly.
- **PII masking is regex-only.** Emails, phones and URLs are masked, but personal names are not.
- **Price is not stored** in chunks, so price questions rely on review text about value.
- **Long chunks are truncated for embedding** (2,000 characters), while BM25 sees the full text.
- **BM25 scale.** `rank_bm25` scores every document in memory for each query, which is slow at 4.45M chunks. A production system would use Elasticsearch/OpenSearch or a sparse inverted index.
- **Embedding precision.** Shards are stored as float16 (small storage saving, negligible expected effect on retrieval).
- **Slang handling.** The slang map is curated and small, so some informal text remains.
