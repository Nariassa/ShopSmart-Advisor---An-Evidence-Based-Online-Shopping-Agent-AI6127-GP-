# Electronics KB — Handoff for Agent Building

## Current retrieval recommendation

Formal human-confirmed evaluation ranks Dense retrieval first, with P@10 of
0.910, MRR@10 of 0.900, and nDCG@10 of 0.743. The downstream generation and
product-comparison pipeline should therefore use the reusable `DenseRetriever`
in `src/retrieval/dense_retriever.py` as its default retrieval path.

The `electronics_kb.py` `search()` function documented below is the original
hybrid compatibility interface. It remains useful for lexical-plus-semantic
experiments, but it is not the best formal configuration. See
`FORMAL_EVALUATION_README.md` for the complete comparison and limitations.

This package is a ready-to-query knowledge base of Amazon Electronics product specs and reviews.

## Setup (do this once)

1. Unzip `electronics_kb_bundle.zip`. You should see:
   ```
   chunks.parquet
   bm25_index.pkl
   chroma_db/
   electronics_kb.py
   ```
   Keep all four in the same working directory.

2. Install dependencies:
   ```
   pip install rank_bm25 chromadb sentence-transformers pandas -q
   ```

## Usage

```python
from electronics_kb import search, get_product_chunks

# Hybrid search (BM25 + dense, fused with Reciprocal Rank Fusion)
results = search("battery life on wireless earbuds", top_k=5)
for r in results:
    print(r["parent_asin"], r["chunk_type"], r["aspect"])
    print(r["text"])
    print("---")

# Restrict to only spec chunks or only review chunks
spec_only = search("waterproof rating", chunk_type="spec")
review_only = search("customer complaints about durability", chunk_type="review")

# Get every chunk for one specific product (not a search — direct lookup)
product_chunks = get_product_chunks("B07C7QKM3M")
```

## Result schema

Every result from `search()` is a dict:

| Field | Meaning |
|---|---|
| `chunk_id` | Unique id, e.g. `"B07C7QKM3M_spec"` or `"B07C7QKM3M_review_12"` |
| `score` | Fused relevance score (higher = more relevant). Not a probability — only meaningful for ranking within one query's results, not comparable across different queries |
| `parent_asin` | Amazon product ID — use this to group/cross-reference chunks from the same product |
| `chunk_type` | `"spec"` or `"review"` |
| `aspect` | e.g. `"battery life"`, `"price/value"`, `"specifications"` (spec chunks are always `"specifications"`; review chunks are aspect-classified via keyword matching) |
| `text` | The actual chunk text. Already prefixed with `[Product: <title> | ASIN: <asin>]` context, so it's usable standalone in an LLM prompt without needing to look up the product separately |

## Things worth knowing before building on top of this

- **One product = one spec chunk + multiple review chunks.** A product with reviews touching 5 different aspects will have 5 separate review chunks (one per aspect group), all sharing the same `parent_asin`. Use `get_product_chunks()` if your agent needs the full picture on one product rather than search relevance.
- **`price` is not included in spec chunks.** It was 67-88% missing across every sub-category in the source data, so it was dropped rather than polluting specs with frequent nulls. If your agent needs pricing, that's a gap to fill from elsewhere.
- **Aspect labels are keyword-based, not model-based.** A zero-shot classifier was tested but was computationally infeasible at this corpus size (~675 hours estimated). Keyword matching against a 10-aspect taxonomy was validated against a zero-shot sample — see `aspect_validation_report.json` if you want the agreement rate — and used for the full corpus. Expect occasional misclassification on ambiguous sentences.
- **Chunk text is capped at 2000 characters** and the corpus is embedded with `BAAI/bge-base-en-v1.5` (512 token limit) — don't expect a chunk to contain an entire long product description verbatim if the source was longer.
- **`search()` loads everything lazily on first call** (a few seconds for BM25/Chroma, plus the embedding model), then keeps it in memory for the rest of the process — no need to reload per query.

## If you need the underlying build process

The full pipeline (sanitization, chunking, indexing) that produced these
artifacts is in the other files in this folder (`step1_sanitize.py` through
`step5_app.py`, `config.py`, `README.md`) — only needed if you want to
regenerate the KB from raw data or change chunking/aspect logic, not for
building an agent on top of the existing index.
