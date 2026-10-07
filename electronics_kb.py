"""
electronics_kb.py — Standalone retrieval interface for the Electronics KB.

For teammates building agents/tools on top of this knowledge base.
You do NOT need to know how the index was built — just:

    from electronics_kb import search

    results = search("battery life on wireless earbuds", top_k=5)
    for r in results:
        print(r["parent_asin"], r["chunk_type"], r["aspect"], r["text"])

Each result dict has:
    chunk_id      — unique id, e.g. "B07C7QKM3M_spec" or "B07C7QKM3M_review_12"
    score         — fused relevance score (higher = more relevant)
    parent_asin   — Amazon product ID, use this to cross-reference/group results
    chunk_type    — "spec" or "review"
    aspect        — e.g. "battery life", "price/value", "specifications"
    text          — the actual chunk text (already includes product title/ASIN context)

Setup (one-time, per environment):
    1. Unzip electronics_kb_bundle.zip so chunks.parquet, bm25_index.pkl,
       and chroma_db/ are all in the same directory as this file.
    2. pip install rank_bm25 chromadb sentence-transformers pandas -q
"""

import pickle
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "chunks.parquet"
BM25_INDEX_PATH = "bm25_index.pkl"
CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "electronics_kb"
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RRF_K = 60
TOP_K_DEFAULT = 10

_chunks = None
_chunk_id_to_pos = None
_bm25 = None
_bm25_ids = None
_embed_model = None
_collection = None


def _ensure_loaded():
    """Lazy-load everything on first call, so importing this module is cheap."""
    global _chunks, _chunk_id_to_pos, _bm25, _bm25_ids, _embed_model, _collection

    if _chunks is not None:
        return  # already loaded

    _chunks = pd.read_parquet(CHUNKS_PATH)
    _chunks["chunk_id"] = _chunks["chunk_id"].astype(str)
    _chunk_id_to_pos = {cid: i for i, cid in enumerate(_chunks["chunk_id"])}

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    _bm25 = bm25_data["bm25"]
    _bm25_ids = bm25_data["chunk_ids"]

    _embed_model = SentenceTransformer(EMBED_MODEL_NAME)  # CPU is fine for query-time single embeds
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    _collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)

    print(f"Electronics KB loaded: {len(_chunks)} chunks "
          f"({(_chunks['chunk_type'] == 'spec').sum()} spec, "
          f"{(_chunks['chunk_type'] == 'review').sum()} review)")


def _bm25_search(query: str, k: int) -> list:
    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)
    top_positions = np.argsort(scores)[::-1][:k]
    # map positions via the ids saved WITH the index: the BM25 corpus was built
    # in length-sorted order, which differs from chunks.parquet's row order
    return [_bm25_ids[i] for i in top_positions]


def _dense_search(query: str, k: int) -> list:
    query_emb = _embed_model.encode([query]).tolist()
    result = _collection.query(query_embeddings=query_emb, n_results=k)
    return result["ids"][0]


def _reciprocal_rank_fusion(rankings: list, k: int = RRF_K) -> dict:
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def search(query: str, top_k: int = TOP_K_DEFAULT, candidate_k: int = 50,
           chunk_type: str = None, parent_asin: str = None) -> list:
    """
    Hybrid (BM25 + dense) search over the Electronics KB, fused via RRF.

    Args:
        query: natural language search text
        top_k: number of results to return
        candidate_k: how many candidates each method retrieves before fusion
        chunk_type: optional filter, "spec" or "review"
        parent_asin: optional filter, restrict results to one product

    Returns: list of result dicts (see module docstring for shape)
    """
    _ensure_loaded()

    bm25_ids = _bm25_search(query, k=candidate_k)
    dense_ids = _dense_search(query, k=candidate_k)
    fused = _reciprocal_rank_fusion([bm25_ids, dense_ids])

    results = []
    for doc_id in fused:
        if doc_id not in _chunk_id_to_pos:
            continue
        row = _chunks.iloc[_chunk_id_to_pos[doc_id]]
        if chunk_type and row["chunk_type"] != chunk_type:
            continue
        if parent_asin and row["parent_asin"] != parent_asin:
            continue
        results.append({
            "chunk_id": doc_id,
            "score": fused[doc_id],
            "parent_asin": row["parent_asin"],
            "chunk_type": row["chunk_type"],
            "aspect": row["aspect"],
            "text": row["text"],
        })
        if len(results) >= top_k:
            break
    return results


def get_product_chunks(parent_asin: str) -> list:
    """Convenience: fetch ALL chunks (spec + every review aspect) for one product.
    Useful for an agent that wants full context on a specific item, not search."""
    _ensure_loaded()
    matches = _chunks[_chunks["parent_asin"] == parent_asin]
    return matches.to_dict("records")


if __name__ == "__main__":
    for r in search("battery life on wireless earbuds", top_k=5):
        print(r["score"], r["chunk_type"], r["aspect"], r["text"][:100])
