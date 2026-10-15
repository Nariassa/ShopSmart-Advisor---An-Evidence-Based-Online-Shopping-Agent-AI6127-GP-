"""Bounded Parquet access and lazy integration with Retrieval, Reranking & Evaluation's retrievers."""

from pathlib import Path
import time


class ParquetEvidenceStore:
    def __init__(self, path, batch_size=8192):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"找不到证据文件：{self.path}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.batch_size = batch_size

    def _batches(self):
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(self.path)
        yield from parquet.iter_batches(batch_size=self.batch_size)

    def lookup(self, chunk_ids):
        import pyarrow as pa
        import pyarrow.compute as pc
        ids = list(dict.fromkeys(str(item) for item in chunk_ids))
        if not ids:
            return []
        found = {}
        for batch in self._batches():
            mask = pc.is_in(batch.column("chunk_id"), value_set=pa.array(ids))
            for row in batch.filter(mask).to_pylist():
                found[row["chunk_id"]] = row
            if len(found) == len(ids):
                break
        missing = [chunk_id for chunk_id in ids if chunk_id not in found]
        if missing:
            raise ValueError(f"检索结果在 Parquet 中找不到对应片段：{missing[:5]}")
        return [found[chunk_id] for chunk_id in ids]

    def product_chunks(self, parent_asin, limit=200):
        import pyarrow.compute as pc
        if limit < 1:
            raise ValueError("limit must be positive.")
        rows = []
        for batch in self._batches():
            mask = pc.equal(batch.column("parent_asin"), str(parent_asin))
            matching = batch.filter(mask).slice(0, limit-len(rows)).to_pylist()
            rows.extend(matching)
            if len(rows) >= limit:
                break
        return rows

    def hydrate(self, results):
        """Accept Retrieval, Reranking & Evaluation's dataclasses/dicts, preserving rank and retrieval scores."""
        rows = [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in results]
        documents = {row["chunk_id"]: row for row in self.lookup([r["chunk_id"] for r in rows])}
        return [{**dict(row.get("metadata") or {}), **row, **documents[str(row["chunk_id"])]}
                for row in rows]


class RetrievalAdapter:
    """Optional end-to-end bridge. Index/model loading happens on first search."""

    def __init__(self, data_dir, mode="dense", rerank=False,
                 embedding_model="BAAI/bge-base-en-v1.5", collection="electronics_kb"):
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("Retrieval mode must be dense, bm25 or hybrid.")
        self.data_dir, self.mode, self.rerank = Path(data_dir).resolve(), mode, rerank
        self.embedding_model, self.collection = embedding_model, collection
        self.store = ParquetEvidenceStore(self.data_dir / "chunks.parquet")
        self.bm25 = self.dense = self.reranker = None

    def search(self, query, top_k=20, candidate_k=50):
        if top_k < 1 or candidate_k < top_k:
            raise ValueError("Require candidate_k >= top_k >= 1.")
        started = time.perf_counter()
        rankings = []
        if self.mode in {"bm25", "hybrid"}:
            from ..retrieval.bm25_retriever import BM25Retriever
            if self.bm25 is None:
                self.bm25 = BM25Retriever(self.data_dir / "bm25_index.pkl").load()
            results, _ = self.bm25.search(query, candidate_k)
            rankings.append(("bm25", [r.to_dict() for r in results]))
        if self.mode in {"dense", "hybrid"}:
            from ..retrieval.dense_retriever import DenseRetriever
            if self.dense is None:
                self.dense = DenseRetriever(self.data_dir / "chroma_db",
                                            model_name=self.embedding_model,
                                            collection_name=self.collection).load()
            results, _ = self.dense.search(query, candidate_k)
            rankings.append(("dense", [r.to_dict() for r in results]))
        if self.mode == "hybrid":
            from ..retrieval.hybrid_retriever import reciprocal_rank_fusion
            fused = reciprocal_rank_fusion(rankings, top_k=candidate_k)
            candidates = self.store.hydrate(fused)
        else:
            candidates = self.store.hydrate(rankings[0][1])
        if self.rerank:
            from ..retrieval.cross_encoder_reranker import CrossEncoderReranker
            if self.reranker is None:
                self.reranker = CrossEncoderReranker().load()
            ranked, _ = self.reranker.rerank(query, candidates, top_k=top_k)
            evidence = [r.to_dict() for r in ranked]
        else:
            evidence = candidates[:top_k]
        return evidence, {"mode": self.mode, "reranked": self.rerank,
                          "total_seconds": time.perf_counter()-started}
