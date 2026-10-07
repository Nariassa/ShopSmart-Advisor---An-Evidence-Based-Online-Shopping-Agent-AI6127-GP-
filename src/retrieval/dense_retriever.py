"""Dense retrieval over the prebuilt Electronics Chroma collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import chromadb
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class DenseSearchResult:
    """One ranked dense-retrieval result."""

    chunk_id: str
    distance: float
    score: float
    rank: int
    metadata: dict

    def to_dict(self) -> dict:
        payload = asdict(self)
        metadata = payload.pop("metadata")
        payload.update(metadata)
        return payload


@dataclass(frozen=True)
class DenseSearchTiming:
    """Latency components for one dense query."""

    encoding_seconds: float
    search_seconds: float
    total_seconds: float


class DenseRetriever:
    """Encode queries and search the supplied persistent Chroma index."""

    def __init__(
        self,
        chroma_path: str | Path,
        model_name: str = "BAAI/bge-base-en-v1.5",
        collection_name: str = "electronics_kb",
        cache_folder: str | Path | None = None,
    ) -> None:
        self.chroma_path = Path(chroma_path)
        self.model_name = model_name
        self.collection_name = collection_name
        self.cache_folder = Path(cache_folder) if cache_folder else None
        self.model = None
        self.client = None
        self.collection = None
        self.model_load_seconds = 0.0
        self.collection_open_seconds = 0.0

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.collection is not None

    @property
    def corpus_size(self) -> int:
        if not self.is_loaded:
            raise RuntimeError("The dense retriever is not loaded.")
        return int(self.collection.count())

    def load(self) -> "DenseRetriever":
        if self.is_loaded:
            return self
        if not self.chroma_path.is_dir():
            raise FileNotFoundError(f"Chroma directory not found: {self.chroma_path}")

        model_started = time.perf_counter()
        self.model = SentenceTransformer(
            self.model_name,
            cache_folder=str(self.cache_folder) if self.cache_folder else None,
        )
        self.model_load_seconds = time.perf_counter() - model_started

        collection_started = time.perf_counter()
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = self.client.get_collection(self.collection_name)
        self.collection_open_seconds = time.perf_counter() - collection_started
        return self

    def search(
        self, query: str, top_k: int = 10
    ) -> tuple[list[DenseSearchResult], DenseSearchTiming]:
        if not self.is_loaded:
            raise RuntimeError("Call load() before search().")
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        total_started = time.perf_counter()
        encoding_started = time.perf_counter()
        embedding = self.model.encode([query]).tolist()
        encoding_seconds = time.perf_counter() - encoding_started

        search_started = time.perf_counter()
        response = self.collection.query(
            query_embeddings=embedding,
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        search_seconds = time.perf_counter() - search_started
        total_seconds = time.perf_counter() - total_started

        ids = response["ids"][0]
        distances = response["distances"][0]
        metadatas = response["metadatas"][0]
        results = [
            DenseSearchResult(
                chunk_id=str(chunk_id),
                distance=float(distance),
                score=1.0 - float(distance),
                rank=rank,
                metadata=dict(metadata or {}),
            )
            for rank, (chunk_id, distance, metadata) in enumerate(
                zip(ids, distances, metadatas), start=1
            )
        ]
        timing = DenseSearchTiming(
            encoding_seconds=encoding_seconds,
            search_seconds=search_seconds,
            total_seconds=total_seconds,
        )
        return results, timing
