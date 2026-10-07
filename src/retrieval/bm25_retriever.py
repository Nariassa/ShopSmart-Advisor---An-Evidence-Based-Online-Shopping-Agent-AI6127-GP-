"""Memory-conscious BM25 retrieval over the prebuilt Electronics index."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
import time

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    """One ranked BM25 search result."""

    chunk_id: str
    score: float
    rank: int

    def to_dict(self) -> dict:
        return asdict(self)


class BM25Retriever:
    """Load and query the supplied rank_bm25 index.

    Tokenization intentionally matches the handoff implementation:
    lowercase text followed by whitespace splitting.
    """

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self.bm25 = None
        self.chunk_ids = None
        self.load_seconds = 0.0

    @property
    def is_loaded(self) -> bool:
        return self.bm25 is not None and self.chunk_ids is not None

    @property
    def corpus_size(self) -> int:
        if not self.is_loaded:
            raise RuntimeError("The BM25 index is not loaded.")
        return int(self.bm25.corpus_size)

    def load(self) -> "BM25Retriever":
        if self.is_loaded:
            return self
        if not self.index_path.is_file():
            raise FileNotFoundError(f"BM25 index not found: {self.index_path}")

        started = time.perf_counter()
        with self.index_path.open("rb") as handle:
            payload = pickle.load(handle)
        self.load_seconds = time.perf_counter() - started

        if not isinstance(payload, dict) or not {"bm25", "chunk_ids"} <= payload.keys():
            raise ValueError("BM25 index must contain 'bm25' and 'chunk_ids'.")

        self.bm25 = payload["bm25"]
        self.chunk_ids = payload["chunk_ids"]
        if len(self.chunk_ids) != self.bm25.corpus_size:
            raise ValueError(
                "BM25 corpus size does not match the number of stored chunk IDs: "
                f"{self.bm25.corpus_size:,} != {len(self.chunk_ids):,}"
            )
        return self

    @staticmethod
    def tokenize(query: str) -> list[str]:
        return query.lower().split()

    def search(self, query: str, top_k: int = 10) -> tuple[list[SearchResult], float]:
        if not self.is_loaded:
            raise RuntimeError("Call load() before search().")
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        started = time.perf_counter()
        scores = self._score_single_pass(self.tokenize(query))
        k = min(top_k, scores.size)

        # argpartition avoids a full O(N log N) sort over 4.45 million scores.
        candidate_positions = np.argpartition(scores, -k)[-k:]
        ordered_positions = candidate_positions[
            np.argsort(scores[candidate_positions], kind="stable")[::-1]
        ]
        elapsed = time.perf_counter() - started

        results = [
            SearchResult(
                chunk_id=str(self.chunk_ids[position]),
                score=float(scores[position]),
                rank=rank,
            )
            for rank, position in enumerate(ordered_positions, start=1)
        ]
        return results, elapsed

    def _score_single_pass(self, query_tokens: list[str]) -> np.ndarray:
        """Compute the standard BM25 score with one sequential corpus scan.

        rank_bm25 scans every document once per query token. On a machine with
        less RAM than the 4.45-million-document index, those repeated scans
        trigger severe paging. This implementation uses the same formula and
        stored index statistics while visiting each document dictionary once.
        """

        scores = np.zeros(self.bm25.corpus_size, dtype=np.float64)
        weighted_terms = [
            (term, float(self.bm25.idf.get(term) or 0.0))
            for term in query_tokens
        ]
        k1 = float(self.bm25.k1)
        b = float(self.bm25.b)
        average_length = float(self.bm25.avgdl)
        numerator_scale = k1 + 1.0

        for position, frequencies in enumerate(self.bm25.doc_freqs):
            length_normalizer = None
            score = 0.0
            for term, inverse_document_frequency in weighted_terms:
                frequency = frequencies.get(term)
                if not frequency:
                    continue
                if length_normalizer is None:
                    length_normalizer = k1 * (
                        1.0 - b
                        + b * float(self.bm25.doc_len[position]) / average_length
                    )
                score += inverse_document_frequency * (
                    frequency * numerator_scale
                    / (frequency + length_normalizer)
                )
            scores[position] = score
        return scores
