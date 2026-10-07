"""Cross-encoder reranking for retrieved Electronics candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Sequence

import numpy as np
from sentence_transformers import CrossEncoder


@dataclass(frozen=True)
class RerankedResult:
    """One result after cross-encoder reranking."""

    chunk_id: str
    score: float
    rank: int
    original_rank: int
    retrieval_score: float
    metadata: dict

    def to_dict(self) -> dict:
        payload = asdict(self)
        metadata = payload.pop("metadata")
        payload.update(metadata)
        return payload


class CrossEncoderReranker:
    """Rerank query-document pairs with a sentence-transformers model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        cache_folder: str | Path | None = None,
        max_length: int = 512,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.cache_folder = Path(cache_folder) if cache_folder else None
        self.max_length = max_length
        self.device = device
        self.model = None
        self.load_seconds = 0.0

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> "CrossEncoderReranker":
        if self.is_loaded:
            return self
        started = time.perf_counter()
        self.model = CrossEncoder(
            self.model_name,
            cache_folder=str(self.cache_folder) if self.cache_folder else None,
            max_length=self.max_length,
            device=self.device,
        )
        self.load_seconds = time.perf_counter() - started
        return self

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict],
        top_k: int | None = None,
        batch_size: int = 16,
    ) -> tuple[list[RerankedResult], float]:
        if not self.is_loaded:
            raise RuntimeError("Call load() before rerank().")
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if not candidates:
            return [], 0.0
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1 when provided.")
        for candidate in candidates:
            if not candidate.get("text"):
                raise ValueError(
                    f"Candidate is missing text: {candidate.get('chunk_id')}"
                )

        started = time.perf_counter()
        pairs = [(query, candidate["text"]) for candidate in candidates]
        raw_scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
        elapsed = time.perf_counter() - started
        if scores.size != len(candidates):
            raise ValueError("The model returned an unexpected number of scores.")

        ignored_keys = {
            "chunk_id",
            "score",
            "rank",
            "relevant",
            "original_rank",
            "retrieval_score",
        }
        order = sorted(
            range(len(candidates)),
            key=lambda index: (-float(scores[index]), int(candidates[index]["rank"])),
        )
        if top_k is not None:
            order = order[:top_k]

        results = []
        for rank, index in enumerate(order, start=1):
            candidate = candidates[index]
            results.append(
                RerankedResult(
                    chunk_id=str(candidate["chunk_id"]),
                    score=float(scores[index]),
                    rank=rank,
                    original_rank=int(candidate["rank"]),
                    retrieval_score=float(candidate["score"]),
                    metadata={
                        key: value
                        for key, value in candidate.items()
                        if key not in ignored_keys
                    },
                )
            )
        return results, elapsed
