"""Reciprocal Rank Fusion for combining retrieval result lists."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class HybridSearchResult:
    """One result produced by Reciprocal Rank Fusion."""

    chunk_id: str
    score: float
    rank: int
    sources: list[str]
    source_ranks: dict[str, int]
    source_scores: dict[str, float]
    metadata: dict

    def to_dict(self) -> dict:
        payload = asdict(self)
        metadata = payload.pop("metadata")
        payload.update(metadata)
        return payload


def reciprocal_rank_fusion(
    rankings: Sequence[tuple[str, Sequence[dict]]],
    rrf_k: int = 60,
    top_k: int = 20,
) -> list[HybridSearchResult]:
    """Fuse ranked lists with standard RRF scoring.

    Each input result must contain `chunk_id` and may contain `rank` and
    `score`. Ranks default to the one-based position in the supplied list.
    """

    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not rankings:
        raise ValueError("At least one ranking is required.")

    fused: dict[str, dict] = {}
    ignored_metadata_keys = {
        "chunk_id",
        "score",
        "distance",
        "rank",
        "relevant",
        "sources",
        "source_ranks",
        "source_scores",
    }

    for source_name, ranking in rankings:
        seen_ids: set[str] = set()
        for position, result in enumerate(ranking, start=1):
            chunk_id = str(result["chunk_id"])
            if chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk_id in {source_name} ranking: {chunk_id}"
                )
            seen_ids.add(chunk_id)
            source_rank = int(result.get("rank", position))
            entry = fused.setdefault(
                chunk_id,
                {
                    "score": 0.0,
                    "sources": [],
                    "source_ranks": {},
                    "source_scores": {},
                    "metadata": {
                        key: value
                        for key, value in result.items()
                        if key not in ignored_metadata_keys
                    },
                },
            )
            entry["score"] += 1.0 / (rrf_k + source_rank)
            entry["sources"].append(source_name)
            entry["source_ranks"][source_name] = source_rank
            if "score" in result:
                entry["source_scores"][source_name] = float(result["score"])

    ordered = sorted(
        fused.items(),
        key=lambda item: (
            -item[1]["score"],
            min(item[1]["source_ranks"].values()),
            item[0],
        ),
    )[:top_k]
    return [
        HybridSearchResult(
            chunk_id=chunk_id,
            score=float(entry["score"]),
            rank=rank,
            sources=list(entry["sources"]),
            source_ranks=dict(entry["source_ranks"]),
            source_scores=dict(entry["source_scores"]),
            metadata=dict(entry["metadata"]),
        )
        for rank, (chunk_id, entry) in enumerate(ordered, start=1)
    ]
