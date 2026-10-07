"""Tests for Reciprocal Rank Fusion."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.hybrid_retriever import reciprocal_rank_fusion  # noqa: E402


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_document_present_in_both_lists_ranks_first(self) -> None:
        bm25 = [
            {"chunk_id": "a", "rank": 1, "score": 8.0, "aspect": "general"},
            {"chunk_id": "shared", "rank": 2, "score": 7.0, "aspect": "battery life"},
        ]
        dense = [
            {"chunk_id": "shared", "rank": 1, "score": 0.9, "aspect": "battery life"},
            {"chunk_id": "b", "rank": 2, "score": 0.8, "aspect": "general"},
        ]
        results = reciprocal_rank_fusion(
            [("bm25", bm25), ("dense", dense)], rrf_k=60, top_k=3
        )
        self.assertEqual(results[0].chunk_id, "shared")
        self.assertEqual(results[0].sources, ["bm25", "dense"])
        self.assertEqual(results[0].source_ranks, {"bm25": 2, "dense": 1})
        self.assertAlmostEqual(results[0].score, 1 / 62 + 1 / 61)

    def test_ties_are_deterministic(self) -> None:
        results = reciprocal_rank_fusion(
            [
                ("bm25", [{"chunk_id": "b", "rank": 1}]),
                ("dense", [{"chunk_id": "a", "rank": 1}]),
            ],
            rrf_k=60,
            top_k=2,
        )
        self.assertEqual([row.chunk_id for row in results], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
