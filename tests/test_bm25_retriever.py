"""Regression tests for the local BM25 retriever."""

from pathlib import Path
import sys
import unittest

import numpy as np
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval import BM25Retriever  # noqa: E402


class BM25RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        corpus = [
            "wireless earbuds long battery life".split(),
            "camera lens optical zoom".split(),
            "battery drains quickly phone".split(),
            "clear sound wireless headphones".split(),
        ]
        bm25 = BM25Okapi(corpus)
        self.retriever = BM25Retriever("unused.pkl")
        self.retriever.bm25 = bm25
        self.retriever.chunk_ids = [f"doc_{i}" for i in range(len(corpus))]

    def test_single_pass_scores_match_rank_bm25(self) -> None:
        queries = [
            "wireless battery",
            "camera zoom",
            "battery battery phone",
            "missing token",
        ]
        for query in queries:
            tokens = query.split()
            expected = self.retriever.bm25.get_scores(tokens)
            actual = self.retriever._score_single_pass(tokens)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_search_returns_ranked_results(self) -> None:
        results, elapsed = self.retriever.search("camera zoom", top_k=2)
        self.assertEqual(results[0].chunk_id, "doc_1")
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertGreaterEqual(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
