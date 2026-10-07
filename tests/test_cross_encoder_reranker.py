"""Tests for cross-encoder reranking logic."""

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.cross_encoder_reranker import CrossEncoderReranker  # noqa: E402


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        return np.asarray([0.2, 0.9, 0.5])


class CrossEncoderRerankerTests(unittest.TestCase):
    def test_rerank_orders_by_model_score(self) -> None:
        reranker = CrossEncoderReranker("unused")
        reranker.model = FakeCrossEncoder()
        candidates = [
            {"chunk_id": "a", "rank": 1, "score": 0.03, "text": "first"},
            {"chunk_id": "b", "rank": 2, "score": 0.02, "text": "second"},
            {"chunk_id": "c", "rank": 3, "score": 0.01, "text": "third"},
        ]
        results, elapsed = reranker.rerank("query", candidates, top_k=3)
        self.assertEqual([row.chunk_id for row in results], ["b", "c", "a"])
        self.assertEqual([row.original_rank for row in results], [2, 3, 1])
        self.assertAlmostEqual(results[0].score, 0.9)
        self.assertGreaterEqual(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
