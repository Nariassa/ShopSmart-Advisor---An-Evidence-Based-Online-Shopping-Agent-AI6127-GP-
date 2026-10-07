"""Unit tests for formal graded retrieval metrics."""

from pathlib import Path
import math
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_formal_evaluation import dcg, query_metrics  # noqa: E402


class FormalEvaluationMetricTests(unittest.TestCase):
    def test_dcg_uses_exponential_graded_gain(self) -> None:
        expected = 3.0 + 1.0 / math.log2(3) + 0.0
        self.assertAlmostEqual(dcg([2, 1, 0], 3), expected)

    def test_binary_and_graded_metrics_are_separate(self) -> None:
        metrics = query_metrics([1, 0, 2, 0, 0], [2, 2, 1, 0, 0])
        self.assertEqual(metrics["precision_at_5"], 0.4)
        self.assertEqual(metrics["highly_relevant_precision_at_5"], 0.2)
        self.assertEqual(metrics["pool_recall_at_5"], 2 / 3)
        self.assertEqual(metrics["mrr_at_5"], 1.0)
        self.assertGreater(metrics["ndcg_at_5"], 0.0)
        self.assertLess(metrics["ndcg_at_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
