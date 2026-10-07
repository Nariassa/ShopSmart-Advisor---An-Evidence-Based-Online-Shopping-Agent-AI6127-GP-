"""Unit tests for the local dense retriever."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.dense_retriever import DenseRetriever  # noqa: E402


class FakeModel:
    def encode(self, queries: list[str]):
        self.queries = queries
        return FakeEmbedding()


class FakeEmbedding:
    def tolist(self) -> list[list[float]]:
        return [[0.1, 0.2, 0.3]]


class FakeCollection:
    def count(self) -> int:
        return 2

    def query(self, query_embeddings, n_results, include):
        self.query_embeddings = query_embeddings
        self.n_results = n_results
        self.include = include
        return {
            "ids": [["chunk_a", "chunk_b"]],
            "distances": [[0.1, 0.25]],
            "metadatas": [[
                {"parent_asin": "ASIN_A", "aspect": "battery life"},
                {"parent_asin": "ASIN_B", "aspect": "sound quality"},
            ]],
        }


class DenseRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = DenseRetriever("unused")
        self.retriever.model = FakeModel()
        self.retriever.collection = FakeCollection()

    def test_search_maps_distance_to_similarity_score(self) -> None:
        results, timing = self.retriever.search("battery life", top_k=2)
        self.assertEqual([row.chunk_id for row in results], ["chunk_a", "chunk_b"])
        self.assertAlmostEqual(results[0].score, 0.9)
        self.assertEqual(results[0].metadata["aspect"], "battery life")
        self.assertGreaterEqual(timing.total_seconds, 0.0)

    def test_corpus_size_uses_collection_count(self) -> None:
        self.assertEqual(self.retriever.corpus_size, 2)


if __name__ == "__main__":
    unittest.main()
