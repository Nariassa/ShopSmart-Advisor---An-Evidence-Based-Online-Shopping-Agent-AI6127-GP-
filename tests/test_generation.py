"""Focused checks for faithfulness, PCS behavior and knowledge-base, retrieval, and aspect-analysis handoff contracts."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.generation import PCSConfig, ShopSmartGenerator, calculate_pcs
from src.generation.adapters import ParquetEvidenceStore
from src.generation.batch import citation_audit, main as batch_main
from src.generation.evidence import normalize_evidence, sentence_spans
from src.generation.grounding import extract_facts, verify_claims
from src.generation.llm import LLMUnavailable, OllamaBackend
from src.generation.sentiment import LexiconSentiment, evidence_sentiment


NOW = datetime(2026, 10, 12, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def review(index, label="positive", aspect="battery life", asin="A", **extra):
    return {"chunk_id": f"{asin}_{index}", "parent_asin": asin, "chunk_type": "review",
            "aspect": aspect, "text": f"Battery evidence {index} is {label}.",
            "sentiment_label": label, "source_review_id": f"{asin}-review-{index}",
            "verified_purchase": True, "helpful_vote": 20, "timestamp": NOW.isoformat(), **extra}


def score(rows, aspects=None, **kwargs):
    items = normalize_evidence(rows)
    classifier = LexiconSentiment()
    sentiments = {item.chunk_id: evidence_sentiment(item, classifier) for item in items}
    return calculate_pcs(items, sentiments, aspects or ["battery life"], now=NOW, **kwargs)


class PCSBehaviorTests(unittest.TestCase):
    def test_no_reviews_or_specs_only_cannot_receive_confidence(self):
        self.assertEqual(score([])["score"], 0)
        self.assertEqual(score([{**review(1), "chunk_type": "spec"}])["score"], 0)

    def test_more_distinct_review_evidence_increases_confidence(self):
        self.assertGreater(score([review(i) for i in range(20)])["score"], score([review(0)])["score"])

    def test_polarization_reduces_confidence_with_equal_sample_size(self):
        consistent = score([review(i) for i in range(10)])
        polarized = score([review(i, "positive" if i < 5 else "negative") for i in range(10)])
        self.assertEqual(polarized["components"]["polarization"], 1)
        self.assertLess(polarized["score"], consistent["score"])

    def test_all_negative_is_consistent_but_has_zero_fit(self):
        result = score([review(i, "negative") for i in range(10)])
        self.assertEqual(result["components"]["consistency"], 1)
        self.assertEqual(result["fit_score"], 0)

    def test_missing_requested_aspect_reduces_score(self):
        rows = [review(i) for i in range(10)]
        complete = score(rows)
        incomplete = score(rows, ["battery life", "comfort"])
        self.assertAlmostEqual(incomplete["score"], complete["score"] / 2, places=1)

    def test_missing_timestamps_are_disclosed_and_not_assumed_recent(self):
        result = score([review(i, timestamp=None) for i in range(4)])
        self.assertEqual(result["components"]["recency"], 0.5)
        self.assertEqual(result["missing_metadata"]["timestamp"], 4)

    def test_old_reviews_decay_by_half_life_and_future_dates_are_unknown(self):
        old = score([review(0, timestamp=(NOW-timedelta(days=180)).timestamp()*1000)])
        future = score([review(0, timestamp=(NOW+timedelta(days=1)).isoformat())])
        self.assertAlmostEqual(old["components"]["recency"], 0.5)
        self.assertEqual(future["missing_metadata"]["timestamp"], 1)

    def test_repeated_source_review_is_not_counted_as_multiple_users(self):
        result = score([review(i, source_review_id="same-review") for i in range(5)])
        self.assertEqual(result["review_evidence_units"], 1)

    def test_repeated_text_without_source_id_is_deduplicated(self):
        result = score([review(i, source_review_id=None, text="The battery is good.") for i in range(5)])
        self.assertEqual(result["review_evidence_units"], 1)
        self.assertEqual(result["missing_metadata"]["source_review_id"], 1)

    def test_verified_and_helpful_evidence_increases_quality(self):
        high = score([review(1)])
        low = score([review(1, verified_purchase=False, helpful_vote=0)])
        self.assertGreater(high["components"]["quality"], low["components"]["quality"])

    def test_member_d_signals_override_local_distribution(self):
        result = score([review(i) for i in range(10)], signals={
            "aspects": {"battery life": {"positive": 5, "negative": 5, "neutral": 0}}})
        self.assertEqual(result["components"]["polarization"], 1)
        self.assertEqual(result["signal_source"], "member_d")

    def test_invalid_signals_and_pcs_configuration_are_rejected(self):
        for invalid in (-0.1, 1.1, float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                score([review(1)], signals={"polarization": invalid})
        with self.assertRaises(ValueError):
            PCSConfig(quality_weight=0.8)
        with self.assertRaises(ValueError):
            score([review(1)], aspect_weights={"battery life": -1})

    def test_cross_product_scoring_is_rejected(self):
        with self.assertRaises(ValueError):
            score([review(1), review(2, asin="B")])


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.classifier = LexiconSentiment()
        self.rows = [{"chunk_id": "A_1", "parent_asin": "A", "chunk_type": "review",
                      "aspect": "battery life",
                      "text": "[Product: Test | ASIN: A] The battery is not good. Charging is reliable."}]
        self.items = normalize_evidence(self.rows)
        sentiments = {item.chunk_id: evidence_sentiment(item, self.classifier) for item in self.items}
        self.facts = extract_facts(self.items, "battery", ["battery life"], self.classifier, sentiments)

    def test_exact_sentence_offsets_match_original_source(self):
        for quote, start, end in sentence_spans(self.items[0]):
            self.assertEqual(self.rows[0]["text"][start:end], quote)
        claims, _ = verify_claims({"claims": [{"fact_id": f.fact_id, "text": f.quote} for f in self.facts]},
                                  self.facts, self.items)
        self.assertEqual(len(claims), 2)
        for claim in claims:
            self.assertEqual(claim["parent_asin"], "A")

    def test_partial_quote_cannot_hide_negation(self):
        negative = next(f for f in self.facts if "not good" in f.quote)
        accepted, rejected = verify_claims({"claims": [{"fact_id": negative.fact_id, "text": "good"}]},
                                            self.facts, self.items)
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "paraphrase_requires_nli")

    def test_fabricated_fact_ids_and_superlatives_are_removed(self):
        payload = {"claims": [{"fact_id": "F999", "text": "Best product ever."},
                               {"fact_id": self.facts[0].fact_id, "text": "Battery lasts 100 hours."}]}
        accepted, rejected = verify_claims(payload, self.facts, self.items)
        self.assertFalse(accepted)
        self.assertEqual(len(rejected), 2)
        accepted, rejected = verify_claims({"claims": [{"fact_id": [], "text": "Invalid ID type."}]},
                                            self.facts, self.items)
        self.assertFalse(accepted)
        self.assertEqual(len(rejected), 1)

    def test_optional_nli_rejects_nonentailed_paraphrases(self):
        class RejectingVerifier:
            def supports(self, premise, claim):
                return False, 0.1
        accepted, rejected = verify_claims({"claims": [{"fact_id": self.facts[0].fact_id,
                                                         "text": "This is the best."}]},
                                            self.facts, self.items, RejectingVerifier())
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "nli_not_entailed")

    def test_header_asin_mismatch_and_conflicting_ids_fail(self):
        with self.assertRaises(ValueError):
            normalize_evidence([{**self.rows[0], "parent_asin": "B"}])
        with self.assertRaises(ValueError):
            normalize_evidence([self.rows[0], {**self.rows[0], "text": "Different content."}])

    def test_long_sentence_is_skipped_instead_of_truncating_evidence(self):
        item = normalize_evidence([{**self.rows[0], "text": "Not " + "great "*200 + "at all."}])[0]
        self.assertEqual(sentence_spans(item), [])

    def test_product_rating_never_substitutes_for_aspect_sentiment(self):
        item = normalize_evidence([{**self.rows[0], "rating": 5, "text": "Battery life is terrible."}])[0]
        self.assertEqual(evidence_sentiment(item, self.classifier).label, "negative")
        self.assertEqual(self.classifier.classify("The sound is not very good.").label, "negative")
        self.assertEqual(self.classifier.classify("Not bad, but the battery is poor.").label, "neutral")
        self.assertEqual(self.classifier.classify("This is not very developed to be helpful.").label, "negative")
        self.assertEqual(self.classifier.classify("They do not connect to your phone.").label, "negative")
        self.assertEqual(self.classifier.classify("It could no longer connect to the app.").label, "negative")


class GeneratorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.rows = json.loads((ROOT/"examples/generation_evidence.json").read_text(encoding="utf-8"))["evidence"]

    def test_offline_generation_has_balanced_traceable_claims(self):
        result = ShopSmartGenerator().generate_recommendation("battery life and comfort", self.rows, now=NOW)
        self.assertEqual(result["backend"], "extractive")
        self.assertTrue(any(c["kind"] == "con" for c in result["claims"]))
        for claim in result["claims"]:
            for citation in claim["citations"]:
                source = result["sources"][citation["source_id"]]
                self.assertEqual(source["text"][citation["start"]:citation["end"]], citation["quote"])
                self.assertEqual(source["parent_asin"], claim["parent_asin"])
        json.dumps(result)

    def test_llm_failure_falls_back_and_strict_mode_raises(self):
        class Unavailable:
            def generate(self, prompt):
                raise LLMUnavailable("模拟服务不可用")
        result = ShopSmartGenerator(llm=Unavailable()).generate_recommendation("battery", self.rows, now=NOW)
        self.assertEqual(result["backend"], "extractive_fallback")
        self.assertEqual(result["warnings"][0]["code"], "llm_fallback")
        with self.assertRaises(LLMUnavailable):
            ShopSmartGenerator(llm=Unavailable(), allow_fallback=False).generate_recommendation("battery", self.rows)

    def test_hallucinated_llm_response_is_removed_and_replaced(self):
        class Hallucinating:
            def generate(self, prompt):
                return {"selections": [{"fact_id": "F999", "product_name": "Invented product"}]}
        result = ShopSmartGenerator(llm=Hallucinating()).generate_recommendation("battery", self.rows, now=NOW)
        self.assertEqual(result["backend"], "extractive_fallback")
        self.assertNotIn("last forever", result["answer"])
        self.assertEqual(len(result["grounding"]["rejected"]), 1)

    def test_llm_selection_is_used_and_opposing_evidence_is_preserved(self):
        class Selective:
            def generate(self, prompt):
                facts = json.loads(prompt.split("EVIDENCE_FACTS_JSON: ")[1])
                fact = next(f for f in facts if f["sentiment"] == "positive")
                return {"selections": [{"fact_id": fact["fact_id"], "product_name": fact["product_name"]}]}
        result = ShopSmartGenerator(llm=Selective()).generate_recommendation("battery and comfort", self.rows, now=NOW)
        self.assertEqual(result["backend"], "llm")
        self.assertTrue(any(c["kind"] == "con" for c in result["claims"]))

    def test_unverifiable_budget_prevents_unqualified_recommendation(self):
        rows = [review(i, text=f"Battery {i} is excellent.") for i in range(20)]
        result = ShopSmartGenerator().generate_recommendation("battery under $200", rows, now=NOW)
        self.assertEqual(result["products"][0]["budget_check"]["status"], "unknown")
        self.assertEqual(result["products"][0]["verdict"], "needs_budget_check")

    def test_negative_consistent_evidence_does_not_become_positive_advice(self):
        rows = [review(i, "negative", text=f"Battery {i} is terrible.") for i in range(20)]
        result = ShopSmartGenerator().generate_recommendation("battery", rows, now=NOW)
        self.assertEqual(result["products"][0]["verdict"], "not_preferred")

    def test_unvalidated_baseline_does_not_issue_a_purchase_recommendation(self):
        rows = [review(i, text=f"Battery {i} is excellent.") for i in range(20)]
        for row in rows:
            del row["sentiment_label"]
        result = ShopSmartGenerator().generate_recommendation("battery", rows, now=NOW)
        self.assertEqual(result["products"][0]["verdict"], "needs_sentiment_validation")

    def test_empty_or_irrelevant_evidence_has_explicit_insufficient_status(self):
        self.assertEqual(ShopSmartGenerator().generate_recommendation("battery", [])["status"], "insufficient_evidence")
        irrelevant = [{**review(0), "aspect": "sound quality", "text": "Sound is excellent."}]
        result = ShopSmartGenerator().generate_recommendation("battery", irrelevant)
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_ollama_transport_uses_json_schema_and_parses_claims(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size):
                return json.dumps({"response": json.dumps({"claims": []}), "done": True}).encode()
        with patch("src.generation.llm.urlopen", return_value=Response()) as transport:
            result = OllamaBackend().generate("test")
            body = json.loads(transport.call_args.args[0].data)
            self.assertFalse(body["stream"])
            self.assertEqual(body["format"]["type"], "object")
            self.assertEqual(result, {"claims": []})

    def test_batch_exports_aligned_contexts_without_fabricated_references(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"output.jsonl"
            ragas = Path(directory)/"ragas.jsonl"
            batch_main(["--queries", str(ROOT/"examples/generation_queries.jsonl"),
                        "--evidence-map", str(ROOT/"examples/generation_evidence_map.json"),
                        "--output", str(output), "--export-ragas", str(ragas)])
            results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            items = [json.loads(line) for line in ragas.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["query_id"], items[0]["query_id"])
            self.assertEqual(citation_audit(results[0])["citation_traceability"], 1)
            self.assertNotIn("reference", items[0])
            self.assertEqual(len(items[0]["retrieved_contexts"]), len(self.rows))

    def test_missing_batch_evidence_map_fails_before_output_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping = Path(directory)/"map.json"
            mapping.write_text('{}', encoding="utf-8")
            output = Path(directory)/"output.jsonl"
            with self.assertRaises(ValueError):
                batch_main(["--queries", str(ROOT/"examples/generation_queries.jsonl"),
                            "--evidence-map", str(mapping), "--output", str(output)])
            self.assertFalse(output.exists())


class ParquetHandoffTests(unittest.TestCase):
    def test_bm25_ids_are_hydrated_in_rank_order_with_metadata(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"chunks.parquet"
            pq.write_table(pa.Table.from_pylist([review(1), review(2), review(3)]), path)
            store = ParquetEvidenceStore(path, batch_size=1)
            rows = store.hydrate([{"chunk_id": "A_3", "rank": 1, "score": 0.7},
                                  {"chunk_id": "A_1", "rank": 2, "score": 0.5}])
            self.assertEqual([row["chunk_id"] for row in rows], ["A_3", "A_1"])
            self.assertEqual(rows[0]["rank"], 1)
            self.assertEqual(rows[0]["score"], 0.7)
            self.assertEqual(len(store.product_chunks("A", limit=2)), 2)
            with self.assertRaises(ValueError):
                store.lookup(["not-found"])


if __name__ == "__main__":
    unittest.main()
