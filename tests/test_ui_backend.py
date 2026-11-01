"""Tests for the UI backend (src/ui/backend.py). No models or knowledge base needed.
Run from the repo root:  python -m unittest tests.test_ui_backend -v
"""

import unittest
from types import SimpleNamespace

import pandas as pd

from src.ui.backend import (
    ShopSmartBackend, compute_pcs, coverage_checklist, extractive_answer, lexicon_sentiment,
)
from src.why_not.why_not import ReviewStore, WhyNotExplainer

Q = "comfortable earbuds with long battery life under $100"


def why_not_result(**over):
    base = {"review_chunks_analysed": 30, "negative_share_overall": 0.1, "deal_breakers": [],
            "polarized": [], "aspect_summary": {"battery life": {"chunks": 5}, "comfort": {"chunks": 1}}}
    base.update(over)
    return base


class CoverageAndPcsTests(unittest.TestCase):
    def test_coverage_statuses_and_price_note(self):
        cov = {c["aspect"]: c for c in coverage_checklist(Q, why_not_result())}
        self.assertEqual(cov["battery life"]["status"], "covered")
        self.assertEqual(cov["comfort"]["status"], "partial")
        self.assertEqual(cov["price/value"]["status"], "missing")
        self.assertIn("not in the knowledge base", cov["price/value"]["note"])

    def test_pcs_range_and_penalties(self):
        good = why_not_result()
        cov = coverage_checklist(Q, good)
        pcs = compute_pcs(good, cov)
        self.assertTrue(0 <= pcs["score"] <= 100)
        self.assertAlmostEqual(sum(pcs["weights"].values()), 1.0, places=2)
        bad = compute_pcs(why_not_result(negative_share_overall=0.6,
                                         deal_breakers=[{"issue": "x"}, {"issue": "y"}]), cov)
        self.assertLess(bad["score"], pcs["score"])
        self.assertIn("deal breaker", bad["explanation"])
        self.assertIn("no evidence on price/value", pcs["explanation"])

    def test_pcs_without_required_aspects_drops_coverage(self):
        pcs = compute_pcs(why_not_result(), [])
        self.assertNotIn("coverage", pcs["components"])

    def test_lexicon_sentiment(self):
        labels = [s.label for s in lexicon_sentiment(["Great sound, love it", "It broke, terrible", "It is blue"])]
        self.assertEqual(labels, ["positive", "negative", "neutral"])


def store():
    def row(asin, title, i, aspect, text, ctype="review"):
        return {"chunk_id": f"{asin}_{ctype}_{i}", "parent_asin": asin, "chunk_type": ctype, "aspect": aspect,
                "text": f"[Product: {title} | ASIN: {asin}] {text}", "rating": 4.2, "category": "Audio"}
    rows = [row("E1", "Buzz Earbuds", 0, "battery life", "Battery lasts long, great."),
            row("E1", "Buzz Earbuds", 1, "general", "Comfortable fit for flights."),
            row("E2", "Calm Earbuds", 0, "sound quality", "Great bass."),
            row("M1", "Desk Monitor", 0, "screen/display", "Sharp screen.")]
    return ReviewStore(pd.DataFrame(rows))


class BackendTests(unittest.TestCase):
    def test_dense_mode_groups_chunks_by_product(self):
        hits = [SimpleNamespace(chunk_id=c, score=s, metadata={"parent_asin": c.split("_")[0]})
                for c, s in [("E1_review_0", .9), ("E2_review_0", .8), ("E1_review_1", .7), ("ZZ_review_0", .6)]]
        dense = SimpleNamespace(search=lambda q, top_k: (hits, None))
        st = store()
        b = ShopSmartBackend(WhyNotExplainer(st, lexicon_sentiment), dense_retriever=dense)
        found, _ = b.search("earbuds", 5)
        self.assertEqual([h.asin for h in found], ["E1", "E2"])          # unknown product skipped
        self.assertEqual([e.chunk_id for e in found[0].evidence], ["E1_review_0", "E1_review_1"])
        self.assertEqual(found[0].title, "Buzz Earbuds")

        ans = b.answer("earbuds", found)
        self.assertEqual(ans["mode"], "extractive")
        self.assertIn("[C1]", ans["text"])
        self.assertEqual(ans["citations"]["C1"]["chunk_id"], "E1_review_0")

        analysis = b.analyse(found[0], "comfortable earbuds with long battery life")
        self.assertIn("score", analysis["pcs"])
        self.assertEqual({c["aspect"] for c in analysis["coverage"]}, {"battery life", "comfort"})
        self.assertIn("Why not", b.why_not_markdown("E1", "earbuds"))

    def test_llm_failure_falls_back_to_quotes(self):
        b = ShopSmartBackend(WhyNotExplainer(store(), lexicon_sentiment),
                             title_search=lambda a, t, c, k: ["E1"], openai_key="invalid")
        b._llm_answer = lambda q, ctx: (_ for _ in ()).throw(ConnectionError("offline"))
        hits, _ = b.search("battery earbuds", 3)
        ans = b.answer("battery earbuds", hits)
        self.assertEqual(ans["mode"], "fallback")
        self.assertIn("LLM unavailable", ans["text"])

    def test_saved_mode_replays_real_results(self):
        b = ShopSmartBackend.from_saved_results()
        questions = b.saved_questions()
        self.assertEqual(len(questions), 20)
        q = questions[0]
        self.assertFalse(b.is_approximate(q))
        hits, _ = b.search(q, 5)
        self.assertEqual(len(hits), 3)                       # top 3 = products with a saved "Why not?"
        ans = b.answer(q, hits)
        self.assertEqual(ans["mode"], "saved")
        self.assertEqual(len(ans["citations"]), 5)           # [C1]..[C5] = Dense top 5
        saved = b.explain(hits[0].asin, q)
        self.assertEqual(saved["query_id"], "Q001")          # the saved RoBERTa result, not recomputed
        self.assertIn("Why not", b.why_not_markdown(hits[0].asin, q))

    def test_saved_mode_new_question_is_approximate(self):
        b = ShopSmartBackend.from_saved_results()
        q = "usb charging cable that lasts"
        self.assertTrue(b.is_approximate(q))
        hits, _ = b.search(q, 3)
        self.assertTrue(hits and all(h.evidence for h in hits))
        self.assertEqual(b.answer(q, hits)["mode"], "extractive")

    def test_extractive_answer_one_line_per_product(self):
        h = SimpleNamespace(asin="A", title="T")
        e = SimpleNamespace(quote="q")
        self.assertEqual(extractive_answer([(h, e), (h, e)]).count("**T**"), 1)


if __name__ == "__main__":
    unittest.main()
