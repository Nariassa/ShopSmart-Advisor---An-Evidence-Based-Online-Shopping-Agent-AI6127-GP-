"""Unit tests for the 'Why not?' module, using small fake data.

No model download or knowledge base is needed: sentiment is faked with
keywords and the similar-product search is faked with a fixed list.
Run from the repo root:  python -m unittest tests.test_why_not -v
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

from src.why_not.why_not import (
    ReviewStore, Sentiment, TitleSimilarity, WhyNotExplainer, aspects_from_query,
    dense_similar_products, severe_issues_in, split_prefix, to_markdown,
)

CAT = "Portable Audio & Accessories"


def chunk(asin, title, i, aspect, text, ctype="review", rating=4.0, cat=CAT):
    return {
        "chunk_id": f"{asin}_{ctype}_{i}", "parent_asin": asin, "chunk_type": ctype,
        "aspect": aspect, "text": f"[Product: {title} | ASIN: {asin}] {text}",
        "rating": rating, "category": cat,
    }


def fake_sentiment(texts):
    out = []
    for t in texts:
        low = t.lower()
        if any(w in low for w in ["won't", "stopped", "terrible", "bad", "died", "awful", "hurt"]):
            out.append(Sentiment("negative", 0.9))
        elif any(w in low for w in ["great", "love", "excellent", "amazing", "lasts"]):
            out.append(Sentiment("positive", 0.9))
        else:
            out.append(Sentiment("neutral", 0.8))
    return out


def build_rows():
    A, TA = "A1", "Buzz Earbuds X"
    rows = [
        chunk(A, TA, 0, "specifications", "Features: bluetooth 5.0", ctype="spec"),
        chunk(A, TA, 1, "battery life", "The battery won't charge after two weeks, terrible."),
        chunk(A, TA, 2, "battery life", "Battery won't charge anymore, bad purchase."),
        chunk(A, TA, 3, "battery life", "Battery is great on day one."),
        chunk(A, TA, 4, "battery life", "Battery stopped charging, won't hold a charge."),
        chunk(A, TA, 5, "sound quality", "Sound is amazing and the bass is great."),
        chunk(A, TA, 6, "sound quality", "Sound is terrible, tinny audio."),
        chunk(A, TA, 7, "sound quality", "I love the sound."),
        chunk(A, TA, 8, "sound quality", "Bad sound, awful volume."),
        chunk(A, TA, 9, "general", "They stopped working after a month."),
        chunk(A, TA, 10, "general", "Mine died and stopped working, awful."),
    ]
    B, TB = "B1", "Calm Earbuds Pro"
    rows += [chunk(B, TB, i, "battery life", f"Battery lasts all day, great {i}.") for i in range(4)]
    rows += [chunk(B, TB, 9, "sound quality", "Sound is fine.")]
    # colour variant of A: same title, must never be suggested
    rows += [chunk("A2", TA, i, "battery life", f"Battery is excellent {i}.") for i in range(4)]
    # similar product but also bad battery: must not be suggested
    rows += [chunk("C1", "Dim Earbuds", i, "battery life", f"Battery won't charge, bad {i}.") for i in range(4)]
    return rows


class WhyNotTests(unittest.TestCase):
    def setUp(self):
        self.store = ReviewStore(pd.DataFrame(build_rows()))
        similar = lambda asin, title, cat, k: ["A2", "C1", "B1", "ZZ_missing"]
        self.explainer = WhyNotExplainer(self.store, fake_sentiment, similar)

    def test_helpers(self):
        self.assertEqual(split_prefix("[Product: T | ASIN: X] hello"), ("T", "hello"))
        self.assertIn("charging / battery failure", severe_issues_in("it won't charge"))
        self.assertEqual(aspects_from_query("comfortable earbuds with long battery"),
                         ["battery life", "comfort"])

    def test_full_explanation(self):
        r = self.explainer.explain("A1", query="comfortable earbuds with long battery life")
        issues = {d["issue"]: d for d in r["deal_breakers"]}
        self.assertIn("charging / battery failure", issues)
        self.assertIn("stopped working", issues)  # found in 'general' chunks

        alts = issues["charging / battery failure"]["alternatives"]
        self.assertEqual([a["asin"] for a in alts], ["B1"])  # not A2 (variant), not C1 (bad)
        self.assertEqual(alts[0]["evidence"]["chunk_id"].split("_")[0], "B1")

        self.assertIn("battery life", [c["aspect"] for c in r["complaints"]])
        self.assertIn("sound quality", [p["aspect"] for p in r["polarized"]])
        self.assertIn("comfort", [m["aspect"] for m in r["missing"]])
        json.dumps(r)  # must be JSON-serialisable for the UI
        self.assertIn("Deal breaker", to_markdown(r))

    def test_no_alternatives_without_deal_breakers(self):
        r = self.explainer.explain("B1")
        self.assertEqual(r["deal_breakers"], [])
        self.assertIn("No deal breakers", r["headline"])

    def test_minor_concerns_not_hidden(self):
        rows = [chunk("M1", "Mostly Fine Speaker", i, "battery life", f"Battery lasts long, great {i}.") for i in range(3)]
        rows += [chunk("M1", "Mostly Fine Speaker", 5, "design/appearance", "The colour looks bad."),
                 chunk("M1", "Mostly Fine Speaker", 6, "general", "Terrible manual.")]
        r = WhyNotExplainer(ReviewStore(pd.DataFrame(rows)), fake_sentiment).explain("M1")
        self.assertEqual(r["deal_breakers"], [])
        self.assertEqual(r["complaints"], [])
        self.assertEqual({m["aspect"] for m in r["minor_concerns"]}, {"design/appearance", "general"})
        self.assertIn("minor concerns", r["headline"])
        self.assertIn("Minor concern (other", to_markdown(r))

    def test_quotes_are_clean_and_on_topic(self):
        from src.why_not.why_not import AspectStats, clean_text
        self.assertEqual(clean_text("Dead after 7 months!<br />Error /> here"), "Dead after 7 months! Error here")
        st = AspectStats("durability", [
            ({"chunk_id": "x1"}, "Good warranty, 7 years.", Sentiment("positive", 0.99)),
            ({"chunk_id": "x2"}, "Two years later it still works perfectly.", Sentiment("positive", 0.80)),
        ])
        self.assertEqual(st.best_quote("positive")["chunk_id"], "x1")
        self.assertEqual(st.best_quote("positive", ["still works"])["chunk_id"], "x2")

    def test_no_reports_alternative(self):
        rows = build_rows()
        rows += [chunk("D1", "Quiet Earbuds", i, "general", f"Works fine, review {i}.", rating=4.6) for i in range(16)]
        rows += [chunk("D2", "Few Review Earbuds", i, "general", f"Works fine {i}.", rating=4.8) for i in range(5)]
        rows += [chunk("D3", "Low Rated Earbuds", i, "general", f"Works fine {i}.", rating=3.2) for i in range(16)]
        store = ReviewStore(pd.DataFrame(rows))
        r = WhyNotExplainer(store, fake_sentiment, lambda *a: ["D1", "D2", "D3", "B1"]).explain("A1")
        alts = {d["issue"]: d for d in r["deal_breakers"]}["stopped working"]["alternatives"]
        self.assertEqual([(a["asin"], a["basis"]) for a in alts], [("D1", "no_reports")])
        self.assertIn("0 of 16 review chunks report", to_markdown(r))

    def test_unknown_product(self):
        self.assertIn("error", self.explainer.explain("NOPE"))

    def test_dense_similar_filters_category(self):
        hits = [SimpleNamespace(metadata={"parent_asin": a, "category": c})
                for a, c in [("A1", CAT), ("X1", "Computers"), ("B1", CAT), ("B1", CAT), ("C1", CAT)]]
        retriever = SimpleNamespace(search=lambda q, top_k: (hits, None))
        find = dense_similar_products(retriever)
        self.assertEqual(find("A1", "Buzz Earbuds X", CAT, 5), ["B1", "C1"])

    def test_title_similarity_fallback(self):
        import numpy as np
        vocab = ["buzz", "calm", "dim", "earbuds", "laptop"]
        def enc(texts):
            v = np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype=float) + 1e-9
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        rows = build_rows() + [chunk("L1", "Laptop Earbuds Bundle", 0, "general", "ok", cat="Computers")]
        store = ReviewStore(pd.DataFrame(rows))
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "t.npz")
            sim = TitleSimilarity(store, encode=enc, cache_path=cache)
            found = sim("A1", "Buzz Earbuds X", CAT, 10)
            self.assertNotIn("A1", found)
            self.assertNotIn("L1", found)          # other category filtered out
            self.assertEqual(set(found), {"A2", "B1", "C1"})
            again = TitleSimilarity(store, encode=enc, cache_path=cache)  # loads from cache
            self.assertEqual(again.asins, sim.asins)
            r = WhyNotExplainer(store, fake_sentiment, sim).explain("A1")
            alts = {d["issue"]: d for d in r["deal_breakers"]}["charging / battery failure"]["alternatives"]
            self.assertEqual([a["asin"] for a in alts], ["B1"])

    def test_parquet_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "chunks.parquet")
            pd.DataFrame(build_rows()).to_parquet(path)
            store = ReviewStore.from_parquet(path)
            self.assertEqual(store.product_info("A1")["title"], "Buzz Earbuds X")
            self.assertEqual(len(store.product_chunks("B1")), 5)


if __name__ == "__main__":
    unittest.main()
