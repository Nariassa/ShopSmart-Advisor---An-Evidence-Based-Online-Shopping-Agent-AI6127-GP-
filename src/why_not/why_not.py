"""
why_not.py - "Why not?" explanations for one Amazon Electronics product.

Given a product (parent_asin) and, optionally, the shopper's query, this module
answers "why might I NOT want to buy this?" using only evidence from the
knowledge base, and cites the review chunks it relies on.

Output sections (one JSON-serialisable dict):
    deal_breakers   severe problems (stopped working, won't charge, ...) that
                    several reviews report, each with cited quotes
    complaints      aspects with a high share of negative reviews
    polarized       aspects where reviewers strongly disagree
    missing         aspects the shopper asked about that no review covers
    alternatives    for each deal breaker: similar products (same category)
                    whose reviews are POSITIVE on that same aspect

Pipeline:
    1. ReviewStore      -> all chunks of the product (from chunks.parquet)
    2. SentimentScorer  -> positive / neutral / negative per review chunk
                           (subjectivity first: neutral = not an opinion,
                           then polarity: positive vs negative)
    3. aspect statistics -> complaints, polarized aspects, missing evidence
    4. severe-issue rules on negative chunks -> deal breakers
    5. dense search for similar products -> score them on the deal-breaker
       aspect -> alternatives with a positive quote as proof

Quick use (Colab or local):
    from src.why_not.why_not import WhyNotExplainer
    explainer = WhyNotExplainer.from_paths("chunks.parquet", "chroma_db")
    result = explainer.explain("B0764GNRDZ", query="earbuds for long flights")
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration (thresholds are deliberately simple so they can be explained
# in the report and tuned in the ablation study)
# ---------------------------------------------------------------------------

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

MIN_ASPECT_CHUNKS = 3          # need at least this many chunks to judge an aspect
COMPLAINT_NEG_SHARE = 0.30     # aspect is a "complaint" if >= 30% negative
POLARIZED_MIN_SHARE = 0.30     # polarized if >= 30% positive AND >= 30% negative
POLARIZED_MIN_CHUNKS = 4
MISSING_MAX_CHUNKS = 1         # "missing" if the product has <= 1 chunk on it
MINOR_MIN_CONFIDENCE = 0.60    # isolated negative mentions shown as "minor concerns"
MINOR_SHOW = 3
DEAL_BREAKER_MIN_REPORTS = 2   # a severe issue must be reported >= 2 times
ALT_MIN_CHUNKS = 3             # an alternative needs >= 3 chunks on the aspect
ALT_MIN_POS_SHARE = 0.50       # ...and at least 50% positive on it
ALT_MIN_REVIEWS = 15           # "no_reports" alternative: enough reviews to trust a zero
ALT_MIN_RATING = 4.0           # ...and a good average rating
ALT_CANDIDATES = 15            # similar products to examine per request
ALT_SHOW = 3                   # alternatives shown per deal breaker
QUOTE_CHARS = 220

# Aspects stored in the KB (assigned by keyword matching in Step1 notebook).
KB_ASPECTS = [
    "price/value", "build quality", "specifications/performance",
    "battery life", "screen/display", "connectivity",
    "customer service", "ease of use", "sound quality", "design/appearance",
]

# Extra aspects the KB taxonomy does not have. Shoppers ask about them
# (e.g. test query Q001 asks about comfort), so we find them by keyword
# over ALL review chunks of a product.
EXTRA_ASPECT_KEYWORDS = {
    "comfort": ["comfort", "comfortable", "uncomfortable", "fit ", "fits ",
                "ear tips", "hurt", "hurts", "pressure", "lightweight", "heavy"],
    "durability": ["durable", "durability", "lasted", "last long", "still works",
                   "stopped working", "broke", "broken", "died", "fell apart",
                   "months later", "years later", "wear and tear", "still going",
                   "still working", "reliable", "holding up", "held up"],
}

# Phrases that show a review PRAISES an aspect - used to pick the most
# convincing quote for an alternative ("lasts all day" beats "good warranty").
POSITIVE_ASPECT_PHRASES = {
    "durability": ["still works", "still working", "still going", "reliable", "lasted",
                   "holding up", "held up", "no problems", "no issues", "durable", "years later"],
    "battery life": ["lasts", "long battery", "all day", "hours", "holds a charge", "great battery"],
    "build quality": ["sturdy", "solid", "well made", "well built", "durable", "quality"],
    "connectivity": ["stable", "never drops", "connects", "pairs", "easy to pair", "no dropouts"],
    "customer service": ["helpful", "replaced", "quick response", "great support", "refund"],
}

# Words in the shopper's query -> aspects they care about.
QUERY_ASPECT_KEYWORDS = {
    "price/value": ["price", "cheap", "budget", "value", "affordable", "cost", "worth", "under $"],
    "build quality": ["build", "quality", "sturdy", "solid", "well made"],
    "specifications/performance": ["performance", "fast", "speed", "lag", "processor", "storage", "memory"],
    "battery life": ["battery", "charge", "charging", "long flight", "all day", "hours"],
    "screen/display": ["screen", "display", "resolution", "bright"],
    "connectivity": ["bluetooth", "wifi", "wi-fi", "connection", "connect", "pairing", "usb", "signal"],
    "customer service": ["warranty", "support", "customer service", "return"],
    "ease of use": ["easy", "simple", "setup", "beginner", "intuitive"],
    "sound quality": ["sound", "audio", "bass", "mic", "microphone", "calls", "noise", "volume"],
    "design/appearance": ["design", "look", "color", "style", "compact", "slim"],
    "comfort": ["comfort", "comfortable", "fit", "wear", "long flights", "ears"],
    "durability": ["durable", "durability", "last", "rugged", "reliable"],
}

# Severe failure patterns. A NEGATIVE chunk matching one of these is a
# "severe report". `aspect` = where to look for a better alternative.
SEVERE_ISSUES = {
    "stopped working": {
        "aspect": "durability",
        "patterns": [r"stopped working", r"quit working", r"stop(?:ped)? functioning",
                     r"\bdied\b", r"\bdead\b", r"no longer works?", r"won'?t work",
                     r"does not work", r"doesn'?t work", r"not working"],
    },
    "broke / fell apart": {
        "aspect": "durability",
        "patterns": [r"\bbroke\b", r"\bbroken\b", r"\bcracked\b", r"fell apart",
                     r"snapped", r"came apart", r"falling apart"],
    },
    "defective on arrival": {
        "aspect": "build quality",
        "patterns": [r"defective", r"\bdoa\b", r"dead on arrival", r"arrived damaged",
                     r"faulty"],
    },
    "charging / battery failure": {
        "aspect": "battery life",
        "patterns": [r"won'?t charge", r"will not charge", r"stopped charging",
                     r"not charging", r"doesn'?t charge", r"does not charge",
                     r"won'?t hold (?:a )?charge", r"not hold(?:ing)? (?:a )?charge",
                     r"battery (?:died|dies|drains?)"],
    },
    "overheating / safety": {
        "aspect": "build quality",
        "patterns": [r"overheat", r"too hot", r"burn(?:ed|t|ing)?\b", r"smok(?:e|ing)",
                     r"melt(?:ed|ing)?", r"\bfire\b", r"swell(?:ed|ing)?"],
    },
    "connection drops": {
        "aspect": "connectivity",
        "patterns": [r"disconnect", r"drops? (?:the )?connection", r"keeps? dropping",
                     r"cuts? out", r"won'?t pair", r"will not pair", r"lost connection"],
    },
    "returned / refund": {
        "aspect": "customer service",
        "patterns": [r"returned (?:it|them|this)", r"return(?:ing)? (?:it|them|this)",
                     r"\brefund", r"sent (?:it|them) back", r"waste of money"],
    },
}
_SEVERE_RE = {name: re.compile("|".join(cfg["patterns"]), re.IGNORECASE)
              for name, cfg in SEVERE_ISSUES.items()}

_PREFIX_RE = re.compile(r"^\[Product:\s*(?P<title>.*)\s*\|\s*ASIN:\s*(?P<asin>[^\]]+)\]\s*")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def split_prefix(text: str) -> tuple[str, str]:
    """'[Product: Title | ASIN: X] body' -> ('Title', 'body')."""
    text = str(text or "")
    match = _PREFIX_RE.match(text)
    if not match:
        return "", text.strip()
    return match.group("title").strip(), text[match.end():].strip()


_HTML_RE = re.compile(r"<br\s*/?>?|/>|<[a-z]+[^>]{0,30}>", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Remove leftover HTML (e.g. '<br />') from Amazon review text."""
    return " ".join(_HTML_RE.sub(" ", str(text)).split())


def shorten(text: str, limit: int = QUOTE_CHARS) -> str:
    text = clean_text(text)
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    low = f" {text.lower()} "
    return any(kw in low for kw in keywords)


def aspects_from_query(query: str | None) -> list[str]:
    """Aspects a shopper cares about, detected from the query wording."""
    if not query:
        return []
    return [aspect for aspect, kws in QUERY_ASPECT_KEYWORDS.items()
            if contains_any(query, kws)]


def severe_issues_in(text: str) -> list[str]:
    return [name for name, rx in _SEVERE_RE.items() if rx.search(text)]


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

class ReviewStore:
    """Fast per-product access to chunks.parquet.

    Loads the parquet once (pyarrow-backed strings keep memory low in Colab),
    sorts it by parent_asin and stores each product's (start, end) row range,
    so fetching one product is a cheap contiguous slice instead of a scan over
    4.45M rows. Recently used products are cached.
    """

    COLUMNS = ["chunk_id", "parent_asin", "chunk_type", "aspect", "text", "rating", "category"]
    CACHE_SIZE = 5000

    def __init__(self, chunks: pd.DataFrame):
        self.df = chunks.sort_values("parent_asin", kind="stable").reset_index(drop=True)
        asins = self.df["parent_asin"].astype(str).to_numpy()
        import numpy as np
        starts = np.flatnonzero(np.r_[True, asins[1:] != asins[:-1]])
        ends = np.r_[starts[1:], len(asins)]
        self._rows = {asins[s]: (int(s), int(e)) for s, e in zip(starts, ends)}
        self._cache: dict[str, list[dict]] = {}

    @classmethod
    def from_parquet(cls, path: str) -> "ReviewStore":
        try:
            df = pd.read_parquet(path, columns=cls.COLUMNS, dtype_backend="pyarrow")
        except TypeError:  # pandas < 2.0
            df = pd.read_parquet(path, columns=cls.COLUMNS)
        return cls(df)

    def has(self, asin: str) -> bool:
        return asin in self._rows

    def product_chunks(self, asin: str) -> list[dict]:
        if asin in self._cache:
            return self._cache[asin]
        span = self._rows.get(asin)
        if span is None:
            return []
        records = self.df.iloc[span[0]:span[1]].to_dict("records")
        if len(self._cache) >= self.CACHE_SIZE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[asin] = records
        return records

    def product_info(self, asin: str) -> dict:
        chunks = self.product_chunks(asin)
        if not chunks:
            return {"asin": asin, "title": "", "category": None, "avg_rating": None}
        title = split_prefix(chunks[0]["text"])[0]
        rating = chunks[0].get("rating")
        return {
            "asin": asin,
            "title": title,
            "category": chunks[0].get("category"),
            "avg_rating": None if rating is None or pd.isna(rating) else float(rating),
        }


# ---------------------------------------------------------------------------
# Sentiment (subjectivity + polarity)
# ---------------------------------------------------------------------------

@dataclass
class Sentiment:
    label: str          # "positive" | "neutral" | "negative"
    score: float        # model confidence for that label


class SentimentScorer:
    """Wraps a Hugging Face 3-class sentiment model, with a cache.

    3 classes give both subtasks the assignment asks for:
    subjectivity (neutral vs opinionated) and polarity (positive vs negative).
    Any callable list[str] -> list[Sentiment] can be used instead (tests use
    a fake one), which also makes it easy to swap in SenticNet or a
    fine-tuned model for the ablation study.
    """

    def __init__(self, model_name: str = SENTIMENT_MODEL, batch_size: int = 32,
                 device: int | None = None):
        from transformers import pipeline
        from transformers.utils import logging as hf_logging
        import torch

        hf_logging.set_verbosity_error()  # hides the harmless "pipelines sequentially on GPU" note

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name,
                             device=device, truncation=True, max_length=512)
        self.batch_size = batch_size
        self.cache: dict[str, Sentiment] = {}

    def __call__(self, texts: list[str]) -> list[Sentiment]:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            for text, out in zip(todo, self.pipe(todo, batch_size=self.batch_size)):
                self.cache[text] = Sentiment(out["label"].lower(), float(out["score"]))
        return [self.cache[t] for t in texts]


# ---------------------------------------------------------------------------
# Similar products (for alternatives)
# ---------------------------------------------------------------------------

def dense_similar_products(retriever, top_k: int = 200) -> Callable:
    """Build a 'find similar products' function from the team's DenseRetriever.

    Searches with the product title, keeps other products in the SAME
    category, drops near-duplicates (same title = colour variant).
    """

    def find(asin: str, title: str, category, limit: int) -> list[str]:
        if not title:
            return []
        results, _ = retriever.search(title, top_k=top_k)
        seen, out = {asin}, []
        for r in results:
            meta = r.metadata if hasattr(r, "metadata") else r
            cand = meta.get("parent_asin")
            if not cand or cand in seen:
                continue
            seen.add(cand)
            if category and str(meta.get("category")) != str(category):
                continue
            out.append(cand)
            if len(out) >= limit:
                break
        return out  # colour variants (same title) are dropped later by the explainer

    return find


class TitleSimilarity:
    """Similar products WITHOUT Chroma: embed one title per product and search.

    Fallback when chroma_db is not available. Uses the same BGE model as the
    KB, but only one vector per product instead of 4.45M chunk vectors, so it
    builds in minutes on a GPU. The vectors can be cached to a .npz file.
    """

    def __init__(self, store: ReviewStore, encode: Callable | None = None,
                 model_name: str = "BAAI/bge-base-en-v1.5", cache_path: str | None = None,
                 batch_size: int = 256):
        import numpy as np

        self.np = np
        if encode is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            encode = lambda texts: model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                                show_progress_bar=len(texts) > 1000)
        self.encode = encode

        if cache_path and _exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            self.asins, self.titles = list(data["asins"]), list(data["titles"])
            self.categories, self.vectors = list(data["categories"]), data["vectors"]
        else:
            first = store.df.drop_duplicates("parent_asin")
            self.asins = [str(a) for a in first["parent_asin"]]
            self.titles = [split_prefix(t)[0] for t in first["text"]]
            self.categories = [str(c) for c in first["category"]]
            self.vectors = np.asarray(self.encode(self.titles), dtype=np.float16)
            if cache_path:
                np.savez(cache_path, asins=np.array(self.asins, dtype=object),
                         titles=np.array(self.titles, dtype=object),
                         categories=np.array(self.categories, dtype=object), vectors=self.vectors)
        self.categories_arr = np.array(self.categories, dtype=object)
        self.matrix = np.asarray(self.vectors, dtype=np.float32)  # convert once, not per query

    def __call__(self, asin: str, title: str, category, limit: int) -> list[str]:
        np = self.np
        if not title:
            return []
        q = np.asarray(self.encode([title]), dtype=np.float32)[0]
        scores = self.matrix @ q
        if category is not None:
            scores[self.categories_arr != str(category)] = -np.inf
        out = []
        for i in np.argsort(-scores):
            if not np.isfinite(scores[i]) or len(out) >= limit:
                break
            if self.asins[i] != asin:
                out.append(self.asins[i])
        return out


def _exists(path: str) -> bool:
    import os
    return os.path.exists(path)


# ---------------------------------------------------------------------------
# Main explainer
# ---------------------------------------------------------------------------

@dataclass
class AspectStats:
    aspect: str
    chunks: list = field(default_factory=list)   # (chunk dict, body text, Sentiment)

    @property
    def n(self) -> int:
        return len(self.chunks)

    def count(self, label: str) -> int:
        return sum(1 for _, _, s in self.chunks if s.label == label)

    def share(self, label: str) -> float:
        return self.count(label) / self.n if self.n else 0.0

    def entropy(self) -> float:
        """Normalised entropy of pos/neu/neg (0 = all agree, 1 = maximal split)."""
        probs = [self.share(l) for l in ("positive", "neutral", "negative")]
        h = -sum(p * math.log(p) for p in probs if p > 0)
        return h / math.log(3)

    def best_quote(self, label: str, prefer: Iterable[str] = ()) -> dict | None:
        """Most confident chunk with this label; chunks containing a `prefer`
        phrase win first (so the quote actually talks about the aspect)."""
        prefer = list(prefer)
        cands = [(sum(p in body.lower() for p in prefer), s.score, c, body)
                 for c, body, s in self.chunks if s.label == label]
        if not cands:
            return None
        _, score, chunk, body = max(cands, key=lambda x: (x[0], x[1]))
        return {"chunk_id": chunk["chunk_id"], "quote": shorten(body), "confidence": round(score, 3)}


class WhyNotExplainer:
    def __init__(self, store: ReviewStore, sentiment: Callable[[list[str]], list[Sentiment]],
                 similar_products: Callable | None = None):
        self.store = store
        self.sentiment = sentiment
        self.similar_products = similar_products

    @classmethod
    def from_paths(cls, chunks_path: str = "chunks.parquet", chroma_path: str | None = "chroma_db",
                   model_cache: str | None = None,
                   title_cache: str | None = "title_vectors.npz") -> "WhyNotExplainer":
        """Load everything. Uses Chroma for alternatives if chroma_path exists,
        otherwise falls back to TitleSimilarity (cached in title_cache)."""
        store = ReviewStore.from_parquet(chunks_path)
        if chroma_path and _exists(chroma_path):
            from src.retrieval.dense_retriever import DenseRetriever
            retriever = DenseRetriever(chroma_path=chroma_path, cache_folder=model_cache).load()
            similar = dense_similar_products(retriever)
            print("Alternatives: dense search over Chroma")
        else:
            similar = TitleSimilarity(store, cache_path=title_cache)
            print(f"Alternatives: title similarity over {len(similar.asins):,} products")
        return cls(store, SentimentScorer(), similar)

    # -- aspect statistics ---------------------------------------------------

    def aspect_stats(self, asin: str, only: Iterable[str] | None = None) -> dict[str, AspectStats]:
        """Sentiment per aspect for one product (KB aspects + extra aspects)."""
        wanted = set(only) if only else None
        reviews = [c for c in self.store.product_chunks(asin) if c["chunk_type"] == "review"]
        groups: dict[str, list] = defaultdict(list)
        for chunk in reviews:
            body = split_prefix(chunk["text"])[1]
            if chunk["aspect"] in KB_ASPECTS:
                groups[chunk["aspect"]].append((chunk, body))
            for extra, kws in EXTRA_ASPECT_KEYWORDS.items():
                if contains_any(body, kws):
                    groups[extra].append((chunk, body))
        if wanted is not None:
            groups = {a: g for a, g in groups.items() if a in wanted}

        bodies = [body for g in groups.values() for _, body in g]
        sentiments = dict(zip(bodies, self.sentiment(bodies))) if bodies else {}
        return {a: AspectStats(a, [(c, b, sentiments[b]) for c, b in g])
                for a, g in groups.items()}

    # -- sections ------------------------------------------------------------

    def _deal_breakers(self, stats: dict[str, AspectStats], all_negative: list) -> list[dict]:
        reports: dict[str, dict] = {}
        for chunk, body, sent in all_negative:
            for issue in severe_issues_in(body):
                rep = reports.setdefault(issue, {"chunks": {}})
                rep["chunks"][chunk["chunk_id"]] = (chunk, body, sent)
        out = []
        for issue, rep in reports.items():
            hits = list(rep["chunks"].values())
            if len(hits) < DEAL_BREAKER_MIN_REPORTS:
                continue
            hits.sort(key=lambda x: x[2].score, reverse=True)
            aspect = SEVERE_ISSUES[issue]["aspect"]
            out.append({
                "issue": issue,
                "aspect": aspect,
                "reports": len(hits),
                "aspect_negative_share": round(stats[aspect].share("negative"), 3) if aspect in stats else None,
                "evidence": [{"chunk_id": c["chunk_id"], "quote": shorten(b)} for c, b, _ in hits[:2]],
            })
        out.sort(key=lambda d: d["reports"], reverse=True)
        return out

    def _complaints(self, stats: dict[str, AspectStats]) -> list[dict]:
        out = []
        for a, st in stats.items():
            if st.n >= MIN_ASPECT_CHUNKS and st.share("negative") >= COMPLAINT_NEG_SHARE:
                out.append({
                    "aspect": a, "chunks": st.n,
                    "negative_share": round(st.share("negative"), 3),
                    "evidence": st.best_quote("negative"),
                })
        # rank by share, weighted by how much evidence there is
        out.sort(key=lambda d: d["negative_share"] * math.log1p(d["chunks"]), reverse=True)
        return out

    def _polarized(self, stats: dict[str, AspectStats]) -> list[dict]:
        out = []
        for a, st in stats.items():
            pos, neg = st.share("positive"), st.share("negative")
            if st.n >= POLARIZED_MIN_CHUNKS and pos >= POLARIZED_MIN_SHARE and neg >= POLARIZED_MIN_SHARE:
                out.append({
                    "aspect": a, "chunks": st.n,
                    "positive_share": round(pos, 3), "negative_share": round(neg, 3),
                    "entropy": round(st.entropy(), 3),
                    "positive_example": st.best_quote("positive"),
                    "negative_example": st.best_quote("negative"),
                })
        out.sort(key=lambda d: d["entropy"], reverse=True)
        return out

    def _missing(self, stats: dict[str, AspectStats], required: list[str]) -> list[dict]:
        return [{"aspect": a, "chunks": stats[a].n if a in stats else 0}
                for a in required
                if (stats[a].n if a in stats else 0) <= MISSING_MAX_CHUNKS]

    def _alternatives(self, asin: str, info: dict, breaker: dict, stats: dict[str, AspectStats],
                      candidates: list[str]) -> list[dict]:
        """Similar products that are better on the deal-breaker problem.

        Two kinds of evidence (reviews rarely PRAISE durability, so both are needed):
          basis="praised"     >= 3 chunks on the aspect, >= 50% positive, better than this product
          basis="no_reports"  >= 15 review chunks, zero negative reports of the same
                              severe issue, and an average rating at least as good
        """
        aspect, issue = breaker["aspect"], breaker["issue"]
        issue_rx = _SEVERE_RE[issue]
        own = stats.get(aspect)
        own_fix = (own.share("positive") - own.share("negative")) if own and own.n else -1.0
        own_rating = info["avg_rating"] or 0.0
        own_title = info["title"].lower().strip()
        scored = []
        for cand in candidates:
            cinfo = self.store.product_info(cand)
            if cinfo["title"].lower().strip() == own_title:
                continue  # colour/size variant of the same product
            reviews = [c for c in self.store.product_chunks(cand) if c["chunk_type"] == "review"]
            hits = [split_prefix(c["text"])[1] for c in reviews]
            hits = [b for b in hits if issue_rx.search(b)]
            issue_reports = sum(s.label == "negative" for s in self.sentiment(hits)) if hits else 0
            if issue_reports >= DEAL_BREAKER_MIN_REPORTS:
                continue  # has the same problem

            cst = self.aspect_stats(cand, only=[aspect]).get(aspect)
            n_aspect = cst.n if cst else 0
            pos = cst.share("positive") if cst else 0.0
            neg = cst.share("negative") if cst else 0.0
            quote = cst.best_quote("positive", POSITIVE_ASPECT_PHRASES.get(aspect, [])) if cst else None
            rating = cinfo["avg_rating"] or 0.0
            entry = {"asin": cand, "title": cinfo["title"], "avg_rating": cinfo["avg_rating"],
                     "aspect": aspect, "issue": issue, "review_chunks": len(reviews),
                     "issue_reports": issue_reports, "aspect_chunks": n_aspect,
                     "positive_share": round(pos, 3), "negative_share": round(neg, 3),
                     "fix_score": round(pos - neg, 3), "evidence": quote}

            if n_aspect >= ALT_MIN_CHUNKS and pos >= ALT_MIN_POS_SHARE and pos - neg > own_fix:
                scored.append({**entry, "basis": "praised"})
            elif (len(reviews) >= ALT_MIN_REVIEWS and issue_reports == 0
                  and rating >= max(ALT_MIN_RATING, own_rating)):
                scored.append({**entry, "basis": "no_reports"})
        scored.sort(key=lambda d: (d["basis"] == "praised", d["fix_score"] if d["basis"] == "praised"
                                   else (d["avg_rating"] or 0), d["review_chunks"]), reverse=True)
        return scored[:ALT_SHOW]

    def _minor_concerns(self, result: dict, negatives: list) -> list[dict]:
        """Negative mentions too rare to be a 'complaint' (each product has only
        ~15 reviews), shown so the panel never hides negative evidence."""
        covered_ids = {e["chunk_id"] for d in result["deal_breakers"] for e in d["evidence"]}
        covered_aspects = ({c["aspect"] for c in result["complaints"]}
                           | {p["aspect"] for p in result["polarized"]})
        groups: dict[str, list] = defaultdict(list)
        for chunk, body, sent in negatives:
            if chunk["chunk_id"] in covered_ids or sent.score < MINOR_MIN_CONFIDENCE:
                continue
            aspect = chunk["aspect"] if chunk["aspect"] in KB_ASPECTS else "general"
            if aspect in covered_aspects:
                continue
            groups[aspect].append((sent.score, chunk, body))
        out = []
        for aspect, items in groups.items():
            items.sort(key=lambda x: x[0], reverse=True)
            _, chunk, body = items[0]
            out.append({"aspect": aspect, "mentions": len(items),
                        "evidence": {"chunk_id": chunk["chunk_id"], "quote": shorten(body)}})
        out.sort(key=lambda d: (d["aspect"] == "general", -d["mentions"]))
        return out[:MINOR_SHOW]

    # -- public entry point --------------------------------------------------

    def explain(self, asin: str, query: str | None = None,
                required_aspects: list[str] | None = None,
                with_alternatives: bool = True) -> dict:
        started = time.perf_counter()
        info = self.store.product_info(asin)
        if not self.store.has(asin):
            return {"asin": asin, "error": "product not found in knowledge base"}

        required = list(dict.fromkeys((required_aspects or []) + aspects_from_query(query)))
        stats = self.aspect_stats(asin)

        # every scored review chunk once (a chunk can sit in several aspects)
        unique = {}
        for st in stats.values():
            for c, b, s in st.chunks:
                unique[c["chunk_id"]] = (c, b, s)
        # 'general' chunks carry no aspect keyword but can still report failures
        general = [c for c in self.store.product_chunks(asin)
                   if c["chunk_type"] == "review" and c["chunk_id"] not in unique]
        if general:
            bodies = [split_prefix(c["text"])[1] for c in general]
            for c, b, s in zip(general, bodies, self.sentiment(bodies)):
                unique[c["chunk_id"]] = (c, b, s)
        negatives = [v for v in unique.values() if v[2].label == "negative"]

        deal_breakers = self._deal_breakers(stats, negatives)
        for d in deal_breakers:
            d["review_chunks"] = len(unique)
            d["report_share"] = round(d["reports"] / len(unique), 3) if unique else None

        t_analysis = time.perf_counter() - started

        t0 = time.perf_counter()
        candidates: list[str] = []
        if with_alternatives and deal_breakers and self.similar_products:
            candidates = self.similar_products(asin, info["title"], info["category"], ALT_CANDIDATES)
            candidates = [c for c in candidates if self.store.has(c)]
        t_search = time.perf_counter() - t0
        t0 = time.perf_counter()
        for breaker in deal_breakers:
            breaker["alternatives"] = (self._alternatives(asin, info, breaker, stats, candidates)
                                       if candidates else [])
        t_alts = time.perf_counter() - t0

        result = {
            **info,
            "query": query,
            "required_aspects": required,
            "review_chunks_analysed": len(unique),
            "negative_share_overall": round(len(negatives) / len(unique), 3) if unique else None,
            "deal_breakers": deal_breakers,
            "complaints": self._complaints(stats),
            "polarized": self._polarized(stats),
            "missing": self._missing(stats, required),
            "aspect_summary": {
                a: {"chunks": st.n,
                    "positive": st.count("positive"),
                    "neutral": st.count("neutral"),
                    "negative": st.count("negative")}
                for a, st in sorted(stats.items())
            },
        }
        result["minor_concerns"] = self._minor_concerns(result, negatives)
        result["headline"] = headline(result)
        result["latency_seconds"] = round(time.perf_counter() - started, 3)
        result["timing"] = {"analysis": round(t_analysis, 3),               # sentiment + rules
                            "similar_search": round(t_search, 3),           # find similar products
                            "alternatives": round(t_alts, 3)}               # score candidates
        return result


def headline(result: dict) -> str:
    """One-line summary for the top of the panel."""
    n = result["review_chunks_analysed"]
    if result["deal_breakers"]:
        top = result["deal_breakers"][0]
        return (f"{len(result['deal_breakers'])} deal breaker(s) found in {n} review chunks - "
                f"most reported: {top['issue']} ({top['reports']} reports).")
    if result["complaints"]:
        return (f"No deal breakers in {n} review chunks; main complaint: "
                f"{result['complaints'][0]['aspect']}.")
    neg = round((result.get("negative_share_overall") or 0) * n)
    if result.get("minor_concerns"):
        return (f"No deal breakers or major complaints in {n} review chunks; "
                f"{neg} negative mention(s), listed below as minor concerns.")
    return f"No deal breakers or major complaints found in {n} review chunks."


def to_markdown(result: dict) -> str:
    """Render a result as the 'Why not?' panel text (works in Streamlit st.markdown)."""
    if "error" in result:
        return f"**Why not?** {result['error']}"
    lines = [f"### ⚠️ Why not *{result['title'] or result['asin']}*?", result["headline"], ""]
    for d in result["deal_breakers"]:
        lines.append(f"🔴 **Deal breaker: {d['issue']}** - reported in {d['reports']} of "
                     f"{d['review_chunks']} review chunks")
        for e in d["evidence"]:
            lines.append(f"> \"{e['quote']}\" `[{e['chunk_id']}]`")
        if d.get("alternatives"):
            lines.append(f"\n✅ **Similar products without this problem:**")
            for a in d["alternatives"]:
                q = a["evidence"]
                name = a["title"] or a["asin"]
                if a["basis"] == "praised":
                    line = (f"- **{name}** - {a['positive_share']:.0%} positive on {a['aspect']} "
                            f"({a['aspect_chunks']} chunks)")
                else:
                    rating = f", rated {a['avg_rating']:.1f}★" if a["avg_rating"] else ""
                    line = (f"- **{name}** - 0 of {a['review_chunks']} review chunks report "
                            f"*{a['issue']}*{rating}")
                if q:
                    line += f": \"{q['quote']}\" `[{q['chunk_id']}]`"
                lines.append(line)
        lines.append("")
    shown = {d["aspect"] for d in result["deal_breakers"]} | {p["aspect"] for p in result["polarized"]}
    for c in result["complaints"]:
        if c["aspect"] in shown:  # already explained as a deal breaker or as mixed opinions
            continue
        q = c["evidence"]
        lines.append(f"🟠 **Common complaint: {c['aspect']}** - {c['negative_share']:.0%} negative "
                     f"({c['chunks']} chunks): \"{q['quote']}\" `[{q['chunk_id']}]`")
    for p in result["polarized"]:
        lines.append(f"🟡 **Mixed opinions: {p['aspect']}** - {p['positive_share']:.0%} positive vs "
                     f"{p['negative_share']:.0%} negative")
    for m in result.get("minor_concerns", []):
        q = m["evidence"]
        label = "other" if m["aspect"] == "general" else m["aspect"]
        lines.append(f"🔸 **Minor concern ({label}, {m['mentions']} mention(s)):** "
                     f"\"{q['quote']}\" `[{q['chunk_id']}]`")
    for m in result["missing"]:
        lines.append(f"⚪ **No evidence:** no reviews mention *{m['aspect']}*")
    return "\n".join(lines)
