"""
backend.py - everything the ShopSmart UI needs, kept separate from the page layout.

The app (app.py) only calls ShopSmartBackend. Each part has a working stand-in,
so the UI runs before every team component is finished, and each stand-in is
replaced by changing one function:

    Part              Real component (used when available)     Stand-in (used otherwise)
    ----------------  ---------------------------------------  ------------------------------------
    Retrieval         Dense retriever over chroma_db           Product-title search (title_vectors.npz)
    Answer            LLM answer with [C#] citations (OpenAI)  Top evidence quotes with [C#] citations
    PCS               Zhongda's PCS module (plug into          PCS v0 computed from the "Why not?"
                      compute_pcs)                             numbers below
    Coverage          aspects_from_query + review evidence     (same - already real)
    "Why not?"        src/why_not/why_not.py                   (same - already real)

Two modes:
    full    the 4.45M-chunk knowledge base (chunks.parquet) and live models  -> any question
    saved   the team's saved outputs in results/ - no database, GPU or API key needed:
              * the 20 formal test questions show the REAL saved results:
                Dense top-20 (formal_dense_results.jsonl), GPT answers with [C#] citations
                (ragas_eval/answers.jsonl) and "Why not?" for the top-3 products
                (why_not_formal_results.jsonl, RoBERTa sentiment)
              * any other question is answered approximately: keyword search over the ~800
                saved review chunks and a word-list sentiment (clearly labelled in the UI)
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.why_not.why_not import (
    ReviewStore, Sentiment, TitleSimilarity, WhyNotExplainer, aspects_from_query,
    clean_text, shorten, split_prefix, to_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Same instruction the team used for the RAGAS evaluation (scripts/ragas_eval/run_ragas.py),
# so answers in the UI match the evaluated generator.
ANSWER_SYSTEM_PROMPT = (
    "You are ShopSmart Advisor. Answer the shopper's question using ONLY the numbered evidence. "
    "Recommend at most 3 products, name them, say why with specifics, mention one drawback if the "
    "evidence shows one, and cite evidence as [C1], [C2]. If the evidence is insufficient, say so. "
    "Do not use outside knowledge."
)
ANSWER_CONTEXTS = 5          # evidence chunks given to the generator (k=5, as in RAGAS)
EVIDENCE_PER_PRODUCT = 3
CANDIDATE_CHUNKS = 60        # dense chunks fetched before grouping into products

STOPWORDS = set("""a an and are as at be best but by can do does for from good great have i if in is it
its me my need of on or that the this to under very want what which who will with would you your
looking recommend should get buy""".split())


# ---------------------------------------------------------------------------
# Data shapes passed to the UI
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    chunk_id: str
    aspect: str
    chunk_type: str
    quote: str
    score: float | None = None


@dataclass
class ProductHit:
    asin: str
    title: str
    category: str | None
    avg_rating: float | None
    rank: int
    evidence: list[Evidence] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PCS v0 (stand-in until the team's PCS module is ready)
# ---------------------------------------------------------------------------

PCS_WEIGHTS = {"evidence_quality": 0.25, "positivity": 0.30, "consistency": 0.20, "coverage": 0.25}
PCS_DEAL_BREAKER_PENALTY = 0.15      # per deal breaker, at most 2 counted
PCS_FULL_EVIDENCE_CHUNKS = 30        # this many review chunks = full evidence-quality score


def compute_pcs(why_not_result: dict, coverage: list[dict]) -> dict:
    """Purchase Confidence Score v0 (0-100) with a breakdown.

    Follows the plan's formula with the signals the knowledge base has:
      evidence_quality  amount of review evidence (helpful votes / verified purchase are not in the KB)
      positivity        1 - share of negative review chunks
      consistency       1 - share of well-covered aspects where reviewers disagree (polarization)
      coverage          share of the shopper's required aspects backed by reviews
      recency           not available (no review timestamps in the KB) -> left out
    minus a penalty for each deal breaker.

    >>> REPLACE this function body with the team's PCS module when it is ready;
        keep the returned keys so app.py does not need to change.
    """
    n = why_not_result.get("review_chunks_analysed", 0) or 0
    neg_share = why_not_result.get("negative_share_overall") or 0.0
    judged = [a for a, s in why_not_result.get("aspect_summary", {}).items() if s["chunks"] >= 4]
    polarized = {p["aspect"] for p in why_not_result.get("polarized", [])}

    components = {
        "evidence_quality": min(1.0, n / PCS_FULL_EVIDENCE_CHUNKS),
        "positivity": 1.0 - neg_share if n else 0.0,
        "consistency": 1.0 - (len(polarized & set(judged)) / len(judged)) if judged else 1.0,
    }
    if coverage:
        components["coverage"] = sum(c["value"] for c in coverage) / len(coverage)
    weights = {k: w for k, w in PCS_WEIGHTS.items() if k in components}
    total_w = sum(weights.values())
    score = sum(components[k] * w for k, w in weights.items()) / total_w
    n_breakers = len(why_not_result.get("deal_breakers", []))
    score -= PCS_DEAL_BREAKER_PENALTY * min(n_breakers, 2)
    score = max(0.0, min(1.0, score))

    return {
        "score": round(100 * score),
        "components": {k: round(v, 3) for k, v in components.items()},
        "weights": {k: round(w / total_w, 3) for k, w in weights.items()},
        "deal_breakers": n_breakers,
        "explanation": _pcs_sentence(round(100 * score), components, n, neg_share, polarized,
                                     coverage, n_breakers),
        "version": "v0 (UI stand-in)",
    }


def _pcs_sentence(score, comp, n, neg_share, polarized, coverage, n_breakers) -> str:
    parts = []
    q = comp["evidence_quality"]
    parts.append(f"{'strong' if q >= 0.8 else 'moderate' if q >= 0.4 else 'limited'} evidence ({n} review chunks)")
    parts.append("mostly positive reviews" if neg_share < 0.2 else
                 "mixed reviews" if neg_share < 0.4 else "many negative reviews")
    parts.append(f"reviewers disagree on {', '.join(sorted(polarized))}" if polarized else "low polarization")
    missing = [c["aspect"] for c in coverage if c["status"] == "missing"]
    if missing:
        parts.append(f"no evidence on {', '.join(missing)}")
    if n_breakers:
        parts.append(f"{n_breakers} deal breaker(s)")
    return f"{score}% confidence - " + ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Evidence coverage checklist
# ---------------------------------------------------------------------------

PRICE_RE = re.compile(r"\$\s?\d|\bunder \d|\bcheap|\bbudget|\bprice", re.IGNORECASE)


def coverage_checklist(query: str, why_not_result: dict) -> list[dict]:
    """Shopper's required aspects -> covered (>=2 review chunks) / partial (1) / missing (0)."""
    summary = why_not_result.get("aspect_summary", {})
    rows = []
    for aspect in aspects_from_query(query):
        n = summary.get(aspect, {}).get("chunks", 0)
        status = "covered" if n >= 2 else "partial" if n == 1 else "missing"
        rows.append({"aspect": aspect, "chunks": n, "status": status,
                     "value": {"covered": 1.0, "partial": 0.5, "missing": 0.0}[status]})
    if PRICE_RE.search(query or ""):
        note = "actual prices are not in the knowledge base; this checks what reviews say about value"
        price_rows = [r for r in rows if r["aspect"] == "price/value"]
        if price_rows:
            price_rows[0]["note"] = note
        else:
            rows.append({"aspect": "price/value", "chunks": summary.get("price/value", {}).get("chunks", 0),
                         "status": "missing", "value": 0.0, "note": note})
    return rows


COVERAGE_ICON = {"covered": "✅", "partial": "⚠️", "missing": "❌"}


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class ShopSmartBackend:
    def __init__(self, explainer: WhyNotExplainer, dense_retriever=None, title_search=None,
                 openai_key: str | None = None, mode_note: str = ""):
        self.explainer = explainer
        self.store: ReviewStore = explainer.store
        self.dense = dense_retriever
        self.title_search = title_search
        self.openai_key = openai_key
        self.mode_note = mode_note
        self._explain_cache: OrderedDict = OrderedDict()
        self.saved: dict[str, dict] = {}          # normalised query -> saved outputs (saved mode only)
        self.mode = "full"

    # -- construction ----------------------------------------------------------

    @classmethod
    def load(cls, chunks_path: str = "chunks.parquet", chroma_path: str | None = "chroma_db",
             title_cache: str | None = "title_vectors.npz", openai_key: str | None = None,
             model_cache: str | None = None) -> "ShopSmartBackend":
        """Full mode: real knowledge base. Uses chroma_db when it exists."""
        from src.why_not.why_not import SentimentScorer, dense_similar_products

        store = ReviewStore.from_parquet(chunks_path)
        dense = None
        if chroma_path and os.path.isdir(chroma_path):
            from src.retrieval.dense_retriever import DenseRetriever
            dense = DenseRetriever(chroma_path=chroma_path, cache_folder=model_cache).load()
            similar, title = dense_similar_products(dense), None
        else:
            title = TitleSimilarity(store, cache_path=title_cache)
            similar = title
        explainer = WhyNotExplainer(store, SentimentScorer(), similar)
        return cls(explainer, dense_retriever=dense, title_search=title,
                   openai_key=openai_key or os.environ.get("OPENAI_API_KEY"))

    @classmethod
    def from_saved_results(cls, results_dir: str | Path | None = None) -> "ShopSmartBackend":
        """Saved mode: replay the team's saved outputs (see module docstring)."""
        results_dir = Path(results_dir or REPO_ROOT / "results")
        rows, dense_runs = {}, {}
        for path in sorted(glob.glob(str(results_dir / "formal_*_results.jsonl"))):
            for line in open(path, encoding="utf-8"):
                run = json.loads(line)
                if Path(path).name == "formal_dense_results.jsonl":
                    dense_runs[run["query_id"]] = run
                for r in run["results"]:
                    rows.setdefault(r["chunk_id"], {k: r.get(k) for k in ReviewStore.COLUMNS})
        if not rows:
            raise FileNotFoundError(f"No formal_*_results.jsonl files found in {results_dir}.")
        store = ReviewStore(pd.DataFrame(list(rows.values())))
        search = KeywordProductSearch(store)
        backend = cls(WhyNotExplainer(store, lexicon_sentiment, search), title_search=search)
        backend.mode = "saved"

        answers = _load_json_or_jsonl(results_dir / "ragas_eval" / "answers.jsonl")
        answers = {a["query_id"]: a for a in answers}
        why_not = defaultdict(dict)
        for r in _load_json_or_jsonl(results_dir / "why_not_formal_results.jsonl"):
            why_not[r["query_id"]][r["asin"]] = r
        for qid, run in dense_runs.items():
            backend.saved[normalise(run["query"])] = {
                "query_id": qid, "query": run["query"], "dense": run["results"],
                "answer": answers.get(qid, {}).get("response"), "why_not": why_not.get(qid, {}),
            }
        backend.mode_note = (f"Saved results: {len(backend.saved)} test questions "
                             f"({len(answers)} saved answers, {sum(map(len, why_not.values()))} saved "
                             f"'Why not?' results); {len(rows):,} saved review chunks")
        return backend

    demo = from_saved_results   # older name, kept so existing scripts keep working

    # -- saved-mode helpers ------------------------------------------------------

    def saved_questions(self) -> list[str]:
        return [v["query"] for v in sorted(self.saved.values(), key=lambda v: v["query_id"])]

    def saved_entry(self, query: str) -> dict | None:
        return self.saved.get(normalise(query))

    def is_approximate(self, query: str) -> bool:
        """True when the results come from the saved-mode keyword fallback."""
        return self.mode == "saved" and self.saved_entry(query) is None

    # -- status shown in the sidebar ------------------------------------------

    @property
    def retrieval_mode(self) -> str:
        if self.mode == "saved":
            return "Saved Dense results (BGE) for the test questions; keyword search otherwise"
        if self.dense is not None:
            return "Dense retrieval (BGE + Chroma)"
        return "Product-title search (stand-in: no chroma_db)"

    @property
    def answer_mode(self) -> str:
        if self.mode == "saved":
            return "Saved gpt-4o-mini answers (RAGAS run) for the test questions; review quotes otherwise"
        return "LLM (gpt-4o-mini)" if self.openai_key else "Evidence quotes (stand-in: no API key)"

    # -- 1. search -------------------------------------------------------------

    def search(self, query: str, n_products: int = 5) -> tuple[list[ProductHit], float]:
        started = time.perf_counter()
        entry = self.saved_entry(query)
        if entry is not None:
            hits = self._saved_search(entry, n_products)
        elif self.dense is not None:
            hits = self._dense_search(query, n_products)
        else:
            hits = self._title_search(query, n_products)
        return hits, time.perf_counter() - started

    def _dense_search(self, query: str, n_products: int) -> list[ProductHit]:
        results, _ = self.dense.search(query, top_k=CANDIDATE_CHUNKS)
        grouped: OrderedDict[str, list] = OrderedDict()
        for r in results:
            asin = r.metadata.get("parent_asin")
            if asin and self.store.has(asin):
                grouped.setdefault(asin, []).append(r)
        hits = []
        for rank, (asin, rs) in enumerate(list(grouped.items())[:n_products], 1):
            by_id = {c["chunk_id"]: c for c in self.store.product_chunks(asin)}
            ev = [self._evidence(by_id[r.chunk_id], r.score) for r in rs[:EVIDENCE_PER_PRODUCT]
                  if r.chunk_id in by_id]
            hits.append(self._hit(asin, rank, ev))
        return hits

    def _saved_search(self, entry: dict, n_products: int) -> list[ProductHit]:
        """Saved Dense top-20 grouped by product. Only products with a saved "Why not?"
        result are shown (the top 3), so every card uses real saved analysis."""
        grouped: OrderedDict[str, list] = OrderedDict()
        for r in entry["dense"]:
            grouped.setdefault(r["parent_asin"], []).append(r)
        hits = []
        for asin, rs in grouped.items():
            if asin not in entry["why_not"] or not self.store.has(asin):
                continue
            by_id = {c["chunk_id"]: c for c in self.store.product_chunks(asin)}
            ev = [self._evidence(by_id[r["chunk_id"]], r.get("score")) for r in rs[:EVIDENCE_PER_PRODUCT]]
            hits.append(self._hit(asin, len(hits) + 1, ev))
            if len(hits) >= n_products:
                break
        return hits

    def _title_search(self, query: str, n_products: int) -> list[ProductHit]:
        asins = self.title_search("", query, None, n_products * 3)
        hits = []
        for asin in asins:
            ev = self._best_chunks_for_query(asin, query)
            if not ev:
                continue
            hits.append(self._hit(asin, len(hits) + 1, ev))
            if len(hits) >= n_products:
                break
        return hits

    def _best_chunks_for_query(self, asin: str, query: str) -> list[Evidence]:
        words = query_terms(query)
        wanted = set(aspects_from_query(query))
        scored = []
        for c in self.store.product_chunks(asin):
            body = split_prefix(c["text"])[1].lower()
            s = sum(w in body for w in words) + 2 * (c["aspect"] in wanted) + 0.1 * (c["chunk_type"] == "review")
            scored.append((s, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._evidence(c, float(s)) for s, c in scored[:EVIDENCE_PER_PRODUCT] if s > 0]

    def _evidence(self, chunk: dict, score) -> Evidence:
        return Evidence(chunk_id=chunk["chunk_id"], aspect=chunk["aspect"], chunk_type=chunk["chunk_type"],
                        quote=shorten(split_prefix(chunk["text"])[1], 260), score=score)

    def _hit(self, asin: str, rank: int, evidence: list[Evidence]) -> ProductHit:
        info = self.store.product_info(asin)
        return ProductHit(asin=asin, title=info["title"] or asin, category=info["category"],
                          avg_rating=info["avg_rating"], rank=rank, evidence=evidence)

    # -- 2. answer -------------------------------------------------------------

    def answer(self, query: str, hits: list[ProductHit]) -> dict:
        """Answer text with [C#] citations, plus the C# -> chunk mapping."""
        contexts = [(h, e) for h in hits for e in h.evidence][:ANSWER_CONTEXTS]
        citations = {f"C{i}": {"chunk_id": e.chunk_id, "product": h.title, "quote": e.quote}
                     for i, (h, e) in enumerate(contexts, 1)}
        entry = self.saved_entry(query)
        if entry is not None and entry["answer"]:
            # the saved answer cites the Dense top-5 chunks as C1..C5 (k=5 in the RAGAS run)
            citations = {}
            for i, r in enumerate(entry["dense"][:ANSWER_CONTEXTS], 1):
                info = self.store.product_info(r["parent_asin"])
                citations[f"C{i}"] = {"chunk_id": r["chunk_id"], "product": info["title"],
                                      "quote": shorten(split_prefix(r["text"])[1], 260)}
            return {"text": entry["answer"], "citations": citations, "mode": "saved"}
        if not contexts:
            return {"text": "I could not find evidence for this question.", "citations": {}, "mode": "none"}
        if self.openai_key:
            try:
                return {"text": self._llm_answer(query, contexts), "citations": citations, "mode": "llm"}
            except Exception as exc:  # network / quota problems: fall back, never crash the page
                note = f"(LLM unavailable: {type(exc).__name__}; showing evidence instead)\n\n"
                return {"text": note + extractive_answer(contexts), "citations": citations, "mode": "fallback"}
        return {"text": extractive_answer(contexts), "citations": citations, "mode": "extractive"}

    def _llm_answer(self, query: str, contexts) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)
        ev = "\n".join(f"[C{i}] {h.title}: {e.quote}" for i, (h, e) in enumerate(contexts, 1))
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0,
            messages=[{"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                      {"role": "user", "content": f"Question: {query}\n\nEvidence:\n{ev}"}])
        return resp.choices[0].message.content

    # -- 3. per-product analysis (PCS + coverage) and "Why not?" -------------

    def explain(self, asin: str, query: str, with_alternatives: bool = False) -> dict:
        entry = self.saved_entry(query)
        if entry is not None and asin in entry["why_not"]:
            return entry["why_not"][asin]      # saved result already includes alternatives
        key = (asin, query, with_alternatives)
        if key not in self._explain_cache:
            if len(self._explain_cache) > 500:
                self._explain_cache.popitem(last=False)
            self._explain_cache[key] = self.explainer.explain(asin, query=query,
                                                              with_alternatives=with_alternatives)
        return self._explain_cache[key]

    def analyse(self, hit: ProductHit, query: str) -> dict:
        result = self.explain(hit.asin, query)
        coverage = coverage_checklist(query, result)
        return {"pcs": compute_pcs(result, coverage), "coverage": coverage, "why_not": result}

    def why_not_markdown(self, asin: str, query: str) -> str:
        return to_markdown(self.explain(asin, query, with_alternatives=True))


# ---------------------------------------------------------------------------
# Stand-ins used in saved mode / without chroma_db
# ---------------------------------------------------------------------------

def normalise(query: str) -> str:
    return " ".join(re.findall(r"[a-z0-9$]+", (query or "").lower()))


def _load_json_or_jsonl(path: Path) -> list[dict]:
    """answers.jsonl is saved as one JSON list; other files are one JSON object per line."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def query_terms(query: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if w not in STOPWORDS and len(w) > 2]


def extractive_answer(contexts) -> str:
    """No-LLM answer: one line per product with its best evidence and a citation."""
    lines, seen = ["Based on the reviews, the best matches are:"], set()
    for i, (h, e) in enumerate(contexts, 1):
        if h.asin in seen:
            continue
        seen.add(h.asin)
        lines.append(f"- **{h.title}**: \"{shorten(e.quote, 140)}\" [C{i}]")
    lines.append("\n*Generated from review quotes only (no language model connected).*")
    return "\n".join(lines)


class KeywordProductSearch:
    """Saved-mode stand-in for product search: word overlap with title + chunk text.
    Same call signature as TitleSimilarity / dense_similar_products."""

    def __init__(self, store: ReviewStore):
        self.store = store
        self.docs = {}
        for asin in store._rows:
            chunks = store.product_chunks(asin)
            title = split_prefix(chunks[0]["text"])[0]
            body = " ".join(split_prefix(c["text"])[1] for c in chunks)
            self.docs[asin] = (title.lower(), body.lower(), chunks[0].get("category"))

    def __call__(self, asin: str, title: str, category, limit: int) -> list[str]:
        words = query_terms(title)
        scored = []
        for a, (t, body, cat) in self.docs.items():
            if a == asin or (category and str(cat) != str(category)):
                continue
            s = sum(2 * (w in t) + (w in body) for w in words)
            if s:
                scored.append((s, a))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored[:limit]]


_POS = set("""great good excellent love loved amazing awesome perfect best happy comfortable comfy clear
solid sturdy reliable recommend recommended fantastic nice easy fast long lasts worth impressed works""".split())
_NEG = set("""bad poor terrible awful worst broke broken died dead stopped return returned refund disappointed
disappointing cheap flimsy useless waste junk problem problems issue issues fails failed failure hate uncomfortable
hurt hurts difficult slow drops disconnects overheats noisy weak""".split())


def lexicon_sentiment(texts: list[str]) -> list[Sentiment]:
    """Tiny word-list sentiment for saved-mode questions outside the test set only (the real app uses the RoBERTa model)."""
    out = []
    for t in texts:
        words = re.findall(r"[a-z']+", clean_text(t).lower())
        neg = sum(w in _NEG for w in words) + sum(1 for i, w in enumerate(words[:-1])
                                                    if w in ("not", "never", "don't", "doesn't", "won't")
                                                    and words[i + 1] in _POS)
        pos = sum(w in _POS for w in words) - sum(1 for i, w in enumerate(words[:-1])
                                                    if w in ("not", "never") and words[i + 1] in _POS)
        if neg > pos:
            out.append(Sentiment("negative", min(0.99, 0.6 + 0.1 * (neg - pos))))
        elif pos > neg:
            out.append(Sentiment("positive", min(0.99, 0.6 + 0.1 * (pos - neg))))
        else:
            out.append(Sentiment("neutral", 0.6))
    return out
