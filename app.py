"""
ShopSmart Advisor - Streamlit UI (basic framework).

Run from the repository root:
    streamlit run app.py

Modes (environment variable SHOPSMART_MODE):
    saved   the team's saved results for the 20 test questions (results/ folder) - no database,
            GPU or API key; other questions get a labelled keyword-search approximation
    full    the 4.45M-chunk knowledge base and live models - any question
    (default: full if chunks.parquet exists, otherwise saved)

Other settings (environment variables, full mode):
    CHUNKS_PATH             default chunks.parquet
    CHROMA_PATH             default chroma_db   (if missing -> product-title search stand-in)
    TITLE_CACHE             default title_vectors.npz
    OPENAI_API_KEY          if set -> LLM answers; otherwise evidence-quote answers

Layout (top to bottom):
    header + search box + example queries
    answer with [C#] citations
    one card per product: evidence | PCS + coverage checklist | "Why not?" button and panel
All logic lives in src/ui/backend.py, so the look can be changed here without touching it.
"""

import json
import os
from pathlib import Path

import streamlit as st

from src.ui.backend import COVERAGE_ICON, ShopSmartBackend

st.set_page_config(page_title="ShopSmart Advisor", page_icon="🛒", layout="wide")

CHUNKS_PATH = os.environ.get("CHUNKS_PATH", "chunks.parquet")
MODE = os.environ.get("SHOPSMART_MODE") or ("saved" if os.environ.get("SHOPSMART_DEMO") == "1"
                                            or not os.path.exists(CHUNKS_PATH) else "full")


# ---------------------------------------------------------------------------
# Loading (once per server, not on every click)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the knowledge base and models (first time only)...")
def load_backend(mode: str) -> ShopSmartBackend:
    if mode == "saved":
        return ShopSmartBackend.from_saved_results()
    return ShopSmartBackend.load(
        chunks_path=CHUNKS_PATH,
        chroma_path=os.environ.get("CHROMA_PATH", "chroma_db"),
        title_cache=os.environ.get("TITLE_CACHE", "title_vectors.npz"),
        openai_key=os.environ.get("OPENAI_API_KEY"),
    )


@st.cache_data
def test_questions() -> list[str]:
    path = Path(__file__).parent / "evaluation" / "formal_queries_v1.jsonl"
    try:
        return [json.loads(line)["query"] for line in open(path, encoding="utf-8")]
    except OSError:
        return ["Which wireless earbuds have long battery life and stay comfortable during long flights?"]


try:
    backend = load_backend(MODE)
except Exception as exc:  # show a readable message instead of a stack trace
    st.error(f"Could not load the backend: {exc}")
    st.info("Check the file paths in UI_README.md, or run in saved mode with SHOPSMART_MODE=saved.")
    st.stop()

questions = backend.saved_questions() if backend.mode == "saved" else test_questions()


# ---------------------------------------------------------------------------
# Sidebar: settings and system status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    n_products = st.slider("Products to show", 1, 10, 5)
    show_answer = st.toggle("Show answer", value=True)
    show_breakdown = st.toggle("Show PCS breakdown", value=False)

    st.divider()
    st.subheader("System status")
    st.caption(f"**Retrieval:** {backend.retrieval_mode}")
    st.caption(f"**Answer:** {backend.answer_mode}")
    st.caption("**Confidence score:** PCS v0 (UI stand-in)")
    if backend.mode_note:
        st.caption(f"**Mode:** {backend.mode_note}")


# ---------------------------------------------------------------------------
# Header and search
# ---------------------------------------------------------------------------

st.title("🛒 ShopSmart Advisor")
st.caption("Evidence-based shopping advice from Amazon Electronics reviews - every claim links to the review it came from.")

if "query" not in st.session_state:
    st.session_state.query = ""

with st.form("search"):
    query = st.text_input("What are you looking for?", value=st.session_state.query,
                          placeholder="e.g. wireless earbuds with long battery life for flights")
    submitted = st.form_submit_button("Search", type="primary")

st.caption("Try an example:")
cols = st.columns(3)
for i, q in enumerate(questions[:6]):
    if cols[i % 3].button(q, key=f"ex{i}", use_container_width=True):
        st.session_state.query = q
        st.session_state.why_not_open = set()
        st.rerun()
if len(questions) > 6:
    picked = st.selectbox(f"...or choose any of the {len(questions)} test questions", [""] + questions,
                          key="pick", format_func=lambda q: q or "Select a question")
    if picked and picked != st.session_state.get("last_pick"):
        st.session_state.last_pick = picked
        st.session_state.query = picked
        st.session_state.why_not_open = set()
        st.rerun()

if submitted:
    st.session_state.query = query
    st.session_state.why_not_open = set()

query = st.session_state.query.strip()
if not query:
    st.info("Type what you need above, or pick an example.")
    st.stop()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

with st.spinner("Searching reviews..."):
    hits, search_seconds = backend.search(query, n_products)

if backend.is_approximate(query):
    st.warning("This question is not one of the saved test questions, so these results are an "
               "**approximation**: keyword search over the saved review chunks and a simple word-list "
               "sentiment. Pick a test question for the real results, or run in full mode.")

if not hits:
    st.warning("No matching products found. Try different words.")
    st.stop()

if show_answer:
    with st.spinner("Writing the answer..."):
        answer = backend.answer(query, hits)
    st.subheader("💬 Answer")
    st.markdown(answer["text"])
    if answer["citations"]:
        with st.expander("Sources for this answer"):
            for cid, c in answer["citations"].items():
                st.markdown(f"**[{cid}]** {c['product']} - \"{c['quote']}\" `{c['chunk_id']}`")

st.subheader(f"🔎 Top {len(hits)} products")
st.session_state.setdefault("why_not_open", set())

for hit in hits:
    analysis = backend.analyse(hit, query)
    pcs, coverage = analysis["pcs"], analysis["coverage"]

    with st.container(border=True):
        left, right = st.columns([3, 2], gap="large")

        with left:
            rating = f" · ⭐ {hit.avg_rating:.1f}" if hit.avg_rating else ""
            short = hit.title if len(hit.title) <= 90 else hit.title[:89].rsplit(" ", 1)[0] + "…"
            st.markdown(f"#### {hit.rank}. {short}", help=hit.title)
            st.caption(f"{hit.category or 'Unknown category'}{rating} · ASIN {hit.asin}")
            st.markdown("**📄 Evidence from reviews**")
            for e in hit.evidence:
                label = "spec" if e.chunk_type == "spec" else e.aspect
                st.markdown(f"> \"{e.quote}\"  \n> <small>`{e.chunk_id}` · {label}</small>",
                            unsafe_allow_html=True)

        with right:
            st.metric("🎯 Purchase Confidence Score", f"{pcs['score']}%")
            st.progress(pcs["score"] / 100)
            st.caption(pcs["explanation"])

            if coverage:
                st.markdown("**Your requirements**")
                st.markdown("  ".join(f"{COVERAGE_ICON[c['status']]} {c['aspect']}" for c in coverage))
                notes = [c["note"] for c in coverage if c.get("note")]
                if notes:
                    st.caption("; ".join(notes))

            if show_breakdown:
                st.markdown("**PCS breakdown**")
                for name, value in pcs["components"].items():
                    w = pcs["weights"].get(name, 0)
                    st.caption(f"{name.replace('_', ' ')}: {value:.2f} (weight {w:.2f})")
                if pcs["deal_breakers"]:
                    st.caption(f"deal breaker penalty: -{min(pcs['deal_breakers'], 2) * 15} points")
                st.caption(f"Version: {pcs['version']}")

            opened = hit.asin in st.session_state.why_not_open
            if st.button("Hide 'Why not?'" if opened else "⚠️ Why not?", key=f"wn_{hit.asin}"):
                st.session_state.why_not_open ^= {hit.asin}
                st.rerun()

        if hit.asin in st.session_state.why_not_open:
            with st.spinner("Checking the reviews for problems..."):
                panel = backend.why_not_markdown(hit.asin, query)
            with st.container(border=True):
                st.markdown(panel)

st.caption(f"Search took {search_seconds:.2f} s · {backend.retrieval_mode}")
