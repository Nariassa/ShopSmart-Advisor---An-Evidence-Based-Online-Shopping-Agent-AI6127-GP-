# ShopSmart Advisor UI (Streamlit)

Basic UI framework for the project. It connects every component that exists
today and uses clearly marked stand-ins for the parts that are still missing,
so each stand-in can be replaced without changing the page.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit page (layout only) |
| `src/ui/backend.py` | Everything the page calls: search, answer, PCS, coverage, "Why not?" |
| `tests/test_ui_backend.py` | Tests for the backend (no models or data needed) |
| `Step5_UI_Colab.ipynb` | Runs the app in Colab and prints a public link |

## What the page shows

1. **Search box** with example queries from `evaluation/formal_queries_v1.jsonl`.
2. **Answer** with `[C#]` citations, and a "Sources" list linking each citation to its `chunk_id`.
3. **One card per product**:
   - title, category, average rating, ASIN
   - **evidence**: review quotes with their `chunk_id`
   - **Purchase Confidence Score** (large), with a one-line explanation and an optional breakdown
   - **requirements checklist**: ✅ covered (2+ review chunks), ⚠️ partial (1), ❌ missing (0)
   - **"Why not?" button**, which opens the panel from `src/why_not/why_not.py`
4. **Sidebar**: number of products, show/hide answer and PCS breakdown, and the system status
   (which components are real and which are stand-ins).

## Real components and stand-ins

| Part | Used when available | Stand-in otherwise | To plug in the real one |
|---|---|---|---|
| Retrieval | Dense retriever (`chroma_db`) | Product-title search (`title_vectors.npz`) | Put `chroma_db/` next to the app or set `CHROMA_PATH` |
| Answer | `gpt-4o-mini` with the RAGAS prompt (saved answers in saved mode) | Top evidence quotes with `[C#]` | Set `OPENAI_API_KEY` |
| PCS | — | **PCS v0** from the "Why not?" statistics | Replace the body of `compute_pcs()` in `src/ui/backend.py`, keeping its returned keys |
| Coverage | `aspects_from_query` + review counts | — | — |
| "Why not?" | `src/why_not/why_not.py` | — | — |

### PCS v0 (stand-in)

Weighted average of four signals (0–1), minus 15 points per deal breaker (at most 2):

| Component | Weight | Meaning |
|---|---|---|
| evidence_quality | 0.25 | review chunks found, capped at 30 (helpful votes and verified purchase are not in the KB) |
| positivity | 0.30 | 1 − share of negative review chunks |
| consistency | 0.20 | 1 − share of well-covered aspects where reviewers disagree |
| coverage | 0.25 | share of the shopper's requirements backed by reviews (left out if the query names none) |

Recency is not included because the knowledge base has no review dates.

## Run it

### Saved mode (default when `chunks.parquet` is not present)

No database, GPU or API key is needed; it runs on any laptop in seconds. It shows the team's
**real saved outputs** for the 20 formal test questions:

| Shown in the UI | Saved file |
|---|---|
| Search results (Dense, BGE) | `results/formal_dense_results.jsonl` |
| Answer with `[C1]`..`[C5]` citations (gpt-4o-mini) | `results/ragas_eval/answers.jsonl` |
| "Why not?" for the top-3 products (RoBERTa sentiment) | `results/why_not_formal_results.jsonl` |

Pick a question from the example buttons or the drop-down. A question that is **not** one of the
20 still works, but as a clearly labelled approximation: keyword search over the ~800 saved review
chunks and a simple word-list sentiment.

```bash
pip install streamlit pandas pyarrow numpy
streamlit run app.py
```

### Full mode (any question, live)

Needs `chunks.parquet` (4.45M chunks). With `chroma_db/` it uses Dense retrieval; without it, it uses the
product-title index `title_vectors.npz`, which is built on first run if missing.

```bash
pip install -r requirements.txt streamlit transformers openai
SHOPSMART_MODE=full streamlit run app.py      # Windows PowerShell: $env:SHOPSMART_MODE="full"; streamlit run app.py
```

Environment variables: `SHOPSMART_MODE` (`saved` / `full`), `CHUNKS_PATH` (default `chunks.parquet`),
`CHROMA_PATH` (`chroma_db`), `TITLE_CACHE` (`title_vectors.npz`), `OPENAI_API_KEY` (optional).
The first page load in full mode takes a few minutes; models and data are then cached.

**Colab:** open `Step5_UI_Colab.ipynb`, choose `MODE`, run all cells and open the printed link.

## Tests

```bash
python -m unittest tests.test_ui_backend tests.test_why_not -v
```

## For the person improving the UI

- Change the look in `app.py` only. All data comes from `ShopSmartBackend`.
- Use saved mode while designing: each reload takes seconds, and the 20 test questions show real results.
- Ideas: a coloured PCS gauge, product images, a side-by-side comparison of two
  products, clickable `[C#]` citations that scroll to the evidence, and dark-mode colours.
