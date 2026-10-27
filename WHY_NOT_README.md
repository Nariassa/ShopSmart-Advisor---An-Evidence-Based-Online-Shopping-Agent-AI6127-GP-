# "Why not?" module

Explains why a shopper might **not** want to buy a product, using only review
evidence from the knowledge base, with every claim cited by `chunk_id`.

## Files

| File | Purpose |
|---|---|
| `src/why_not/why_not.py` | The logic (import this from the UI) |
| `src/why_not/__init__.py` | Makes `src/why_not` a Python package |
| `tests/test_why_not.py` | Unit tests on small fake data (no model or KB needed) |
| `Step4_WhyNot.ipynb` | Colab notebook: run on the real KB, batch results, sanity check, export for human checking |

## What it returns

`WhyNotExplainer.explain(asin, query=None)` returns one JSON-serialisable dict:

| Key | Meaning |
|---|---|
| `headline` | One-line summary for the top of the panel |
| `deal_breakers` | Severe issues reported at least twice (stopped working, won't charge, broke, overheating, connection drops, returned/refund), with quotes |
| `deal_breakers[].alternatives` | Similar products (same category, found by dense search on the title) that are at least 50% positive on the same aspect and better than this product, each with a positive quote |
| `complaints` | Aspects with at least 3 chunks and at least 30% negative |
| `polarized` | Aspects with at least 30% positive and at least 30% negative (reviewers disagree) |
| `missing` | Aspects the shopper asked about with at most 1 review chunk |
| `aspect_summary` | Positive / neutral / negative counts per aspect (can feed PCS) |
| `latency_seconds` | Time for this explanation |

Alternatives are only searched when a deal breaker exists: if the product has
no serious problem, the panel does not push other products.

## Method

1. **Evidence**: all review chunks of the product from `chunks.parquet`
   (indexed by `parent_asin` for constant-time lookup).
2. **Sentiment**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
   (positive / neutral / negative), which covers subjectivity detection
   (neutral vs opinionated) and polarity detection. The product-title prefix is
   removed before scoring.
3. **Aspects**: the 10 KB aspects plus two extra keyword aspects, *comfort*
   and *durability*, which the KB taxonomy lacks but shoppers ask about.
4. **Deal breakers**: rule-based severe-failure patterns, applied only to
   negative chunks, including "general" chunks that have no aspect label.
5. **Similar products**: dense search over Chroma when `chroma_db` exists;
   otherwise `TitleSimilarity` embeds one title per product with the same BGE
   model (cached to a `.npz` file) and searches within the same category.
6. **Alternatives** (only when there is a deal breaker), two kinds of evidence:
   - *praised*: at least 3 chunks on the deal-breaker aspect, at least 50%
     positive, and a better positive-minus-negative score than this product;
     shown with a quote that praises the aspect.
   - *no_reports*: at least 15 review chunks, zero negative reports of the
     same severe issue, and an average rating of at least 4.0 and not lower
     than this product's. Needed because reviewers rarely praise durability.
   Colour variants (same title) and products with the same issue are excluded.

The thresholds are constants at the top of `why_not.py`, so they can be varied
in the ablation study.

## Usage

```python
from src.why_not.why_not import WhyNotExplainer, to_markdown

explainer = WhyNotExplainer.from_paths("chunks.parquet", "chroma_db")
result = explainer.explain("B0764GNRDZ", query="earbuds for long flights")
print(to_markdown(result))   # in Streamlit: st.markdown(to_markdown(result))
```

## Tests

```bash
python -m unittest tests.test_why_not -v
```

## Limitations

- Each chunk has the product's **average** rating, not the reviewer's own
  star rating, so sentiment comes from a model rather than from stars.
- Aspect labels in the KB are keyword-based, so some chunks are mislabelled.
- Severe-issue detection is rule-based; precision is measured with the
  human-check export in the notebook.
- Each product has at most about 15 reviews, so shares on small aspects are noisy;
  the minimum-evidence thresholds guard against this.
