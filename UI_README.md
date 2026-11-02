# ShopSmart UI: run and hand off

## Current delivery

The shopping interface has two workspaces:

- Discover: editable queries, grounded recommendations, source inspection, evidence, requirement fit, risk analysis and adding products to comparison.
- Compare: up to three products, source evidence, requirement coverage, removal and JSON export.
- Day/Night appearance; English consumer-facing copy. Evaluation/debugging information stays outside the product UI.

## Quick preview

From this folder:

```powershell
python launch_ui.py --mode saved --port 8501
```

An existing preview may already occupy port 8501; use 8503 to run another instance. Saved mode uses the installed base Python environment. It requires streamlit, pandas, pyarrow and numpy.

## Full knowledge base

The launcher defaults to data in the parent directory: `chunks.parquet` plus `embedding_shards/` for bounded-memory search, or `chunks.parquet` plus `chroma_db/` for Chroma search. Keep large artifacts outside Git. Existing artifacts can be used directly without copying or reindexing.

For another location set `SHOPSMART_DATA_DIR` to the directory containing those files. Individual overrides `CHUNKS_PATH`, `CHROMA_PATH` and `EMBEDDING_SHARDS_PATH` are also supported.

```powershell
$env:SHOPSMART_DATA_DIR = 'D:\data\electronics_kb'
python launch_ui.py --mode full --port 8502
```

The default full launcher uses memory-mapped embedding shards and on-demand Parquet reads. It searches ALL 4,450,749 BGE vectors with exact cosine scoring in bounded batches; it is slower than a resident Chroma HNSW index but avoids loading that roughly 14 GB index plus the entire corpus into RAM. The original Chroma path remains available on a computer with sufficient memory:

```powershell
python launch_ui.py --mode full --retriever chroma --port 8502
```

Both paths use the same corpus and BAAI/bge-base-en-v1.5 embeddings. On this 16 GB computer only about 3 GB RAM was free during inspection, so loading the full Chroma index was not attempted.

## Environment setup

Use Python 3.11 and create the project-local environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
```

The first full run downloads the BGE and Cardiff sentiment models if not cached. Model downloads require internet access. OPENAI_API_KEY is optional: without it the app returns cited evidence excerpts; with it, the current UI backend calls gpt-4o-mini for generated recommendations. Provide credentials through the process environment; do not put them in source files or chat.

## Current verified status (2026-11-02)

The earlier native-library load failures no longer reproduced on retry. Data libraries, Torch, the BGE encoder and the Cardiff sentiment pipeline loaded successfully without changing Windows security settings. Both models have now been downloaded and cached.

Full knowledge-base validation passed for five real queries: each returned three products, source evidence, team PCS scores, requirement coverage and risk analysis. Measured search calls took 23.15-25.43 seconds in the backend validation run. The separate full-data UI check also passed search, adding three products, Compare, Night mode and removing all products.

After startup, full mode is available on http://127.0.0.1:8502; saved mode can run separately on port 8501. First page preparation and new live searches can take time; wait for the loading indicator to finish. Appearance changes and comparison actions reuse existing search and answer outputs.

The validation run used no OpenAI API key. Live search and transformer sentiment passed, while answers used cited review excerpts. Configure OPENAI_API_KEY in the launching environment only if generated recommendation prose is required.

## PCS integration

Full mode now calls the collaborator's `calculate_pcs` from `origin/wangzhongda` through `src/ui/pcs_adapter.py`; the three required source modules were copied unchanged. Their exact source commit is documented in `src/generation/UI_INTEGRATION.md`. This is an import of selected code, not a merge of the full generation branch.

The adapter supplies actual product review evidence and existing transformer sentiment to the PCS module and preserves the UI return shape. It uses the available product-review scope, not an invented review count. Missing timestamps/verification/helpfulness remain missing. Saved mode keeps the historical v0 scoring behavior. The collaborator's PCS is experimental evidence confidence, not purchase-success probability.

## Verification completed

- Actual Parquet metadata and all 45 embedding shard headers match: 4,450,749 rows/vectors.
- Real product evidence for two ASINs read successfully from the full corpus using bounded reads.
- 21 tests pass: existing backend/risk tests plus exact shard-search and lazy evidence/PCS adaptation checks.
- UI smoke checks cover the 20 saved queries, no results, empty input, three-product comparison, removing products, exports being rendered, both themes and New search.
- Search and answer outputs are reused for presentation-only interactions so changing themes or adding to comparison does not issue another paid generation request.
- Live runtime and five-query checks passed. The full-data UI check passed. Paid generation was not called; no API key is configured in the inspected session.

After native libraries load, run the actual five-query check (may take several minutes):

```powershell
.\.venv\Scripts\python.exe scripts/check_live_ui.py
```

This writes an ignored local live-validation.json only after the checks succeed. Run scripts/check_full_ui.py with the project environment to repeat the full UI smoke check without paid generation.

## Scope

This delivery concerns UI and integration. Annotation is handled by collaborators; annotation/report completion is not assessed or assigned to the UI owner.
