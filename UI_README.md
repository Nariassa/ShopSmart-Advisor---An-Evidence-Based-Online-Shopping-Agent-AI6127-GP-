# ShopSmart UI: setup and startup

The UI provides Discover, a three-product Compare workspace, Day/Night themes,
and a small Team dialog in the footer. Commands below are for Windows PowerShell.
Run them from the repository root (the folder containing `launch_ui.py`).

## 1. Get the latest code

For a new checkout:

```powershell
git clone https://github.com/Nariassa/ShopSmart-Advisor---An-Evidence-Based-Online-Shopping-Agent-AI6127-GP-.git shopsmart-advisor
cd shopsmart-advisor
```

For an existing checkout, save your own changes before updating:

```powershell
git switch main
git pull --ff-only origin main
```

## 2. Create the Python environment

Use Python 3.11. Create `.venv` once, then install the dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
```

If `py -3.11` is unavailable, install Python 3.11 first or use its full executable
path. If `.venv` already exists with compatible dependencies, skip its creation
and update dependencies with the last command. Activation is optional: these
commands explicitly use the environment's Python executable. CPU execution is
supported; a GPU is not required.

## 3. Configure your local API key

Copy `.env.example` to `.env` in the repository root only if `.env` does not
already exist:

```powershell
if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
notepad .env
```

Enter your own key in that file and save it:

```dotenv
OPENAI_API_KEY=your-own-api-key
```

The app automatically loads `.env`. The current backend uses `gpt-4o-mini` for
generated recommendations. You do not need to paste the key each time you
launch. An existing `OPENAI_API_KEY` environment variable takes precedence over
`.env`. Restart the server after changing the key.

`.env` is ignored by Git. Do not put credentials in `.venv`, tracked source
files, screenshots, or the example template. Each teammate supplies their own
key; it is not included in the repository. API requests can incur usage charges.

Without a key, retrieval and product analysis still work, but recommendations
show cited review excerpts instead of generated answers. API failures also
fall back to excerpts, so a working page alone does not prove a successful model
request. Check your API usage records when verifying live generation.

## 4. Prepare the full knowledge base

The database is not included in Git. Obtain the shared artifacts from the team
or follow [Local setup](LOCAL_SETUP.md). The default UI retrieval path needs:

- `chunks.parquet`: the full 4,450,749-chunk corpus;
- `embedding_shards/`: all 45 matching embedding/ID shard pairs.

The easiest layout is:

```text
electronics_kb/
|-- chunks.parquet
|-- embedding_shards/
|   |-- shard_..._emb.npy
|   `-- shard_..._ids.npy
`-- shopsmart-advisor/
    |-- launch_ui.py
    |-- .env
    `-- .venv/
```

By default, the launcher looks for data in the repository's parent folder.
If your files are elsewhere, set their folder in the same PowerShell terminal
before launching:

```powershell
$env:SHOPSMART_DATA_DIR = 'D:\data\electronics_kb'
```

If you downloaded artifacts directly into the repository root, use:

```powershell
$env:SHOPSMART_DATA_DIR = (Get-Location).Path
```

Individual `CHUNKS_PATH`, `EMBEDDING_SHARDS_PATH` and `CHROMA_PATH` environment
variables override those paths; remove stale overrides if the wrong files are
selected. Set data-path variables in the launching terminal, not just in `.env`:
the launcher checks paths before the app loads `.env`.

The default shard retriever scans all vectors in bounded batches. It uses the
full corpus, rather than a sample, without loading the entire Chroma index into
memory. Only selected relevant evidence is sent to the answer model.

## 5. Start the full application

From the repository root:

```powershell
.\.venv\Scripts\python.exe launch_ui.py --mode full --port 8503
```

Keep the terminal running, then open **http://127.0.0.1:8503/** in Chrome.
The first run may download the BGE embedding and Cardiff sentiment models;
internet access is needed if they are not cached. Wait for initialization and
search indicators to finish. Earlier full-corpus backend search measurements
were approximately 23-25 seconds on the development machine; actual time varies
with hardware and caching, and generation/analysis adds time.

For subsequent sessions, open the repository folder, set the data path again
if you use a non-default location, run the same startup command, and open the URL.
No reindexing or database copy is needed for existing compatible artifacts.
To stop the server, press `Ctrl+C` in its terminal.

### Optional: Chroma retrieval

With a complete `chroma_db/` in the selected data folder and enough RAM:

```powershell
.\.venv\Scripts\python.exe launch_ui.py --mode full --retriever chroma --port 8503
```

Both paths use BGE embeddings over the same corpus. The Chroma index is large;
the default shard path is preferable on the 16 GB development computer.

### Optional: lightweight UI preview

To inspect the interface without the full database:

```powershell
.\.venv\Scripts\python.exe launch_ui.py --mode saved --port 8501
```

Open **http://127.0.0.1:8501/**. Saved mode replays recorded outputs for known
test questions and uses a limited fallback for other inputs. It does not
demonstrate live full-corpus retrieval or live answer generation. Use `--mode
full` on port 8503 for end-to-end testing.

## 6. Test the application

See the [manual UI checklist](UI_TEST_CHECKLIST.md) and the
[step-by-step recording guide](VIDEO_TEST_GUIDE.md).

Start with a fresh browser session and submit:

```text
I need wireless earbuds for long flights, with long battery life and a comfortable fit.
```

Inspect the cited sources, requirement evidence and Why not? findings. Add three
products to Compare, remove a product, export the comparison, switch themes, and
collapse/reopen the sidebar. Open Team from the footer. Comparison and theme
interactions reuse the current search and answer; submitting a new query can
trigger another paid request.

Run the automated tests without full-corpus search or paid model requests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For developers, `scripts/check_full_ui.py` exercises full-data search and the UI
workflow without paid generation. It expects the default parent-folder data
layout, explicitly ignores `.env`, and removes `OPENAI_API_KEY` for that process.
`scripts/check_live_ui.py` also expects that layout and checks five full-data
queries without calling answer generation. These checks can take several minutes.

## Troubleshooting

| Problem | What to do |
|---|---|
| Project environment not found | Create `.venv` in the repository root using step 2. |
| Missing corpus or embedding shards | Check the data layout and set `SHOPSMART_DATA_DIR` in the launching terminal. Git does not include the database. |
| Native dependency preflight fails | Read the local `runtime-check.log`; confirm Python 3.11 and CPU Torch. Do not disable Windows security protections. |
| Port already in use | Open the existing service, stop it in its original terminal, or choose another port and open the matching URL. |
| Page does not open | Keep the terminal running and inspect its error output; use the exact port printed there. |
| Key changes have no effect | Restart the server; check whether a terminal environment variable overrides `.env`. |
| Answer contains only excerpts | Check API credentials, quota and connectivity. A valid key is required for live generated prose. |
| Old page styling | Refresh the browser after updating the code; restart the service if necessary. |

## Integration scope and verification

Full mode uses the collaborator's PCS implementation through
`src/ui/pcs_adapter.py`; provenance is recorded in
`src/generation/UI_INTEGRATION.md`. Saved mode retains historical scores.
Confidence scores represent experimental evidence confidence, not a probability
that a purchase will succeed.

The corpus and shard headers were checked at 4,450,749 rows/vectors. Full-data
search, product analysis and the UI workflow were tested, and a separate real
API-backed browser query returned generated prose with source citations.
All 36 automated tests passed in the project Python 3.11 environment before
this UI handoff on 2026-11-04.
Recent UI fixes were checked with local saved-output previews, which make no
paid generation requests. Browser export download remains a manual acceptance
check. A known generation-quality issue is that a comfort claim can exceed its
supporting evidence; citation presence alone does not establish faithfulness.
This UI work does not reassess collaborators' annotation or report completion.
