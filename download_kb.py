"""Download the shared electronics KB files from Google Drive.

The script is resumable: existing non-empty files are skipped, and interrupted
gdown transfers are resumed when possible.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import gdown


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "download_manifest.json"


def download_one(file_id: str, destination: Path, retries: int = 3) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"SKIP {destination.name} ({destination.stat().st_size:,} bytes)", flush=True)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        print(f"GET  {destination.name} (attempt {attempt}/{retries})", flush=True)
        try:
            result = gdown.download(
                id=file_id,
                output=str(destination),
                quiet=False,
                resume=True,
            )
            if result and destination.exists() and destination.stat().st_size > 0:
                print(f"DONE {destination.name} ({destination.stat().st_size:,} bytes)", flush=True)
                return
        except Exception as exc:  # keep the batch resumable
            print(f"WARN {destination.name}: {exc}", file=sys.stderr, flush=True)
        if attempt < retries:
            time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to download {destination.name}")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    # Small code/docs first, then the large corpus/index files.
    priority = {
        "HANDOFF_README.md": 0,
        "config.py": 1,
        "electronics_kb.py": 2,
        "step3b_embed_sharded.py": 3,
        "chunks.parquet": 4,
        "bm25_index.pkl": 5,
    }
    files = sorted(manifest["files"], key=lambda item: priority[item["name"]])
    for item in files:
        download_one(item["id"], ROOT / item["name"])

    for item in manifest["embedding_shards"]:
        download_one(item["id"], ROOT / "embedding_shards" / item["name"])

    print("All knowledge-base files are present.", flush=True)


if __name__ == "__main__":
    main()
