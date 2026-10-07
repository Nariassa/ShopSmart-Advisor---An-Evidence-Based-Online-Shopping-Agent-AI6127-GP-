"""
STEP 3b — RESUMABLE, SHARDED EMBEDDING (Drive-backed)

Why this exists: a Colab runtime reset wiped /content, including the
Chroma index that took hours to build. This version writes embeddings in
shards to Google Drive (.npy + ids), so a reset only costs the shard that
was in progress. Re-running the script skips every shard already on Drive.

Flow:
  1. Mount Drive, load chunks.parquet (from Drive)
  2. Apply the same length-cap + sort used before (order must be identical
     between runs, so shard boundaries stay stable)
  3. Embed shard by shard -> save to Drive
  4. Build Chroma (local disk) from the saved shards — no GPU needed

Run in Colab with a GPU runtime.
"""

import os
import numpy as np
import pandas as pd
from google.colab import drive

drive.mount("/content/drive")

# ---- paths (edit DRIVE_DIR if you keep files elsewhere) ----
DRIVE_DIR = "/content/drive/MyDrive/electronics_kb"
CHUNKS_PATH = f"{DRIVE_DIR}/chunks.parquet"
SHARD_DIR = f"{DRIVE_DIR}/embedding_shards"
os.makedirs(SHARD_DIR, exist_ok=True)

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
MAX_CHARS = 2000
SHARD_SIZE = 100_000     # chunks per shard: ~5 min of work at ~312 chunks/sec
ENCODE_BATCH = 256

# ------------------------------------------------------------
# Load + deterministic ordering (must match on every run)
# ------------------------------------------------------------
chunks = pd.read_parquet(CHUNKS_PATH)
chunks["chunk_id"] = chunks["chunk_id"].astype(str)
chunks["text"] = chunks["text"].apply(lambda t: t[:MAX_CHARS] if isinstance(t, str) else t)
chunks["text_len"] = chunks["text"].str.len()
# stable sort + chunk_id tiebreak so ordering is identical across sessions
chunks = chunks.sort_values(["text_len", "chunk_id"], kind="stable").reset_index(drop=True)
chunks = chunks.drop(columns=["text_len"])

n_shards = (len(chunks) + SHARD_SIZE - 1) // SHARD_SIZE
print(f"{len(chunks)} chunks -> {n_shards} shards of up to {SHARD_SIZE}")

# ------------------------------------------------------------
# Embed shard by shard, skipping any shard already saved
# ------------------------------------------------------------
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")
print(f"Embedding model device: {embed_model.device}")  # must say cuda:0

for shard_idx in range(n_shards):
    emb_path = f"{SHARD_DIR}/shard_{shard_idx:04d}_emb.npy"
    ids_path = f"{SHARD_DIR}/shard_{shard_idx:04d}_ids.npy"
    if os.path.exists(emb_path) and os.path.exists(ids_path):
        continue  # already done in an earlier session

    shard = chunks.iloc[shard_idx * SHARD_SIZE:(shard_idx + 1) * SHARD_SIZE]
    embeddings = embed_model.encode(
        shard["text"].tolist(),
        batch_size=ENCODE_BATCH,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float16)  # halves Drive storage; negligible effect on retrieval

    # write ids first, embeddings last: a shard only counts as done when both exist
    np.save(ids_path, shard["chunk_id"].to_numpy(dtype=str))
    np.save(emb_path, embeddings)
    print(f"Saved shard {shard_idx + 1}/{n_shards}")

print("All shards embedded.")

# ------------------------------------------------------------
# Build Chroma locally from the saved shards (fast; no GPU work)
# ------------------------------------------------------------
import chromadb

CHROMA_PATH = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    "electronics_kb", metadata={"hnsw:space": "cosine"}
)

already = collection.count()
print(f"Chroma currently holds {already} chunks")

meta_lookup = chunks.set_index("chunk_id")
ADD_BATCH = 5000  # Chroma limits the max batch size per add() call

for shard_idx in range(n_shards):
    start = shard_idx * SHARD_SIZE
    end = min(start + SHARD_SIZE, len(chunks))
    if end <= already:
        continue  # this shard is already in Chroma

    ids = np.load(f"{SHARD_DIR}/shard_{shard_idx:04d}_ids.npy")
    emb = np.load(f"{SHARD_DIR}/shard_{shard_idx:04d}_emb.npy").astype(np.float32)
    rows = meta_lookup.loc[ids]

    for i in range(0, len(ids), ADD_BATCH):
        sl = slice(i, i + ADD_BATCH)
        sub = rows.iloc[sl]
        collection.add(
            ids=ids[sl].tolist(),
            embeddings=emb[sl].tolist(),
            documents=sub["text"].tolist(),
            metadatas=[
                {
                    "parent_asin": r.parent_asin,
                    "chunk_type": r.chunk_type,
                    "aspect": r.aspect,
                    "category": str(r.category),
                    "rating": float(r.rating) if pd.notna(r.rating) else 0.0,
                }
                for r in sub.itertuples()
            ],
        )
    print(f"Chroma: loaded shard {shard_idx + 1}/{n_shards}")

print(f"Done. Chroma holds {collection.count()} chunks.")
