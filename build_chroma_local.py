"""Build the local Chroma index from the downloaded embedding shards.

Unlike the shared Colab script, this file has no Google Drive mount or GPU
dependency. DuckDB performs the deterministic corpus sort out of core, so the
4.45M-row parquet file does not need to fit in RAM. The build is resumable.
"""

from __future__ import annotations

import math
from pathlib import Path

import chromadb
import duckdb
import numpy as np


ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = ROOT / "chunks.parquet"
SHARD_DIR = ROOT / "embedding_shards"
CHROMA_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "electronics_kb"
MAX_CHARS = 2000
SHARD_SIZE = 100_000
ADD_BATCH = 5_000
HNSW_BATCH_SIZE = 50_000
HNSW_SYNC_THRESHOLD = 100_000
DUCKDB_MEMORY_LIMIT = "3GB"


def open_duckdb() -> duckdb.DuckDBPyConnection:
    temp_dir = ROOT / "tmp_duckdb"
    temp_dir.mkdir(exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
    connection.execute(f"SET temp_directory = '{temp_dir.as_posix()}'")
    connection.execute("SET preserve_insertion_order = false")
    return connection


def validate_shard_files(total_chunks: int) -> int:
    expected = math.ceil(total_chunks / SHARD_SIZE)
    emb_paths = sorted(SHARD_DIR.glob("shard_*_emb.npy"))
    ids_paths = sorted(SHARD_DIR.glob("shard_*_ids.npy"))
    if len(emb_paths) != expected or len(ids_paths) != expected:
        raise RuntimeError(
            f"Expected {expected} embedding/id shard pairs for {total_chunks:,} chunks; "
            f"found {len(emb_paths)} embedding and {len(ids_paths)} id files."
        )

    total_ids = 0
    for shard_idx in range(expected):
        ids_path = SHARD_DIR / f"shard_{shard_idx:04d}_ids.npy"
        emb_path = SHARD_DIR / f"shard_{shard_idx:04d}_emb.npy"
        ids = np.load(ids_path, mmap_mode="r")
        emb = np.load(emb_path, mmap_mode="r")
        if ids.ndim != 1 or emb.ndim != 2 or len(ids) != len(emb):
            raise RuntimeError(f"Shape mismatch in shard {shard_idx:04d}: ids={ids.shape}, emb={emb.shape}")
        expected_rows = min(SHARD_SIZE, total_chunks - shard_idx * SHARD_SIZE)
        if len(ids) != expected_rows:
            raise RuntimeError(
                f"Unexpected row count in shard {shard_idx:04d}: {len(ids):,} != {expected_rows:,}"
            )
        total_ids += len(ids)
        print(f"Validated shard {shard_idx + 1}/{expected}: {len(ids):,} rows, {emb.shape[1]} dims")

    if total_ids != total_chunks:
        raise RuntimeError(f"Shard rows {total_ids:,} != corpus rows {total_chunks:,}")
    return expected


def metadata_for(row: tuple) -> dict:
    rating = row[5]
    if rating is None or (isinstance(rating, float) and math.isnan(rating)):
        rating = 0.0
    return {
        "parent_asin": str(row[1]),
        "chunk_type": str(row[2]),
        "aspect": str(row[3]),
        "category": str(row[6]),
        "rating": float(rating),
    }


def main() -> None:
    db = open_duckdb()
    parquet_path = CHUNKS_PATH.as_posix().replace("'", "''")
    total_chunks = db.execute(
        f"SELECT count(*) FROM read_parquet('{parquet_path}')"
    ).fetchone()[0]
    print(f"Found {total_chunks:,} chunks in {CHUNKS_PATH.name}")
    n_shards = validate_shard_files(total_chunks)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:batch_size": HNSW_BATCH_SIZE,
            "hnsw:sync_threshold": HNSW_SYNC_THRESHOLD,
        },
    )
    already = collection.count()
    if already > total_chunks:
        raise RuntimeError(f"Chroma has {already:,} rows, more than corpus size {total_chunks:,}")
    print(f"Chroma currently contains {already:,}/{total_chunks:,} chunks")

    # This exactly matches the shared script: cap text at 2000 chars, then sort
    # by capped length and chunk_id (pandas' default null placement is last).
    sorted_query = f"""
        SELECT
            CAST(chunk_id AS VARCHAR) AS chunk_id,
            parent_asin,
            chunk_type,
            aspect,
            substr(text, 1, {MAX_CHARS}) AS text,
            rating,
            category
        FROM read_parquet('{parquet_path}')
        ORDER BY length(substr(text, 1, {MAX_CHARS})) NULLS LAST, chunk_id
        OFFSET {already}
    """
    cursor = db.execute(sorted_query)

    for shard_idx in range(n_shards):
        ids = np.load(SHARD_DIR / f"shard_{shard_idx:04d}_ids.npy")
        shard_start = shard_idx * SHARD_SIZE
        local_start = max(0, already - shard_start)
        if local_start >= len(ids):
            print(f"Skip shard {shard_idx + 1}/{n_shards} (already indexed)")
            continue

        emb = np.load(SHARD_DIR / f"shard_{shard_idx:04d}_emb.npy", mmap_mode="r")
        for i in range(local_start, len(ids), ADD_BATCH):
            end = min(i + ADD_BATCH, len(ids))
            batch_ids = ids[i:end].tolist()
            rows = cursor.fetchmany(end - i)
            if len(rows) != end - i:
                raise RuntimeError(
                    f"Sorted corpus ended early in shard {shard_idx:04d}: got {len(rows)} rows"
                )
            sorted_ids = [str(row[0]) for row in rows]
            if sorted_ids != batch_ids:
                mismatch = next(
                    (j for j, (left, right) in enumerate(zip(sorted_ids, batch_ids)) if left != right),
                    0,
                )
                raise RuntimeError(
                    f"Chunk ordering mismatch in shard {shard_idx:04d} at local row {i + mismatch:,}: "
                    f"parquet={sorted_ids[mismatch]!r}, shard={batch_ids[mismatch]!r}"
                )
            collection.add(
                ids=batch_ids,
                embeddings=emb[i:end].astype(np.float32),
                documents=[row[4] for row in rows],
                metadatas=[metadata_for(row) for row in rows],
            )
        already = collection.count()
        print(f"Indexed shard {shard_idx + 1}/{n_shards}: {already:,}/{total_chunks:,}", flush=True)

    final_count = collection.count()
    if final_count != total_chunks:
        raise RuntimeError(f"Final Chroma count {final_count:,} != corpus count {total_chunks:,}")
    print(f"Done. Chroma contains {final_count:,} chunks at {CHROMA_PATH}")


if __name__ == "__main__":
    main()
