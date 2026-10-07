"""Run the frozen retrieval baselines on the 20-query formal evaluation set.

The formal query set has no automatic relevance labels. This script stores
ranked candidates and timing metadata only. Final effectiveness metrics must be
computed after human qrels are complete.

Each stage is resumable. Completed query records are saved after every query.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Iterable

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval import BM25Retriever  # noqa: E402
from retrieval.cross_encoder_reranker import CrossEncoderReranker  # noqa: E402
from retrieval.dense_retriever import DenseRetriever  # noqa: E402
from retrieval.hybrid_retriever import reciprocal_rank_fusion  # noqa: E402


FORMAL_QUERY_FIELDS = (
    "query_id",
    "query",
    "intent",
    "target_aspects",
    "target_category",
    "difficulty",
)
EXPECTED_QUERY_COUNT = 20
CORPUS_SIZE = 4_450_749


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("bm25", "dense", "hybrid", "rerank", "verify"),
        required=True,
    )
    parser.add_argument("--retrieval-depth", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "formal_queries_v1.jsonl",
    )
    parser.add_argument(
        "--chunks-path", type=Path, default=PROJECT_ROOT / "chunks.parquet"
    )
    parser.add_argument(
        "--bm25-index-path", type=Path, default=PROJECT_ROOT / "bm25_index.pkl"
    )
    parser.add_argument(
        "--chroma-path", type=Path, default=PROJECT_ROOT / "chroma_db"
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=PROJECT_ROOT / "model_cache" / "sentence_transformers",
    )
    parser.add_argument(
        "--reranker-cache",
        type=Path,
        default=PROJECT_ROOT / "model_cache" / "cross_encoder",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "results"
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_queries(path: Path) -> list[dict]:
    queries = read_jsonl(path)
    if len(queries) != EXPECTED_QUERY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_QUERY_COUNT} formal queries, found {len(queries)}."
        )
    query_ids = [record.get("query_id") for record in queries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Formal query IDs must be unique.")
    for record in queries:
        missing = [field for field in FORMAL_QUERY_FIELDS if field not in record]
        if missing:
            raise ValueError(
                f"Formal query {record.get('query_id')} is missing: {missing}"
            )
    return queries


def query_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_resumable_records(path: Path, queries: list[dict]) -> list[dict]:
    if not path.is_file():
        return []
    records = read_jsonl(path)
    expected_ids = [query["query_id"] for query in queries]
    record_ids = [record["query_id"] for record in records]
    if record_ids != expected_ids[: len(record_ids)]:
        raise ValueError(f"Existing records are not a valid query prefix: {path}")
    return records


def enrich_records(chunks_path: Path, records: list[dict]) -> None:
    chunk_ids = sorted(
        {
            str(result["chunk_id"])
            for record in records
            for result in record["results"]
        }
    )
    ids_frame = pd.DataFrame({"chunk_id": chunk_ids})
    con = duckdb.connect()
    con.register("requested_ids", ids_frame)
    rows = con.execute(
        """
        SELECT
            c.chunk_id,
            c.parent_asin,
            c.chunk_type,
            c.aspect,
            c.category,
            c.rating,
            c.text
        FROM read_parquet(?) AS c
        INNER JOIN requested_ids AS r USING (chunk_id)
        """,
        [str(chunks_path)],
    ).fetchall()
    con.close()
    metadata = {
        str(row[0]): {
            "parent_asin": row[1],
            "chunk_type": row[2],
            "aspect": row[3],
            "category": row[4],
            "rating": row[5],
            "text": row[6],
        }
        for row in rows
    }
    missing = sorted(set(chunk_ids) - set(metadata))
    if missing:
        raise ValueError(f"Missing metadata for {len(missing)} candidates.")
    for record in records:
        for result in record["results"]:
            result.update(metadata[str(result["chunk_id"])])


def latency_summary(records: list[dict], latency_field: str) -> dict:
    values = [float(record[latency_field]) for record in records]
    ordered = sorted(values)
    p95_index = max(0, int((0.95 * len(ordered) + 0.999999)) - 1)
    return {
        "query_count": len(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": ordered[p95_index],
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def stage_paths(results_dir: Path) -> dict[str, Path]:
    return {
        "bm25": results_dir / "formal_bm25_results.jsonl",
        "dense": results_dir / "formal_dense_results.jsonl",
        "hybrid": results_dir / "formal_hybrid_rrf_results.jsonl",
        "rerank": results_dir / "formal_cross_encoder_results.jsonl",
    }


def run_bm25(args: argparse.Namespace, queries: list[dict]) -> None:
    paths = stage_paths(args.results_dir)
    output_path = paths["bm25"]
    records = load_resumable_records(output_path, queries)
    completed_ids = {record["query_id"] for record in records}

    retriever = BM25Retriever(args.bm25_index_path).load()
    if retriever.corpus_size != CORPUS_SIZE:
        raise ValueError(f"Unexpected BM25 corpus size: {retriever.corpus_size:,}")
    print(
        f"Loaded BM25 index with {retriever.corpus_size:,} documents in "
        f"{retriever.load_seconds:.2f} seconds",
        flush=True,
    )

    for index, query_record in enumerate(queries, start=1):
        if query_record["query_id"] in completed_ids:
            print(
                f"[{index:02d}/{len(queries):02d}] {query_record['query_id']} "
                "already complete",
                flush=True,
            )
            continue
        results, elapsed = retriever.search(
            query_record["query"], top_k=args.retrieval_depth
        )
        records.append(
            {
                **query_record,
                "latency_seconds": elapsed,
                "retrieval_depth": args.retrieval_depth,
                "results": [result.to_dict() for result in results],
            }
        )
        write_jsonl_atomic(output_path, records)
        print(
            f"[{index:02d}/{len(queries):02d}] {query_record['query_id']} "
            f"completed in {elapsed:.2f} seconds",
            flush=True,
        )

    enrich_records(args.chunks_path, records)
    write_jsonl_atomic(output_path, records)
    write_json(
        args.results_dir / "formal_bm25_run.json",
        {
            "stage": "bm25",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query_digest_sha256": query_digest(args.queries_path),
            "corpus_size": retriever.corpus_size,
            "index_load_seconds": retriever.load_seconds,
            "retrieval_depth": args.retrieval_depth,
            "latency": latency_summary(records, "latency_seconds"),
            "evaluation_status": "awaiting_human_qrels",
        },
    )


def run_dense(args: argparse.Namespace, queries: list[dict]) -> None:
    paths = stage_paths(args.results_dir)
    output_path = paths["dense"]
    records = load_resumable_records(output_path, queries)
    completed_ids = {record["query_id"] for record in records}

    retriever = DenseRetriever(
        chroma_path=args.chroma_path,
        cache_folder=args.embedding_cache,
    ).load()
    if retriever.corpus_size != CORPUS_SIZE:
        raise ValueError(f"Unexpected Chroma corpus size: {retriever.corpus_size:,}")
    _, warmup = retriever.search("electronics product", top_k=1)
    print(
        f"Loaded dense retriever in "
        f"{retriever.model_load_seconds + retriever.collection_open_seconds:.2f} "
        f"seconds; warm-up={warmup.total_seconds:.2f} seconds",
        flush=True,
    )

    for index, query_record in enumerate(queries, start=1):
        if query_record["query_id"] in completed_ids:
            print(
                f"[{index:02d}/{len(queries):02d}] {query_record['query_id']} "
                "already complete",
                flush=True,
            )
            continue
        results, timing = retriever.search(
            query_record["query"], top_k=args.retrieval_depth
        )
        records.append(
            {
                **query_record,
                "latency_seconds": timing.total_seconds,
                "encoding_seconds": timing.encoding_seconds,
                "search_seconds": timing.search_seconds,
                "retrieval_depth": args.retrieval_depth,
                "results": [result.to_dict() for result in results],
            }
        )
        write_jsonl_atomic(output_path, records)
        print(
            f"[{index:02d}/{len(queries):02d}] {query_record['query_id']} "
            f"completed in {timing.total_seconds:.3f} seconds",
            flush=True,
        )

    enrich_records(args.chunks_path, records)
    write_jsonl_atomic(output_path, records)
    write_json(
        args.results_dir / "formal_dense_run.json",
        {
            "stage": "dense",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query_digest_sha256": query_digest(args.queries_path),
            "corpus_size": retriever.corpus_size,
            "model": retriever.model_name,
            "model_load_seconds": retriever.model_load_seconds,
            "collection_open_seconds": retriever.collection_open_seconds,
            "warmup_seconds": warmup.total_seconds,
            "retrieval_depth": args.retrieval_depth,
            "latency": latency_summary(records, "latency_seconds"),
            "evaluation_status": "awaiting_human_qrels",
        },
    )


def validate_complete(records: list[dict], queries: list[dict], source: str) -> None:
    expected_ids = [query["query_id"] for query in queries]
    record_ids = [record["query_id"] for record in records]
    if record_ids != expected_ids:
        raise ValueError(f"{source} results do not match the formal query set.")


def run_hybrid(args: argparse.Namespace, queries: list[dict]) -> None:
    paths = stage_paths(args.results_dir)
    bm25_records = read_jsonl(paths["bm25"])
    dense_records = read_jsonl(paths["dense"])
    validate_complete(bm25_records, queries, "BM25")
    validate_complete(dense_records, queries, "Dense")

    records = []
    for index, (query_record, bm25_record, dense_record) in enumerate(
        zip(queries, bm25_records, dense_records), start=1
    ):
        started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [
                ("bm25", bm25_record["results"][: args.retrieval_depth]),
                ("dense", dense_record["results"][: args.retrieval_depth]),
            ],
            rrf_k=args.rrf_k,
            top_k=args.retrieval_depth,
        )
        fusion_seconds = time.perf_counter() - started
        sequential_seconds = (
            float(bm25_record["latency_seconds"])
            + float(dense_record["latency_seconds"])
            + fusion_seconds
        )
        parallel_seconds = (
            max(
                float(bm25_record["latency_seconds"]),
                float(dense_record["latency_seconds"]),
            )
            + fusion_seconds
        )
        records.append(
            {
                **query_record,
                "latency_seconds": sequential_seconds,
                "parallel_estimate_seconds": parallel_seconds,
                "fusion_seconds": fusion_seconds,
                "candidate_depth_per_source": args.retrieval_depth,
                "rrf_k": args.rrf_k,
                "results": [result.to_dict() for result in fused],
            }
        )
        print(
            f"[{index:02d}/{len(queries):02d}] {query_record['query_id']} "
            f"fused in {fusion_seconds:.6f} seconds",
            flush=True,
        )

    write_jsonl_atomic(paths["hybrid"], records)
    write_json(
        args.results_dir / "formal_hybrid_rrf_run.json",
        {
            "stage": "hybrid_rrf",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query_digest_sha256": query_digest(args.queries_path),
            "rrf_k": args.rrf_k,
            "candidate_depth_per_source": args.retrieval_depth,
            "saved_depth": args.retrieval_depth,
            "latency": latency_summary(records, "latency_seconds"),
            "evaluation_status": "awaiting_human_qrels",
        },
    )


def run_rerank(args: argparse.Namespace, queries: list[dict]) -> None:
    paths = stage_paths(args.results_dir)
    hybrid_records = read_jsonl(paths["hybrid"])
    validate_complete(hybrid_records, queries, "Hybrid")

    reranker = CrossEncoderReranker(
        cache_folder=args.reranker_cache,
        max_length=args.max_length,
        device="cpu",
    ).load()
    warmup_candidate = dict(hybrid_records[0]["results"][0])
    _, warmup_seconds = reranker.rerank(
        hybrid_records[0]["query"], [warmup_candidate], batch_size=1
    )
    print(
        f"Loaded cross-encoder in {reranker.load_seconds:.2f} seconds; "
        f"warm-up={warmup_seconds:.3f} seconds",
        flush=True,
    )

    records = []
    for index, (query_record, hybrid_record) in enumerate(
        zip(queries, hybrid_records), start=1
    ):
        reranked, rerank_seconds = reranker.rerank(
            query_record["query"],
            hybrid_record["results"][: args.retrieval_depth],
            top_k=args.retrieval_depth,
            batch_size=args.batch_size,
        )
        total_seconds = float(hybrid_record["latency_seconds"]) + rerank_seconds
        records.append(
            {
                **query_record,
                "latency_seconds": total_seconds,
                "retrieval_seconds": hybrid_record["latency_seconds"],
                "rerank_seconds": rerank_seconds,
                "parallel_estimate_seconds": float(
                    hybrid_record["parallel_estimate_seconds"]
                )
                + rerank_seconds,
                "candidate_depth": args.retrieval_depth,
                "results": [result.to_dict() for result in reranked],
            }
        )
        print(
            f"[{index:02d}/{len(queries):02d}] {query_record['query_id']} "
            f"reranked in {rerank_seconds:.3f} seconds",
            flush=True,
        )

    write_jsonl_atomic(paths["rerank"], records)
    write_json(
        args.results_dir / "formal_cross_encoder_run.json",
        {
            "stage": "hybrid_rrf_cross_encoder",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query_digest_sha256": query_digest(args.queries_path),
            "model": reranker.model_name,
            "candidate_depth": args.retrieval_depth,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "model_load_seconds": reranker.load_seconds,
            "warmup_seconds": warmup_seconds,
            "latency": latency_summary(records, "latency_seconds"),
            "mean_rerank_seconds": statistics.fmean(
                float(record["rerank_seconds"]) for record in records
            ),
            "evaluation_status": "awaiting_human_qrels",
        },
    )


def verify(args: argparse.Namespace, queries: list[dict]) -> None:
    paths = stage_paths(args.results_dir)
    summary = {}
    for stage, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {stage} results: {path}")
        records = read_jsonl(path)
        validate_complete(records, queries, stage)
        result_counts = [len(record["results"]) for record in records]
        if result_counts != [args.retrieval_depth] * len(queries):
            raise ValueError(f"Unexpected result depth in {stage}.")
        duplicate_queries = []
        for record in records:
            chunk_ids = [result["chunk_id"] for result in record["results"]]
            if len(set(chunk_ids)) != len(chunk_ids):
                duplicate_queries.append(record["query_id"])
            if any(not result.get("text") for result in record["results"]):
                raise ValueError(
                    f"Missing candidate text in {stage}/{record['query_id']}."
                )
        if duplicate_queries:
            raise ValueError(
                f"Duplicate candidate IDs in {stage}: {duplicate_queries}"
            )
        summary[stage] = {
            "query_count": len(records),
            "result_depth": args.retrieval_depth,
            "unique_candidate_pairs": sum(result_counts),
        }

    manifest = {
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query_path": (
            args.queries_path.relative_to(PROJECT_ROOT).as_posix()
            if args.queries_path.is_relative_to(PROJECT_ROOT)
            else str(args.queries_path)
        ),
        "query_digest_sha256": query_digest(args.queries_path),
        "query_count": len(queries),
        "corpus_size": CORPUS_SIZE,
        "frozen_configuration": {
            "bm25_tokenization": "lowercase_whitespace_split",
            "dense_model": "BAAI/bge-base-en-v1.5",
            "rrf_k": args.rrf_k,
            "candidate_depth_per_source": args.retrieval_depth,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "cross_encoder_depth": args.retrieval_depth,
        },
        "stages": summary,
        "evaluation_status": "awaiting_human_qrels",
    }
    write_json(args.results_dir / "formal_retrieval_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.retrieval_depth < 1:
        raise ValueError("retrieval-depth must be at least 1.")
    queries = load_queries(args.queries_path)
    if args.stage == "bm25":
        run_bm25(args, queries)
    elif args.stage == "dense":
        run_dense(args, queries)
    elif args.stage == "hybrid":
        run_hybrid(args, queries)
    elif args.stage == "rerank":
        run_rerank(args, queries)
    else:
        verify(args, queries)


if __name__ == "__main__":
    main()
