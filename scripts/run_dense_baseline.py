"""Run the dense-retrieval baseline on the shared BM25 pilot queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.dense_retriever import DenseRetriever  # noqa: E402
from run_bm25_baseline import (  # noqa: E402
    calculate_metrics,
    relevance_counts,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-depth", type=int, default=100)
    parser.add_argument("--saved-depth", type=int, default=20)
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "bm25_pilot_queries.jsonl",
    )
    parser.add_argument(
        "--chunks-path", type=Path, default=PROJECT_ROOT / "chunks.parquet"
    )
    parser.add_argument(
        "--chroma-path", type=Path, default=PROJECT_ROOT / "chroma_db"
    )
    parser.add_argument(
        "--model-name", default="BAAI/bge-base-en-v1.5"
    )
    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=PROJECT_ROOT / "model_cache" / "sentence_transformers",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "dense_pilot_results.jsonl",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "dense_pilot_metrics.json",
    )
    parser.add_argument(
        "--metrics-csv-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "dense_pilot_metrics.csv",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mark_relevance(record: dict) -> None:
    for result in record["results"]:
        if record["suite"] == "known_item_short_title":
            result["relevant"] = (
                result.get("parent_asin") == record["target_parent_asin"]
            )
        else:
            result["relevant"] = result.get("aspect") == record["target_aspect"]


def main() -> None:
    args = parse_args()
    if args.saved_depth > args.retrieval_depth:
        raise ValueError("saved-depth cannot exceed retrieval-depth")
    queries = read_jsonl(args.queries_path)
    if not queries:
        raise ValueError(f"No queries found in {args.queries_path}")
    print(f"Loaded {len(queries)} shared pilot queries", flush=True)

    retriever = DenseRetriever(
        chroma_path=args.chroma_path,
        model_name=args.model_name,
        cache_folder=args.cache_folder,
    ).load()
    print(
        f"Loaded model in {retriever.model_load_seconds:.2f} seconds and opened "
        f"the collection in {retriever.collection_open_seconds:.2f} seconds",
        flush=True,
    )
    count_started = time.perf_counter()
    corpus_size = retriever.corpus_size
    collection_count_seconds = time.perf_counter() - count_started
    if corpus_size != 4_450_749:
        raise ValueError(f"Unexpected Chroma count: {corpus_size:,}")
    print(
        f"Validated {corpus_size:,} indexed chunks in "
        f"{collection_count_seconds:.2f} seconds",
        flush=True,
    )

    _, warmup_timing = retriever.search("electronics product", top_k=1)
    print(
        f"Completed index warm-up in {warmup_timing.total_seconds:.2f} seconds",
        flush=True,
    )

    run_records = []
    timing_rows = []
    for index, query_record in enumerate(queries, start=1):
        results, timing = retriever.search(
            query_record["query"], top_k=args.retrieval_depth
        )
        run_record = {
            **query_record,
            "latency_seconds": timing.total_seconds,
            "encoding_seconds": timing.encoding_seconds,
            "search_seconds": timing.search_seconds,
            "retrieval_depth": args.retrieval_depth,
            "results": [result.to_dict() for result in results],
        }
        mark_relevance(run_record)
        run_records.append(run_record)
        timing_rows.append(timing)
        print(
            f"[{index:02d}/{len(queries):02d}] {query_record['query_id']} "
            f"completed in {timing.total_seconds:.3f} seconds "
            f"(encode={timing.encoding_seconds:.3f}, search={timing.search_seconds:.3f})",
            flush=True,
        )

    counts = relevance_counts(args.chunks_path, queries)
    metrics, per_query = calculate_metrics(
        run_records,
        counts,
        retriever.model_load_seconds
        + retriever.collection_open_seconds
        + collection_count_seconds,
    )
    metrics["run"].update(
        {
            "retriever": "dense",
            "embedding_model": args.model_name,
            "chroma_collection": retriever.collection_name,
            "model_load_seconds": retriever.model_load_seconds,
            "collection_open_seconds": retriever.collection_open_seconds,
            "collection_count_seconds": collection_count_seconds,
            "index_warmup_seconds": warmup_timing.total_seconds,
            "mean_encoding_seconds": statistics.fmean(
                row.encoding_seconds for row in timing_rows
            ),
            "mean_vector_search_seconds": statistics.fmean(
                row.search_seconds for row in timing_rows
            ),
        }
    )

    saved_records = []
    for record in run_records:
        saved = dict(record)
        saved["results"] = record["results"][: args.saved_depth]
        saved_records.append(saved)
    write_jsonl(args.results_path, saved_records)

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    per_query_frame = pd.DataFrame(per_query)
    timing_frame = pd.DataFrame(
        {
            "query_id": [record["query_id"] for record in run_records],
            "encoding_seconds": [row.encoding_seconds for row in timing_rows],
            "search_seconds": [row.search_seconds for row in timing_rows],
        }
    )
    per_query_frame.merge(timing_frame, on="query_id").to_csv(
        args.metrics_csv_path, index=False
    )

    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved results to {args.results_path}", flush=True)
    print(f"Saved metrics to {args.metrics_path}", flush=True)


if __name__ == "__main__":
    main()
