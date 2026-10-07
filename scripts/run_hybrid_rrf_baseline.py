"""Fuse saved BM25 and dense pilot runs with Reciprocal Rank Fusion."""

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

from retrieval.hybrid_retriever import reciprocal_rank_fusion  # noqa: E402
from run_bm25_baseline import (  # noqa: E402
    calculate_metrics,
    relevance_counts,
    write_jsonl,
)


QUERY_FIELDS = (
    "query_id",
    "query",
    "source_title",
    "suite",
    "target_parent_asin",
    "target_aspect",
    "category",
    "label_source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument("--saved-depth", type=int, default=20)
    parser.add_argument(
        "--bm25-results-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "bm25_pilot_results.jsonl",
    )
    parser.add_argument(
        "--dense-results-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "dense_pilot_results.jsonl",
    )
    parser.add_argument(
        "--bm25-metrics-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "bm25_pilot_metrics.json",
    )
    parser.add_argument(
        "--dense-metrics-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "dense_pilot_metrics.json",
    )
    parser.add_argument(
        "--chunks-path", type=Path, default=PROJECT_ROOT / "chunks.parquet"
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "hybrid_rrf_pilot_results.jsonl",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "hybrid_rrf_pilot_metrics.json",
    )
    parser.add_argument(
        "--metrics-csv-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "hybrid_rrf_pilot_metrics.csv",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index_records(records: list[dict], source_name: str) -> dict[str, dict]:
    indexed = {}
    for record in records:
        query_id = record["query_id"]
        if query_id in indexed:
            raise ValueError(f"Duplicate {source_name} query_id: {query_id}")
        indexed[query_id] = record
    return indexed


def validate_pair(bm25_record: dict, dense_record: dict, candidate_depth: int) -> None:
    for field in QUERY_FIELDS:
        if bm25_record.get(field) != dense_record.get(field):
            raise ValueError(
                f"Query mismatch for {bm25_record['query_id']} in field {field}"
            )
    if len(bm25_record["results"]) < candidate_depth:
        raise ValueError("BM25 saved depth is smaller than candidate-depth.")
    if len(dense_record["results"]) < candidate_depth:
        raise ValueError("Dense saved depth is smaller than candidate-depth.")


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
    if args.saved_depth < 10:
        raise ValueError("saved-depth must be at least 10 for the pilot metrics.")
    if args.saved_depth > args.candidate_depth * 2:
        raise ValueError("saved-depth cannot exceed the fused candidate pool.")

    bm25_records = read_jsonl(args.bm25_results_path)
    dense_records = read_jsonl(args.dense_results_path)
    bm25_by_id = index_records(bm25_records, "BM25")
    dense_by_id = index_records(dense_records, "dense")
    if set(bm25_by_id) != set(dense_by_id):
        raise ValueError("BM25 and dense runs do not contain the same query IDs.")

    run_records = []
    timing_rows = []
    for query_id in bm25_by_id:
        bm25_record = bm25_by_id[query_id]
        dense_record = dense_by_id[query_id]
        validate_pair(bm25_record, dense_record, args.candidate_depth)

        fusion_started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [
                ("bm25", bm25_record["results"][: args.candidate_depth]),
                ("dense", dense_record["results"][: args.candidate_depth]),
            ],
            rrf_k=args.rrf_k,
            top_k=args.saved_depth,
        )
        fusion_seconds = time.perf_counter() - fusion_started
        bm25_seconds = float(bm25_record["latency_seconds"])
        dense_seconds = float(dense_record["latency_seconds"])
        sequential_seconds = bm25_seconds + dense_seconds + fusion_seconds
        parallel_estimate_seconds = max(bm25_seconds, dense_seconds) + fusion_seconds

        run_record = {
            **{field: bm25_record.get(field) for field in QUERY_FIELDS},
            "latency_seconds": sequential_seconds,
            "bm25_seconds": bm25_seconds,
            "dense_seconds": dense_seconds,
            "fusion_seconds": fusion_seconds,
            "parallel_estimate_seconds": parallel_estimate_seconds,
            "candidate_depth_per_source": args.candidate_depth,
            "rrf_k": args.rrf_k,
            "results": [result.to_dict() for result in fused],
        }
        mark_relevance(run_record)
        run_records.append(run_record)
        timing_rows.append(
            {
                "query_id": query_id,
                "bm25_seconds": bm25_seconds,
                "dense_seconds": dense_seconds,
                "fusion_seconds": fusion_seconds,
                "sequential_seconds": sequential_seconds,
                "parallel_estimate_seconds": parallel_estimate_seconds,
            }
        )
        print(
            f"{query_id} fused in {fusion_seconds:.6f} seconds; "
            f"sequential estimate={sequential_seconds:.3f} seconds",
            flush=True,
        )

    queries = [
        {field: record.get(field) for field in QUERY_FIELDS}
        for record in run_records
    ]
    counts = relevance_counts(args.chunks_path, queries)
    bm25_metrics = read_json(args.bm25_metrics_path)
    dense_metrics = read_json(args.dense_metrics_path)
    combined_load_seconds = (
        float(bm25_metrics["run"]["index_load_seconds"])
        + float(dense_metrics["run"]["index_load_seconds"])
    )
    metrics, per_query = calculate_metrics(
        run_records, counts, combined_load_seconds
    )
    metrics["run"].update(
        {
            "retriever": "hybrid_rrf",
            "rrf_k": args.rrf_k,
            "candidate_depth_per_source": args.candidate_depth,
            "fusion_mode": "offline_replay_of_saved_rankings",
            "mean_fusion_seconds": statistics.fmean(
                row["fusion_seconds"] for row in timing_rows
            ),
            "mean_sequential_latency_seconds": statistics.fmean(
                row["sequential_seconds"] for row in timing_rows
            ),
            "mean_parallel_estimate_seconds": statistics.fmean(
                row["parallel_estimate_seconds"] for row in timing_rows
            ),
            "bm25_cold_load_seconds": bm25_metrics["run"]["index_load_seconds"],
            "dense_cold_load_seconds": dense_metrics["run"]["index_load_seconds"],
        }
    )
    metrics["limitations"].extend(
        [
            "Fusion used the top 20 saved candidates from each source.",
            "End-to-end latency is reconstructed from saved source runs rather than rerun in one process.",
            "The parallel latency value is an estimate and may be infeasible on a 16 GB machine.",
        ]
    )

    write_jsonl(args.results_path, run_records)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    per_query_frame = pd.DataFrame(per_query)
    timing_frame = pd.DataFrame(timing_rows)
    per_query_frame.merge(timing_frame, on="query_id").to_csv(
        args.metrics_csv_path, index=False
    )

    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved results to {args.results_path}", flush=True)
    print(f"Saved metrics to {args.metrics_path}", flush=True)


if __name__ == "__main__":
    main()
