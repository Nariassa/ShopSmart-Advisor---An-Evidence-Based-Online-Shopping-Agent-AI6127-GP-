"""Rerank the saved hybrid pilot candidates with a cross-encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.cross_encoder_reranker import CrossEncoderReranker  # noqa: E402
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
    parser.add_argument(
        "--model-name", default="cross-encoder/ms-marco-MiniLM-L6-v2"
    )
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument("--saved-depth", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--hybrid-results-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "hybrid_rrf_pilot_results.jsonl",
    )
    parser.add_argument(
        "--hybrid-metrics-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "hybrid_rrf_pilot_metrics.json",
    )
    parser.add_argument(
        "--chunks-path", type=Path, default=PROJECT_ROOT / "chunks.parquet"
    )
    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=PROJECT_ROOT / "model_cache" / "cross_encoder",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "cross_encoder_pilot_results.jsonl",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "cross_encoder_pilot_metrics.json",
    )
    parser.add_argument(
        "--metrics-csv-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "cross_encoder_pilot_metrics.csv",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_candidate_texts(chunks_path: Path, records: list[dict]) -> tuple[dict, float]:
    chunk_ids = sorted(
        {
            result["chunk_id"]
            for record in records
            for result in record["results"]
        }
    )
    ids_frame = pd.DataFrame({"chunk_id": chunk_ids})
    started = time.perf_counter()
    con = duckdb.connect()
    con.register("requested_ids", ids_frame)
    rows = con.execute(
        """
        SELECT c.chunk_id, c.text
        FROM read_parquet(?) AS c
        INNER JOIN requested_ids AS r USING (chunk_id)
        """,
        [str(chunks_path)],
    ).fetchall()
    con.close()
    elapsed = time.perf_counter() - started
    texts = {str(chunk_id): str(text) for chunk_id, text in rows}
    missing = sorted(set(chunk_ids) - set(texts))
    if missing:
        raise ValueError(f"Missing text for {len(missing)} candidates: {missing[:3]}")
    return texts, elapsed


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
    if args.saved_depth > args.candidate_depth:
        raise ValueError("saved-depth cannot exceed candidate-depth.")
    if args.saved_depth < 10:
        raise ValueError("saved-depth must be at least 10 for the pilot metrics.")

    hybrid_records = read_jsonl(args.hybrid_results_path)
    if not hybrid_records:
        raise ValueError(f"No hybrid records found in {args.hybrid_results_path}")
    for record in hybrid_records:
        if len(record["results"]) < args.candidate_depth:
            raise ValueError(
                f"Hybrid saved depth is too small for {record['query_id']}"
            )

    texts, text_lookup_seconds = load_candidate_texts(
        args.chunks_path, hybrid_records
    )
    print(
        f"Loaded text for {len(texts)} unique candidates in "
        f"{text_lookup_seconds:.2f} seconds",
        flush=True,
    )

    reranker = CrossEncoderReranker(
        model_name=args.model_name,
        cache_folder=args.cache_folder,
        max_length=args.max_length,
        device="cpu",
    ).load()
    print(
        f"Loaded cross-encoder {args.model_name} in "
        f"{reranker.load_seconds:.2f} seconds",
        flush=True,
    )

    warmup_candidate = dict(hybrid_records[0]["results"][0])
    warmup_candidate["text"] = texts[warmup_candidate["chunk_id"]]
    _, warmup_seconds = reranker.rerank(
        hybrid_records[0]["query"], [warmup_candidate], batch_size=1
    )
    print(f"Completed reranker warm-up in {warmup_seconds:.3f} seconds", flush=True)

    run_records = []
    timing_rows = []
    for index, hybrid_record in enumerate(hybrid_records, start=1):
        candidates = []
        for result in hybrid_record["results"][: args.candidate_depth]:
            candidate = dict(result)
            candidate["text"] = texts[candidate["chunk_id"]]
            candidates.append(candidate)

        reranked, rerank_seconds = reranker.rerank(
            hybrid_record["query"],
            candidates,
            top_k=args.saved_depth,
            batch_size=args.batch_size,
        )
        retrieval_seconds = float(hybrid_record["latency_seconds"])
        parallel_retrieval_seconds = float(
            hybrid_record["parallel_estimate_seconds"]
        )
        sequential_seconds = retrieval_seconds + rerank_seconds
        parallel_estimate_seconds = parallel_retrieval_seconds + rerank_seconds
        run_record = {
            **{field: hybrid_record.get(field) for field in QUERY_FIELDS},
            "latency_seconds": sequential_seconds,
            "retrieval_seconds": retrieval_seconds,
            "rerank_seconds": rerank_seconds,
            "parallel_estimate_seconds": parallel_estimate_seconds,
            "candidate_depth": args.candidate_depth,
            "results": [result.to_dict() for result in reranked],
        }
        mark_relevance(run_record)
        run_records.append(run_record)
        timing_rows.append(
            {
                "query_id": run_record["query_id"],
                "retrieval_seconds": retrieval_seconds,
                "rerank_seconds": rerank_seconds,
                "sequential_seconds": sequential_seconds,
                "parallel_estimate_seconds": parallel_estimate_seconds,
            }
        )
        print(
            f"[{index:02d}/{len(hybrid_records):02d}] "
            f"{run_record['query_id']} reranked in {rerank_seconds:.3f} seconds",
            flush=True,
        )

    queries = [
        {field: record.get(field) for field in QUERY_FIELDS}
        for record in run_records
    ]
    counts = relevance_counts(args.chunks_path, queries)
    hybrid_metrics = read_json(args.hybrid_metrics_path)
    combined_load_seconds = (
        float(hybrid_metrics["run"]["index_load_seconds"])
        + reranker.load_seconds
    )
    metrics, per_query = calculate_metrics(
        run_records, counts, combined_load_seconds
    )
    metrics["run"].update(
        {
            "retriever": "hybrid_rrf_cross_encoder",
            "reranker_model": args.model_name,
            "candidate_depth": args.candidate_depth,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "text_lookup_seconds": text_lookup_seconds,
            "reranker_load_seconds": reranker.load_seconds,
            "reranker_warmup_seconds": warmup_seconds,
            "mean_rerank_seconds": statistics.fmean(
                row["rerank_seconds"] for row in timing_rows
            ),
            "mean_sequential_latency_seconds": statistics.fmean(
                row["sequential_seconds"] for row in timing_rows
            ),
            "mean_parallel_estimate_seconds": statistics.fmean(
                row["parallel_estimate_seconds"] for row in timing_rows
            ),
            "hybrid_cold_load_seconds": hybrid_metrics["run"][
                "index_load_seconds"
            ],
        }
    )
    metrics["limitations"].extend(
        [
            "The cross-encoder reranked only the top 20 saved hybrid candidates.",
            "The reranker was trained on MS MARCO rather than Amazon Electronics relevance judgments.",
            "End-to-end latency is reconstructed from saved retrieval runs plus measured reranking.",
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
