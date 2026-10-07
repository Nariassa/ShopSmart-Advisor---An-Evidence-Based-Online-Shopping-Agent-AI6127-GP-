"""Run a reproducible pilot BM25 baseline on the Electronics knowledge base.

The pilot contains two silver-label suites:

1. Short-title known-item retrieval, where every chunk sharing the target ASIN
   is relevant.
2. Natural-language aspect retrieval, where the existing keyword-derived
   aspect label supplies relevance.

These suites validate the retrieval pipeline and provide an initial baseline.
They do not replace manually judged queries and qrels for the final report.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import sys
from typing import Iterable

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval import BM25Retriever  # noqa: E402


ASPECT_QUERIES = [
    ("aspect_battery_01", "wireless earbuds with long battery life", "battery life"),
    ("aspect_sound_01", "headphones with clear sound and strong bass", "sound quality"),
    ("aspect_display_01", "monitor with a bright sharp screen", "screen/display"),
    ("aspect_connectivity_01", "bluetooth connection keeps dropping", "connectivity"),
    ("aspect_build_01", "durable sturdy build quality", "build quality"),
    ("aspect_value_01", "best value budget electronics", "price/value"),
    ("aspect_ease_01", "easy to set up and simple to use", "ease of use"),
    ("aspect_design_01", "slim lightweight stylish design", "design/appearance"),
    ("aspect_service_01", "warranty support and customer service", "customer service"),
    ("aspect_performance_01", "fast reliable device performance", "specifications/performance"),
    ("aspect_battery_02", "phone battery drains quickly", "battery life"),
    ("aspect_sound_02", "speaker audio quality complaints", "sound quality"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-queries", type=int, default=3)
    parser.add_argument("--aspect-queries", type=int, default=6)
    parser.add_argument("--max-title-tokens", type=int, default=5)
    parser.add_argument("--retrieval-depth", type=int, default=100)
    parser.add_argument("--saved-depth", type=int, default=20)
    parser.add_argument(
        "--index-path", type=Path, default=PROJECT_ROOT / "bm25_index.pkl"
    )
    parser.add_argument(
        "--chunks-path", type=Path, default=PROJECT_ROOT / "chunks.parquet"
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "bm25_pilot_queries.jsonl",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "bm25_pilot_results.jsonl",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "bm25_pilot_metrics.json",
    )
    parser.add_argument(
        "--metrics-csv-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "bm25_pilot_metrics.csv",
    )
    return parser.parse_args()


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_queries(
    chunks_path: Path,
    product_query_count: int,
    aspect_query_count: int,
    max_title_tokens: int,
) -> list[dict]:
    con = duckdb.connect()
    sql = """
        SELECT
            parent_asin,
            regexp_extract(text, '^\\[Product: (.*?) \\| ASIN:', 1) AS title,
            category
        FROM read_parquet(?)
        WHERE chunk_type = 'spec'
          AND length(regexp_extract(text, '^\\[Product: (.*?) \\| ASIN:', 1))
              BETWEEN 25 AND 140
          AND instr(text, chr(65533)) = 0
        ORDER BY hash(parent_asin)
        LIMIT ?
    """
    product_rows = con.execute(
        sql, [str(chunks_path), product_query_count]
    ).fetchall()
    con.close()

    queries = []
    for index, (parent_asin, title, category) in enumerate(product_rows, start=1):
        short_title = " ".join(title.split()[:max_title_tokens])
        queries.append({
            "query_id": f"product_{index:03d}",
            "query": short_title,
            "source_title": title,
            "suite": "known_item_short_title",
            "target_parent_asin": parent_asin,
            "target_aspect": None,
            "category": category,
            "label_source": "parent_asin",
        })
    queries.extend(
        {
            "query_id": query_id,
            "query": query,
            "source_title": None,
            "suite": "aspect_silver",
            "target_parent_asin": None,
            "target_aspect": aspect,
            "category": None,
            "label_source": "keyword_derived_aspect",
        }
        for query_id, query, aspect in ASPECT_QUERIES[:aspect_query_count]
    )
    return queries


def enrich_results(chunks_path: Path, run_records: list[dict]) -> None:
    chunk_ids = sorted(
        {
            result["chunk_id"]
            for record in run_records
            for result in record["results"]
        }
    )
    ids_frame = pd.DataFrame({"chunk_id": chunk_ids})
    con = duckdb.connect()
    con.register("requested_ids", ids_frame)
    metadata_rows = con.execute(
        """
        SELECT c.chunk_id, c.parent_asin, c.chunk_type, c.aspect, c.category
        FROM read_parquet(?) AS c
        INNER JOIN requested_ids AS r USING (chunk_id)
        """,
        [str(chunks_path)],
    ).fetchall()
    con.close()
    metadata = {
        row[0]: {
            "parent_asin": row[1],
            "chunk_type": row[2],
            "aspect": row[3],
            "category": row[4],
        }
        for row in metadata_rows
    }

    for record in run_records:
        for result in record["results"]:
            result.update(metadata.get(result["chunk_id"], {}))
            if record["suite"] == "known_item_short_title":
                result["relevant"] = (
                    result.get("parent_asin") == record["target_parent_asin"]
                )
            else:
                result["relevant"] = result.get("aspect") == record["target_aspect"]


def relevance_counts(chunks_path: Path, queries: list[dict]) -> dict[str, int]:
    target_asins = [
        q["target_parent_asin"] for q in queries if q["target_parent_asin"]
    ]
    con = duckdb.connect()
    asin_frame = pd.DataFrame({"parent_asin": target_asins})
    con.register("target_asins", asin_frame)
    asin_rows = con.execute(
        """
        SELECT c.parent_asin, COUNT(*)
        FROM read_parquet(?) AS c
        INNER JOIN target_asins AS t USING (parent_asin)
        GROUP BY c.parent_asin
        """,
        [str(chunks_path)],
    ).fetchall()
    aspect_rows = con.execute(
        """
        SELECT aspect, COUNT(*)
        FROM read_parquet(?)
        GROUP BY aspect
        """,
        [str(chunks_path)],
    ).fetchall()
    con.close()
    counts = {f"asin:{key}": int(value) for key, value in asin_rows}
    counts.update({f"aspect:{key}": int(value) for key, value in aspect_rows})
    return counts


def dcg(relevance: list[bool], k: int) -> float:
    return sum(
        (1.0 if is_relevant else 0.0) / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevance[:k], start=1)
    )


def metric_at_k(relevance: list[bool], total_relevant: int, k: int) -> dict:
    observed = relevance[:k]
    relevant_retrieved = sum(observed)
    ideal_relevance = [True] * min(total_relevant, k)
    ideal_dcg = dcg(ideal_relevance, k)
    first_rank = next(
        (rank for rank, value in enumerate(observed, start=1) if value), None
    )
    return {
        f"precision_at_{k}": relevant_retrieved / k,
        f"recall_at_{k}": (
            relevant_retrieved / total_relevant if total_relevant else 0.0
        ),
        f"success_at_{k}": float(relevant_retrieved > 0),
        f"mrr_at_{k}": 1.0 / first_rank if first_rank else 0.0,
        f"ndcg_at_{k}": dcg(observed, k) / ideal_dcg if ideal_dcg else 0.0,
    }


def calculate_metrics(
    run_records: list[dict], total_counts: dict[str, int], index_load_seconds: float
) -> tuple[dict, list[dict]]:
    per_query = []
    for record in run_records:
        relevance = [bool(result.get("relevant")) for result in record["results"]]
        if record["target_parent_asin"]:
            count_key = f"asin:{record['target_parent_asin']}"
        else:
            count_key = f"aspect:{record['target_aspect']}"
        total_relevant = total_counts.get(count_key, 0)
        query_metrics = {
            "query_id": record["query_id"],
            "suite": record["suite"],
            "latency_seconds": record["latency_seconds"],
            "total_relevant": total_relevant,
        }
        for k in (1, 5, 10):
            query_metrics.update(metric_at_k(relevance, total_relevant, k))
        per_query.append(query_metrics)

    aggregate = {}
    for suite in sorted({row["suite"] for row in per_query}):
        rows = [row for row in per_query if row["suite"] == suite]
        metric_names = [
            key
            for key in rows[0]
            if key not in {"query_id", "suite", "latency_seconds", "total_relevant"}
        ]
        aggregate[suite] = {
            "query_count": len(rows),
            **{
                name: statistics.fmean(row[name] for row in rows)
                for name in metric_names
            },
        }

    latencies = [record["latency_seconds"] for record in run_records]
    sorted_latencies = sorted(latencies)
    p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
    payload = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "corpus_size": 4_450_749,
            "index_load_seconds": index_load_seconds,
            "query_count": len(run_records),
            "mean_query_latency_seconds": statistics.fmean(latencies),
            "median_query_latency_seconds": statistics.median(latencies),
            "p95_query_latency_seconds": sorted_latencies[p95_index],
            "evaluation_status": "pilot_silver_labels",
        },
        "aggregate": aggregate,
        "limitations": [
            "Short-title queries contain lexical overlap with the indexed product prefix.",
            "Aspect relevance uses keyword-derived labels from the corpus pipeline.",
            "The pilot is for pipeline validation and initial comparison only.",
            "Final claims require manually judged natural-language queries and qrels.",
        ],
    }
    return payload, per_query


def main() -> None:
    args = parse_args()
    if args.saved_depth > args.retrieval_depth:
        raise ValueError("saved-depth cannot exceed retrieval-depth")

    queries = build_queries(
        args.chunks_path,
        args.product_queries,
        args.aspect_queries,
        args.max_title_tokens,
    )
    write_jsonl(args.queries_path, queries)
    print(f"Prepared {len(queries)} pilot queries at {args.queries_path}", flush=True)

    retriever = BM25Retriever(args.index_path).load()
    print(
        f"Loaded BM25 index with {retriever.corpus_size:,} documents "
        f"in {retriever.load_seconds:.2f} seconds",
        flush=True,
    )

    run_records = []
    for index, query_record in enumerate(queries, start=1):
        results, elapsed = retriever.search(
            query_record["query"], top_k=args.retrieval_depth
        )
        run_record = {
            **query_record,
            "latency_seconds": elapsed,
            "retrieval_depth": args.retrieval_depth,
            "results": [result.to_dict() for result in results],
        }
        run_records.append(run_record)
        print(
            f"[{index:02d}/{len(queries):02d}] {query_record['query_id']} "
            f"completed in {elapsed:.2f} seconds",
            flush=True,
        )

    enrich_results(args.chunks_path, run_records)
    counts = relevance_counts(args.chunks_path, queries)
    metrics, per_query = calculate_metrics(
        run_records, counts, retriever.load_seconds
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
    pd.DataFrame(per_query).to_csv(args.metrics_csv_path, index=False)

    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved results to {args.results_path}", flush=True)
    print(f"Saved metrics to {args.metrics_path}", flush=True)


if __name__ == "__main__":
    main()
