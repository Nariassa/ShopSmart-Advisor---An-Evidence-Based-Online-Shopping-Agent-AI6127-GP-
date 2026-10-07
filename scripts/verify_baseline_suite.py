"""Validate all retrieval baseline artifacts and write a suite manifest."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
QUERY_PATH = PROJECT_ROOT / "evaluation" / "bm25_pilot_queries.jsonl"
MANIFEST_PATH = RESULTS_ROOT / "baseline_suite_manifest.json"

EXPECTED_CORPUS_SIZE = 4_450_749
EXPECTED_QUERY_COUNT = 9
EXPECTED_SAVED_DEPTH = 20

BASELINES = [
    {
        "name": "bm25",
        "description": "Lexical BM25 retrieval",
        "results": RESULTS_ROOT / "bm25_pilot_results.jsonl",
        "metrics": RESULTS_ROOT / "bm25_pilot_metrics.json",
        "readme": PROJECT_ROOT / "BM25_BASELINE_README.md",
        "runner": PROJECT_ROOT / "scripts" / "run_bm25_baseline.py",
    },
    {
        "name": "dense",
        "description": "BAAI/bge-base-en-v1.5 dense retrieval",
        "results": RESULTS_ROOT / "dense_pilot_results.jsonl",
        "metrics": RESULTS_ROOT / "dense_pilot_metrics.json",
        "readme": PROJECT_ROOT / "DENSE_BASELINE_README.md",
        "runner": PROJECT_ROOT / "scripts" / "run_dense_baseline.py",
    },
    {
        "name": "hybrid_rrf",
        "description": "BM25 plus dense retrieval with Reciprocal Rank Fusion",
        "results": RESULTS_ROOT / "hybrid_rrf_pilot_results.jsonl",
        "metrics": RESULTS_ROOT / "hybrid_rrf_pilot_metrics.json",
        "readme": PROJECT_ROOT / "HYBRID_RRF_BASELINE_README.md",
        "runner": PROJECT_ROOT / "scripts" / "run_hybrid_rrf_baseline.py",
    },
    {
        "name": "cross_encoder",
        "description": "Hybrid RRF followed by cross-encoder reranking",
        "results": RESULTS_ROOT / "cross_encoder_pilot_results.jsonl",
        "metrics": RESULTS_ROOT / "cross_encoder_pilot_metrics.json",
        "readme": PROJECT_ROOT / "CROSS_ENCODER_BASELINE_README.md",
        "runner": PROJECT_ROOT / "scripts" / "run_cross_encoder_baseline.py",
    },
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def main() -> None:
    queries = read_jsonl(QUERY_PATH)
    if len(queries) != EXPECTED_QUERY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_QUERY_COUNT} queries, found {len(queries)}"
        )
    expected_query_ids = [record["query_id"] for record in queries]
    if len(set(expected_query_ids)) != len(expected_query_ids):
        raise ValueError("Pilot query IDs are not unique.")

    comparison_path = RESULTS_ROOT / "baseline_comparison.csv"
    with comparison_path.open("r", encoding="utf-8", newline="") as handle:
        comparison_rows = list(csv.DictReader(handle))
    comparison_names = [row["retriever"] for row in comparison_rows]
    expected_names = [baseline["name"] for baseline in BASELINES]
    if comparison_names != expected_names:
        raise ValueError(
            f"Comparison rows do not match baseline order: {comparison_names}"
        )

    manifest_runs = []
    for baseline in BASELINES:
        for key in ("results", "metrics", "readme", "runner"):
            if not baseline[key].is_file():
                raise FileNotFoundError(
                    f"Missing {baseline['name']} {key}: {baseline[key]}"
                )

        results = read_jsonl(baseline["results"])
        metrics = read_json(baseline["metrics"])
        query_ids = [record["query_id"] for record in results]
        if query_ids != expected_query_ids:
            raise ValueError(f"Query order mismatch in {baseline['name']} results")
        if len(results) != EXPECTED_QUERY_COUNT:
            raise ValueError(f"Unexpected result count for {baseline['name']}")
        for record in results:
            hits = record["results"]
            if len(hits) != EXPECTED_SAVED_DEPTH:
                raise ValueError(
                    f"{baseline['name']} {record['query_id']} has {len(hits)} hits"
                )
            chunk_ids = [hit["chunk_id"] for hit in hits]
            if len(set(chunk_ids)) != len(chunk_ids):
                raise ValueError(
                    f"Duplicate hit in {baseline['name']} {record['query_id']}"
                )

        run = metrics["run"]
        if int(run["corpus_size"]) != EXPECTED_CORPUS_SIZE:
            raise ValueError(f"Corpus-size mismatch for {baseline['name']}")
        if int(run["query_count"]) != EXPECTED_QUERY_COUNT:
            raise ValueError(f"Query-count mismatch for {baseline['name']}")
        if run["evaluation_status"] != "pilot_silver_labels":
            raise ValueError(f"Unexpected evaluation status for {baseline['name']}")

        aspect = metrics["aggregate"]["aspect_silver"]
        known_item = metrics["aggregate"]["known_item_short_title"]
        manifest_runs.append(
            {
                "name": baseline["name"],
                "description": baseline["description"],
                "status": "complete",
                "artifacts": {
                    "results": relative(baseline["results"]),
                    "metrics": relative(baseline["metrics"]),
                    "readme": relative(baseline["readme"]),
                    "runner": relative(baseline["runner"]),
                },
                "latency": {
                    "cold_initialization_seconds": run["index_load_seconds"],
                    "mean_query_seconds": run["mean_query_latency_seconds"],
                    "p95_query_seconds": run["p95_query_latency_seconds"],
                },
                "aspect_metrics": {
                    "precision_at_5": aspect["precision_at_5"],
                    "precision_at_10": aspect["precision_at_10"],
                    "mrr_at_10": aspect["mrr_at_10"],
                    "ndcg_at_10": aspect["ndcg_at_10"],
                },
                "known_item_metrics": {
                    "success_at_10": known_item["success_at_10"],
                    "mrr_at_10": known_item["mrr_at_10"],
                    "ndcg_at_10": known_item["ndcg_at_10"],
                },
            }
        )

    manifest = {
        "suite": "amazon_electronics_retrieval_baselines",
        "status": "complete",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_size": EXPECTED_CORPUS_SIZE,
        "evaluation": {
            "status": "pilot_silver_labels",
            "query_count": EXPECTED_QUERY_COUNT,
            "saved_depth": EXPECTED_SAVED_DEPTH,
            "query_file": relative(QUERY_PATH),
            "comparison_file": relative(comparison_path),
        },
        "baseline_order": expected_names,
        "baselines": manifest_runs,
        "next_stage": "manual_queries_and_qrels",
    }
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(
        f"Baseline suite validation passed: {len(BASELINES)} baselines, "
        f"{EXPECTED_QUERY_COUNT} queries, {EXPECTED_SAVED_DEPTH} saved hits each."
    )
    print(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
