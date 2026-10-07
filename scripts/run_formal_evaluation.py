"""Evaluate all formal retrieval runs with the human-validated graded qrels."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "evaluation"
RESULTS = ROOT / "results"
RUNS = {
    "BM25": RESULTS / "formal_bm25_results.jsonl",
    "Dense": RESULTS / "formal_dense_results.jsonl",
    "Hybrid RRF": RESULTS / "formal_hybrid_rrf_results.jsonl",
    "Cross-Encoder": RESULTS / "formal_cross_encoder_results.jsonl",
}
DEPTHS = (1, 5, 10, 20)
RELEVANT_THRESHOLD = 1


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dcg(grades: list[int], k: int) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades[:k], start=1)
    )


def query_metrics(grades: list[int], judged_grades: list[int]) -> dict:
    total_relevant = sum(grade >= RELEVANT_THRESHOLD for grade in judged_grades)
    ideal = sorted(judged_grades, reverse=True)
    metrics = {
        "judged_pool_size": len(judged_grades),
        "pool_relevant_count": total_relevant,
        "pool_highly_relevant_count": sum(grade == 2 for grade in judged_grades),
    }
    for k in DEPTHS:
        observed = grades[:k]
        relevant = sum(grade >= RELEVANT_THRESHOLD for grade in observed)
        first = next(
            (rank for rank, grade in enumerate(observed, start=1) if grade >= RELEVANT_THRESHOLD),
            None,
        )
        ideal_dcg = dcg(ideal, k)
        metrics.update(
            {
                f"precision_at_{k}": relevant / k,
                f"highly_relevant_precision_at_{k}": sum(grade == 2 for grade in observed) / k,
                f"pool_recall_at_{k}": relevant / total_relevant if total_relevant else 0.0,
                f"success_at_{k}": float(relevant > 0),
                f"mrr_at_{k}": 1.0 / first if first else 0.0,
                f"ndcg_at_{k}": dcg(observed, k) / ideal_dcg if ideal_dcg else 0.0,
            }
        )
    return metrics


def load_qrels() -> tuple[dict[str, dict[str, int]], Path]:
    path = EVALUATION / "qrels_human_validated_v1.txt"
    qrels: dict[str, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        query_id, _, chunk_id, grade_text = line.split()
        grade = int(grade_text)
        assert grade in (0, 1, 2)
        query_qrels = qrels.setdefault(query_id, {})
        assert chunk_id not in query_qrels
        query_qrels[chunk_id] = grade
    assert len(qrels) == 20
    assert sum(len(rows) for rows in qrels.values()) == 791
    return qrels, path


def evaluate_run(system: str, path: Path, qrels: dict[str, dict[str, int]]) -> tuple[dict, list[dict]]:
    records = read_jsonl(path)
    assert len(records) == 20, system
    assert len({record["query_id"] for record in records}) == 20, system
    per_query = []
    for record in records:
        query_id = record["query_id"]
        results = record["results"]
        assert len(results) == 20, (system, query_id)
        chunk_ids = [result["chunk_id"] for result in results]
        assert len(set(chunk_ids)) == 20, (system, query_id)
        assert [int(result["rank"]) for result in results] == list(range(1, 21))
        assert all(chunk_id in qrels[query_id] for chunk_id in chunk_ids)
        grades = [qrels[query_id][chunk_id] for chunk_id in chunk_ids]
        metrics = query_metrics(grades, list(qrels[query_id].values()))
        per_query.append(
            {
                "system": system,
                "query_id": query_id,
                "difficulty": record["difficulty"],
                "intent": record["intent"],
                "latency_seconds": record["latency_seconds"],
                **metrics,
            }
        )

    metric_names = [
        key
        for key in per_query[0]
        if key
        not in {
            "system",
            "query_id",
            "difficulty",
            "intent",
            "latency_seconds",
            "judged_pool_size",
            "pool_relevant_count",
            "pool_highly_relevant_count",
        }
    ]
    latencies = sorted(row["latency_seconds"] for row in per_query)
    aggregate = {
        "system": system,
        "query_count": len(per_query),
        "mean_latency_seconds": statistics.fmean(latencies),
        "median_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": latencies[math.ceil(0.95 * len(latencies)) - 1],
        **{
            name: statistics.fmean(row[name] for row in per_query)
            for name in metric_names
        },
    }
    return aggregate, per_query


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    qrels, qrels_path = load_qrels()
    aggregates, all_per_query = [], []
    run_hashes = {}
    for system, path in RUNS.items():
        aggregate, per_query = evaluate_run(system, path, qrels)
        aggregates.append(aggregate)
        all_per_query.extend(per_query)
        run_hashes[system] = digest(path)

    ranking = sorted(
        aggregates,
        key=lambda row: (-row["ndcg_at_10"], -row["mrr_at_10"], -row["precision_at_10"]),
    )
    payload = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_status": "complete_human_validated_single_reviewer_qrels",
            "query_count": 20,
            "candidate_depth": 20,
            "qrels_path": qrels_path.relative_to(ROOT).as_posix(),
            "qrels_sha256": digest(qrels_path),
            "relevant_threshold": RELEVANT_THRESHOLD,
            "binary_metric_definition": "Labels 1 and 2 are relevant; label 0 is not relevant.",
            "graded_metric_definition": "nDCG uses gains 2^label - 1 for labels 0, 1, and 2.",
            "recall_definition": "Recall denominators contain relevant candidates in the pooled top-20 union, not the full corpus.",
            "human_validation": "One project member reviewed all 791 pairs and confirmed the labels.",
            "independent_second_annotator": False,
            "inter_annotator_agreement": None,
        },
        "aggregate": aggregates,
        "ranking_by_ndcg_at_10": [row["system"] for row in ranking],
        "per_query": all_per_query,
        "source_hashes": {"qrels": digest(qrels_path), **run_hashes},
    }
    metrics_path = RESULTS / "formal_human_validated_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(RESULTS / "formal_human_validated_comparison.csv", aggregates)
    write_csv(RESULTS / "formal_human_validated_per_query.csv", all_per_query)
    manifest = {
        "status": "complete",
        "query_count": 20,
        "system_count": 4,
        "candidate_depth": 20,
        "qrels_pair_count": 791,
        "best_system_by_ndcg_at_10": ranking[0]["system"],
        "ranking_by_ndcg_at_10": [row["system"] for row in ranking],
        "metrics_sha256": digest(metrics_path),
        "comparison_sha256": digest(RESULTS / "formal_human_validated_comparison.csv"),
        "per_query_sha256": digest(RESULTS / "formal_human_validated_per_query.csv"),
        "qrels_sha256": digest(qrels_path),
    }
    (RESULTS / "formal_human_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "aggregate": aggregates}, indent=2))


if __name__ == "__main__":
    main()
