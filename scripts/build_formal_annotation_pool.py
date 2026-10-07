"""Pool and deduplicate formal retrieval candidates for human annotation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

SYSTEM_PATHS = {
    "bm25": RESULTS_DIR / "formal_bm25_results.jsonl",
    "dense": RESULTS_DIR / "formal_dense_results.jsonl",
    "hybrid_rrf": RESULTS_DIR / "formal_hybrid_rrf_results.jsonl",
    "cross_encoder": RESULTS_DIR / "formal_cross_encoder_results.jsonl",
}
QUERY_FIELDS = (
    "query_id",
    "query",
    "intent",
    "target_aspects",
    "target_category",
    "difficulty",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def stable_order_key(query_id: str, chunk_id: str) -> str:
    return hashlib.sha256(f"{query_id}|{chunk_id}".encode("utf-8")).hexdigest()


def validate_runs(runs: dict[str, list[dict]]) -> list[str]:
    reference_ids = [record["query_id"] for record in runs["bm25"]]
    if len(reference_ids) != 20 or len(set(reference_ids)) != 20:
        raise ValueError("The BM25 run must contain 20 unique formal queries.")
    for system, records in runs.items():
        query_ids = [record["query_id"] for record in records]
        if query_ids != reference_ids:
            raise ValueError(f"Query order mismatch in {system} results.")
        for record in records:
            if len(record["results"]) != 20:
                raise ValueError(
                    f"Expected 20 candidates in {system}/{record['query_id']}."
                )
    return reference_ids


def main() -> None:
    runs = {system: read_jsonl(path) for system, path in SYSTEM_PATHS.items()}
    query_ids = validate_runs(runs)
    indexed = {
        system: {record["query_id"]: record for record in records}
        for system, records in runs.items()
    }

    pool_records = []
    query_summaries = []
    for query_id in query_ids:
        query_record = indexed["bm25"][query_id]
        candidates: dict[str, dict] = {}
        system_sets = {}

        for system in SYSTEM_PATHS:
            system_record = indexed[system][query_id]
            system_ids = set()
            for result in system_record["results"]:
                chunk_id = str(result["chunk_id"])
                system_ids.add(chunk_id)
                candidate = candidates.setdefault(
                    chunk_id,
                    {
                        "chunk_id": chunk_id,
                        "parent_asin": result.get("parent_asin"),
                        "chunk_type": result.get("chunk_type"),
                        "aspect": result.get("aspect"),
                        "category": result.get("category"),
                        "rating": result.get("rating"),
                        "text": result.get("text"),
                        "systems": [],
                        "ranks": {},
                        "scores": {},
                    },
                )
                if not candidate.get("text") and result.get("text"):
                    candidate["text"] = result["text"]
                candidate["systems"].append(system)
                candidate["ranks"][system] = int(result["rank"])
                candidate["scores"][system] = float(result["score"])
            system_sets[system] = system_ids

        if any(not candidate.get("text") for candidate in candidates.values()):
            raise ValueError(f"Candidate text is missing for {query_id}.")

        ordered = sorted(
            candidates.values(),
            key=lambda candidate: stable_order_key(query_id, candidate["chunk_id"]),
        )
        for display_order, candidate in enumerate(ordered, start=1):
            pool_records.append(
                {
                    "annotation_id": f"{query_id}::{candidate['chunk_id']}",
                    **{field: query_record[field] for field in QUERY_FIELDS},
                    "display_order": display_order,
                    **candidate,
                }
            )

        query_summaries.append(
            {
                "query_id": query_id,
                "unique_candidates": len(ordered),
                "bm25_dense_overlap": len(
                    system_sets["bm25"] & system_sets["dense"]
                ),
                "hybrid_covered_by_source_union": system_sets["hybrid_rrf"]
                <= (system_sets["bm25"] | system_sets["dense"]),
                "cross_encoder_covered_by_hybrid": system_sets["cross_encoder"]
                <= system_sets["hybrid_rrf"],
            }
        )

    annotation_ids = [record["annotation_id"] for record in pool_records]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Duplicate annotation IDs remain after pooling.")
    if not all(
        row["hybrid_covered_by_source_union"]
        and row["cross_encoder_covered_by_hybrid"]
        for row in query_summaries
    ):
        raise ValueError("Candidate lineage validation failed.")

    output_path = EVALUATION_DIR / "formal_annotation_pool.jsonl"
    write_jsonl(output_path, pool_records)

    candidate_counts = [row["unique_candidates"] for row in query_summaries]
    overlaps = [row["bm25_dense_overlap"] for row in query_summaries]
    manifest = {
        "status": "complete",
        "query_count": len(query_ids),
        "candidate_pair_count": len(pool_records),
        "pooling_depth_per_system": 20,
        "systems": list(SYSTEM_PATHS),
        "candidate_counts": {
            "minimum_per_query": min(candidate_counts),
            "maximum_per_query": max(candidate_counts),
            "mean_per_query": statistics.fmean(candidate_counts),
        },
        "bm25_dense_overlap": {
            "minimum_per_query": min(overlaps),
            "maximum_per_query": max(overlaps),
            "mean_per_query": statistics.fmean(overlaps),
        },
        "blinding": {
            "candidate_display_order": "stable_sha256_shuffle",
            "system_ranks_kept_on_separate_provenance_sheet": True,
        },
        "query_summaries": query_summaries,
    }
    manifest_path = EVALUATION_DIR / "formal_annotation_pool_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
