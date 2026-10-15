"""Generate aligned Q4 outputs for System Integration, User Interface & Evaluation; no invented evaluation labels."""

import argparse
import json
from pathlib import Path
import sys

from .__main__ import read_records
from .adapters import ParquetEvidenceStore
from .generator import ShopSmartGenerator
from .llm import add_ollama_arguments, backend_from_args


def citation_audit(result):
    """Mechanical traceability only; this is not RAGAS faithfulness."""
    total = valid = 0
    for claim in result["claims"]:
        for citation in claim["citations"]:
            total += 1
            source = result["sources"].get(citation["source_id"])
            if (source and source["chunk_id"] == citation["chunk_id"]
                    and source["parent_asin"] == claim["parent_asin"]
                    and source["text"][citation["start"]:citation["end"]] == citation["quote"]):
                valid += 1
    return {"citation_count": total, "valid_citation_count": valid,
            "citation_traceability": valid/total if total else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate System Integration, User Interface & Evaluation evaluation inputs from Retrieval, Reranking & Evaluation's retrieved evidence")
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--evidence-map", required=True, type=Path,
                        help="JSON object: query_id -> evidence list or evidence file path")
    parser.add_argument("--chunks-path", type=Path, help="Needed if the map contains only chunk IDs")
    parser.add_argument("--signals", type=Path)
    parser.add_argument("--backend", choices=("extractive", "ollama"), default="extractive")
    add_ollama_arguments(parser)
    parser.add_argument("--prompt-version", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--require-llm", action="store_true")
    parser.add_argument("--nli-model", help="Local English NLI model for single-fact rewrites")
    parser.add_argument("--max-claims", type=int, default=12)
    parser.add_argument("--max-context-chars", type=int, default=10000)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--export-ragas", type=Path)
    args = parser.parse_args(argv)
    if args.require_llm and args.backend != "ollama":
        parser.error("--require-llm 需要同时选择 --backend ollama。")
    queries = read_records(args.queries)
    mapping = json.loads(args.evidence_map.read_text(encoding="utf-8-sig"))
    signals = json.loads(args.signals.read_text(encoding="utf-8-sig")) if args.signals else None
    ids = [q["query_id"] for q in queries]
    if len(set(ids)) != len(ids):
        raise ValueError("query_id 必须唯一，请先修正查询文件。")
    missing = [qid for qid in ids if qid not in mapping]
    if missing:
        raise ValueError(f"缺少以下查询的真实检索证据：{missing}；请补齐 --evidence-map。")
    # Resolve all inputs before publishing output, so a missing map entry
    # cannot leave a misleading partially completed evaluation file.
    records_by_id = {}
    store = ParquetEvidenceStore(args.chunks_path) if args.chunks_path else None
    for query_id in ids:
        value = mapping[query_id]
        if isinstance(value, str):
            evidence_path = Path(value)
            if not evidence_path.is_absolute():
                evidence_path = args.evidence_map.parent/evidence_path
            value = read_records(evidence_path)
        if not isinstance(value, list):
            raise ValueError(f"{query_id} 的证据必须是列表或文件路径。")
        if any(not row.get("text") for row in value):
            if store is None:
                raise ValueError("证据缺少 text，需要传入 --chunks-path。")
            value = store.hydrate(value)
        records_by_id[query_id] = value
    backend = backend_from_args(args) if args.backend == "ollama" else None
    verifier = None
    if args.nli_model:
        from .grounding import NLIVerifier
        verifier = NLIVerifier(args.nli_model)
    generator = ShopSmartGenerator(llm=backend, allow_fallback=not args.require_llm,
                                   grounding_verifier=verifier,
                                   max_claims=args.max_claims, max_context_chars=args.max_context_chars,
                                   prompt_version=args.prompt_version)
    results, ragas_inputs = [], []
    for query in queries:
        query_id = query["query_id"]
        aspects = query.get("target_aspects")
        if isinstance(aspects, str):
            aspects = [a.strip() for a in aspects.split(";") if a.strip()]
        output = generator.generate_recommendation(query["query"], records_by_id[query_id],
                                                    requested_aspects=aspects, product_signals=signals)
        results.append({"query_id": query_id, **output, "mechanical_audit": citation_audit(output)})
        item = {"query_id": query_id, "user_input": query["query"],
                "response": output["answer"],
                "retrieved_contexts": [row["text"] for row in records_by_id[query_id]]}
        if query.get("reference"):
            item["reference"] = query["reference"]
        ragas_inputs.append(item)
        print(f"{query_id}: {output['backend']}, {len(output['claims'])} 条有来源结论")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in results), encoding="utf-8")
    if args.export_ragas:
        args.export_ragas.parent.mkdir(parents=True, exist_ok=True)
        args.export_ragas.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in ragas_inputs), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        print(f"批量生成未完成：{exc}\n请核对查询与证据映射后重试。", file=sys.stderr)
        raise SystemExit(1)
