"""Run Grounded Generation & Purchase Confidence Scoring independently, or connect it to the supplied knowledge base."""

import argparse
import json
from pathlib import Path
import sys

from .adapters import ParquetEvidenceStore, RetrievalAdapter
from .generator import ShopSmartGenerator
from .llm import add_ollama_arguments, backend_from_args


ROOT = Path(__file__).resolve().parents[2]


def read_records(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, list) else payload["evidence"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Grounded Generation & Purchase Confidence Scoring: grounded recommendations and PCS")
    parser.add_argument("--query", default="Are these headphones good for battery life and comfort?")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true", help="Use clearly labeled synthetic evidence")
    source.add_argument("--evidence", type=Path, help="Retrieval, Reranking & Evaluation evidence JSON/JSONL")
    source.add_argument("--product", action="append", help="Real ASIN; repeat for comparison")
    source.add_argument("--retrieve", choices=("dense", "bm25", "hybrid"))
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent)
    parser.add_argument("--chunks-path", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--product-limit", type=int, default=200)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--aspects", help="Semicolon-separated requested aspect names")
    parser.add_argument("--signals", type=Path, help="Aspect Analysis & Purchase Decision Explanation signals JSON, keyed by ASIN")
    parser.add_argument("--max-price", type=float)
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--backend", choices=("extractive", "ollama"), default="extractive")
    add_ollama_arguments(parser)
    parser.add_argument("--prompt-version", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--max-facts", type=int, default=30)
    parser.add_argument("--max-claims", type=int, default=12)
    parser.add_argument("--max-context-chars", type=int, default=10000)
    parser.add_argument("--require-llm", action="store_true", help="Fail if LLM does not produce supported claims")
    parser.add_argument("--sentiment-model", help="Explicitly load/download a transformer classifier")
    parser.add_argument("--nli-model", help="Explicitly load/download an NLI model for paraphrases")
    parser.add_argument("--output", type=Path, help="Write UI/evaluation JSON and sibling Markdown")
    args = parser.parse_args(argv)
    if args.require_llm and args.backend != "ollama":
        parser.error("--require-llm 需要同时选择 --backend ollama。")
    retrieval_timing = None
    if args.evidence:
        records = read_records(args.evidence)
        if records and any(not row.get("text") for row in records):
            store = ParquetEvidenceStore(args.chunks_path or args.data_dir / "chunks.parquet")
            records = store.hydrate(records)
    elif args.product:
        store = ParquetEvidenceStore(args.chunks_path or args.data_dir / "chunks.parquet")
        records = [row for asin in args.product for row in store.product_chunks(asin, args.product_limit)]
    elif args.retrieve:
        adapter = RetrievalAdapter(args.data_dir, args.retrieve, args.rerank)
        records, retrieval_timing = adapter.search(args.query, args.top_k, args.candidate_k)
    else:
        records = read_records(ROOT / "examples" / "generation_evidence.json")
    classifier = verifier = None
    if args.sentiment_model:
        from .sentiment import TransformerSentiment
        classifier = TransformerSentiment(args.sentiment_model)
    if args.nli_model:
        from .grounding import NLIVerifier
        verifier = NLIVerifier(args.nli_model)
    llm = backend_from_args(args) if args.backend == "ollama" else None
    generator = ShopSmartGenerator(llm, classifier, verifier, allow_fallback=not args.require_llm,
                                   max_facts=args.max_facts, max_claims=args.max_claims,
                                   max_context_chars=args.max_context_chars, prompt_version=args.prompt_version)
    signals = json.loads(args.signals.read_text(encoding="utf-8-sig")) if args.signals else None
    constraints = {"max_price": args.max_price, "currency": args.currency} if args.max_price is not None else None
    result = generator.generate_recommendation(
        args.query, records, requested_aspects=args.aspects.split(";") if args.aspects else None,
        product_signals=signals, constraints=constraints)
    if retrieval_timing:
        result["retrieval_timing"] = retrieval_timing
    result["input_source"] = ("synthetic_demo" if not (args.evidence or args.product or args.retrieve)
                              else "parquet_product_sample" if args.product
                              else "retrieval" if args.retrieve else "evidence_file")
    if result["input_source"] == "synthetic_demo":
        print("演示数据为人工构造样例；真实 Amazon 数据请使用 --product、--evidence 或 --retrieve。")
    if args.output:
        target = args.output.resolve()
        if target.suffix.lower() != ".json":
            parser.error("--output 文件需要使用 .json 扩展名。")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        target.with_suffix(".md").write_text(result["answer"], encoding="utf-8")
    if result["backend"] == "llm":
        print(f"本地模型：{args.model}；模型结论通过检查：{result['grounding']['llm_verified_claims']} 条。")
    print(result["answer"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        print(f"运行未完成：{exc}\n请按 docs/MEMBER_C.md 核对数据路径、参数及所选模型环境后重试。", file=sys.stderr)
        raise SystemExit(1)
