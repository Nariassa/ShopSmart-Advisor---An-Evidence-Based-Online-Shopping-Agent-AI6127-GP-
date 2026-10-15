"""Compare frozen prompt versions on identical facts, before rule supplementation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

from .__main__ import ROOT, read_records
from .compare import safe_model_name
from .evidence import canonical_aspect, normalize_evidence
from .generator import infer_constraints
from .grounding import extract_facts, verify_claims
from .llm import add_ollama_arguments, backend_from_args
from .prompts import build_grounded_prompt
from .sentiment import LexiconSentiment, evidence_sentiment


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_rubric(rubric, query, aspects, facts):
    if rubric["query"] != query or rubric["requested_aspects"] != aspects:
        raise ValueError("评分表的问题或需求方面不匹配，请使用当前测试对应的评分表。")
    if rubric["facts_sha256"] != digest([f.to_dict() for f in facts]):
        raise ValueError("评分表的证据哈希不匹配，请先核对固定证据。")
    labels = rubric["labels"]
    if set(labels) != {f.fact_id for f in facts}:
        raise ValueError("评分表必须完整覆盖所有证据 ID。")
    for label in labels.values():
        if label["relevance"] not in (0, 1, 2) or not set(label["aspects"]) <= set(aspects):
            raise ValueError("评分表相关性或需求方面无效。")
    return labels


def selection_metrics(payload, accepted, rejected, aspects, labels):
    proposed = payload["claims"]
    count = len(accepted)
    metrics = {"proposed_claims": len(proposed), "verified_unique_claims": count,
               "rejected_claims": len(rejected),
               "verified_unique_per_proposal": count / len(proposed) if proposed else None}
    if labels is None:
        return metrics
    selected = [labels[c["fact_id"]] for c in accepted]
    direct = [label for label in selected if label["relevance"] == 2]
    covered = sorted({aspect for label in direct for aspect in label["aspects"]})
    metrics.update({"direct_relevance_rate": len(direct) / count if count else None,
                    "useful_relevance_rate": sum(label["relevance"] >= 1 for label in selected) / count if count else None,
                    "irrelevant_claims": sum(label["relevance"] == 0 for label in selected),
                    "background_claims": sum(label["relevance"] == 1 for label in selected),
                    "requested_aspects_covered_by_direct_evidence": covered,
                    "requested_aspect_coverage": len(covered) / len(aspects) if aspects else None})
    return metrics


def percent(value):
    return f"{value:.0%}" if value is not None else "—"


def render_summary(summary):
    lines = ["# 原版与改进版 Prompt 对照测试", "", f"问题：{summary['query']}", "",
             f"需求方面：{'; '.join(summary['requested_aspects'])}", "",
             f"输入片段 {summary['evidence_records']} 条；模型实际接收相同的 {summary['fact_count']} 条证据句。",
             f"证据句 SHA-256：`{summary['facts_sha256']}`", "",
             "各版本使用相同 JSON schema、模型参数、证据及顺序；v2/v3 增加明确需求和相关性规则。",
             "直接调用模型并验证其原始输出，不加入规则补充证据，也不以最终答案引用检查率代替相关性。", "",
             "| 模型 | Prompt | 原始提议 | 验证通过 | 拒绝 | 直接相关率 | 有用证据率 | 无关条数 | 方面覆盖 | 生成耗时 | 输入 token |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["results"]:
        if not row["completed"]:
            lines.append(f"| {row['model']} | {row['prompt_version']} | 失败 | — | — | — | — | — | — | — | — |")
            continue
        m, generation = row["metrics"], row["generation"] or {}
        lines.append(f"| {row['model']} | {row['prompt_version']} | {m['proposed_claims']} | "
                     f"{m['verified_unique_claims']} | {m['rejected_claims']} | "
                     f"{percent(m.get('direct_relevance_rate'))} | {percent(m.get('useful_relevance_rate'))} | "
                     f"{m.get('irrelevant_claims', '—')} | {percent(m.get('requested_aspect_coverage'))} | "
                     f"{generation.get('generation_seconds', 0):.2f}s | {generation.get('prompt_tokens', '—')} |")
    lines += ["", "评分口径：相关性 2=直接回答需求，1=有用背景，0=无关或没有可用信息。",
              "直接相关率、有用证据率和方面覆盖均基于验证通过的模型证据；拒绝项另列。",
              "方面覆盖只由直接证据计入，规格或操作说明不会自动当作体验证据。",
              "评分表由开发助手在运行新模型结果前逐句审阅制定，可由系统集成、交互与综合评估模块 复核；它不是 791 条人工标注，",
              "不是独立人工评估或 RAGAS 分数，也没有传给模型。",
              "这是一道开发问题、每个模型每个版本运行一次的探索性对照，不能证明其他问题或正式测试集上的提升。",
              "temperature=0、seed=42 不保证跨设备完全复现。生成耗时仅包含解码；加载和完整请求耗时保存在 JSON。",
              "本问题没有预算或类别限制，约束规则已接入，但本次未验证这类场景的效果。", ""]
    if summary.get("rubric_source") is None:
        lines += ["未提供评分表，只报告机械验证和运行统计；没有估计语义相关性。", ""]
    for row in summary["results"]:
        if not row["completed"]:
            lines += [f"运行失败：{row['model']} / {row['prompt_version']}：{row['error']}", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare original and improved prompts on identical local-model evidence")
    parser.add_argument("--models", nargs="+", default=["qwen2.5:7b", "llama3.1:8b"])
    parser.add_argument("--versions", nargs="+", choices=["v1", "v2", "v3"], default=["v1", "v2"])
    parser.add_argument("--evidence", type=Path,
                        default=ROOT/"outputs/local_model_integration_2026-10-13/input_evidence.json")
    parser.add_argument("--query", default="Is this key finder reliable over Bluetooth and easy to set up?")
    parser.add_argument("--aspects", default="connectivity;ease of use")
    parser.add_argument("--constraints", type=Path, help="Optional JSON object of explicit user requirements")
    parser.add_argument("--rubric", type=Path, help="Optional pre-reviewed fact relevance labels, never sent to LLM")
    parser.add_argument("--max-facts", type=int, default=18)
    parser.add_argument("--max-claims", type=int, default=8)
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/prompt_comparison")
    add_ollama_arguments(parser, include_model=False)
    args = parser.parse_args(argv)
    names = [safe_model_name(model) for model in args.models]
    if len(set(names)) != len(names):
        raise ValueError("模型名称重复或输出文件名冲突，请修改 --models。")
    if len(set(args.versions)) != len(args.versions):
        raise ValueError("Prompt 版本重复，请修改 --versions。")
    records = read_records(args.evidence)
    evidence = normalize_evidence(records)
    classifier = LexiconSentiment()
    sentiments = {item.chunk_id: evidence_sentiment(item, classifier) for item in evidence}
    aspects = list(dict.fromkeys(canonical_aspect(a) for a in args.aspects.split(';') if a.strip()))
    constraints = json.loads(args.constraints.read_text(encoding="utf-8")) if args.constraints else infer_constraints(args.query)
    if not isinstance(constraints, dict):
        raise ValueError("需求限制文件必须是 JSON 对象。")
    facts = extract_facts(evidence, args.query, aspects, classifier, sentiments,
                          args.max_facts, args.max_context_chars)
    if not facts:
        raise ValueError("没有可用证据句，请检查问题和输入证据。")
    rubric = json.loads(args.rubric.read_text(encoding="utf-8")) if args.rubric else None
    labels = validate_rubric(rubric, args.query, aspects, facts) if rubric else None
    fact_rows = [f.to_dict() for f in facts]
    summary = {"scope": "single_development_query_prompt_pilot_not_formal_evaluation",
               "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "query": args.query, "requested_aspects": aspects, "constraints": constraints,
               "evidence_source": str(args.evidence.resolve()), "evidence_records": len(records),
               "evidence_sha256": digest(records), "fact_count": len(facts), "facts_sha256": digest(fact_rows),
               "rubric_source": str(args.rubric.resolve()) if args.rubric else None,
               "repeats_per_model_version": 1, "options": {"temperature": 0, "seed": 42,
                   "num_ctx": args.num_ctx, "num_predict": args.num_predict, "keep_alive": args.keep_alive},
               "max_claims": args.max_claims, "results": []}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir/"input_evidence.json", records)
    write_json(args.output_dir/"input_facts.json", fact_rows)
    if rubric:
        write_json(args.output_dir/"evaluation_rubric.json", rubric)
    prompts = {}
    for version in args.versions:
        prompts[version] = build_grounded_prompt(args.query, facts, max_claims=args.max_claims,
                                                requested_aspects=aspects, constraints=constraints, version=version)
        (args.output_dir/f"prompt_{version}.txt").write_text(prompts[version], encoding="utf-8")
    # Evidence payloads must remain byte-for-byte identical between the versions.
    evidence_payloads = {prompt.split("EVIDENCE_FACTS_JSON: ", 1)[1] for prompt in prompts.values()}
    if len(evidence_payloads) != 1:
        raise ValueError("各版本的证据不一致，已停止对照运行。")
    failed = False
    for model, name in zip(args.models, names):
        for version in args.versions:
            prompt = prompts[version]
            row = {"model": model, "prompt_version": version, "completed": False,
                   "facts_sha256": summary["facts_sha256"],
                   "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
            print(f"运行：{model} / {version}", flush=True)
            started = time.perf_counter()
            try:
                backend = backend_from_args(args, model)
                payload = backend.generate(prompt)
                accepted, rejected = verify_claims(payload, facts, evidence, max_claims=args.max_claims)
                if len(payload["claims"]) > args.max_claims:
                    raise ValueError("模型输出超过本次结论数量上限。")
                row.update({"completed": True, "metrics": selection_metrics(payload, accepted, rejected, aspects, labels),
                            "generation": backend.describe().get("last_generation"),
                            "payload": payload, "accepted_claims": accepted, "rejected_claims": rejected})
                print(f"完成：{model} / {version}，提议 {len(payload['claims'])}，通过 {len(accepted)}，拒绝 {len(rejected)}", flush=True)
            except (ValueError, RuntimeError, OSError) as error:
                failed = True
                row["error"] = str(error)
                print(f"运行失败：{model} / {version}：{error}。请检查服务和模型后重试。", flush=True)
            row["wall_seconds"] = time.perf_counter() - started
            write_json(args.output_dir/f"{name}_{version}.json", row)
            summary["results"].append(row)
            write_json(args.output_dir/"summary.json", summary)
            (args.output_dir/"summary.md").write_text(render_summary(summary), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as error:
        print(f"对照测试未完成：{error}。请检查证据文件、评分表和 Ollama 服务后重试。", file=sys.stderr)
        raise SystemExit(1)
