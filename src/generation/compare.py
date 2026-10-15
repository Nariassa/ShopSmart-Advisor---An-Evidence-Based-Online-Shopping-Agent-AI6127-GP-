"""Run the same real evidence through both installed local models, sequentially."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

from .__main__ import ROOT, read_records
from .adapters import ParquetEvidenceStore
from .batch import citation_audit
from .generator import ShopSmartGenerator
from .llm import add_ollama_arguments, backend_from_args


def safe_model_name(model):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', model)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run Qwen and Llama through Grounded Generation & Purchase Confidence Scoring with identical evidence')
    parser.add_argument('--models', nargs='+', default=['qwen2.5:7b', 'llama3.1:8b'])
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--product', action='append', help='Real ASIN; default B09KH866S8')
    source.add_argument('--evidence', type=Path, help='Retrieval, Reranking & Evaluation evidence JSON/JSONL')
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent)
    parser.add_argument('--product-limit', type=int, default=40)
    parser.add_argument('--query', default='Is this key finder reliable over Bluetooth and easy to set up?')
    parser.add_argument('--aspects', default='connectivity;ease of use')
    parser.add_argument('--max-facts', type=int, default=18)
    parser.add_argument('--max-claims', type=int, default=8)
    parser.add_argument('--max-context-chars', type=int, default=8000)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs'/'local_model_integration')
    add_ollama_arguments(parser, include_model=False)
    parser.add_argument('--prompt-version', choices=('v1', 'v2', 'v3'), default='v1')
    parser.add_argument('--nli-model', help='Local English NLI model for single-fact rewrites')
    args = parser.parse_args(argv)
    names = [safe_model_name(model) for model in args.models]
    if len(set(names)) != len(names):
        raise ValueError('模型名称重复或输出文件名冲突，请修正 --models。')
    if args.evidence:
        records = read_records(args.evidence)
        if any(not row.get('text') for row in records):
            records = ParquetEvidenceStore(args.data_dir/'chunks.parquet').hydrate(records)
        origin = {'kind': 'evidence_file', 'path': str(args.evidence.resolve())}
    else:
        asins = args.product or ['B09KH866S8']
        store = ParquetEvidenceStore(args.data_dir/'chunks.parquet')
        records = [row for asin in asins for row in store.product_chunks(asin, args.product_limit)]
        origin = {'kind': 'parquet_product_sample', 'product_ids': asins,
                  'path': str(store.path), 'limit_per_product': args.product_limit}
    if not records:
        raise ValueError('没有找到真实证据，请检查 ASIN 或 --evidence 文件。')
    aspects = [aspect.strip() for aspect in args.aspects.split(';') if aspect.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = json.dumps(records, ensure_ascii=False, sort_keys=True, indent=2)
    evidence_path = args.output_dir/'input_evidence.json'
    evidence_path.write_text(snapshot, encoding='utf-8')
    evaluation_time = datetime.now(timezone.utc)
    summary = {'query': args.query, 'requested_aspects': aspects, 'input_source': origin,
               'evidence_records': len(records), 'evidence_sha256': hashlib.sha256(snapshot.encode()).hexdigest(),
               'evaluation_time_utc': evaluation_time.isoformat(),
               'scope': 'generation_integration_smoke_test_not_formal_evaluation', 'results': []}
    failed = False
    verifier = None
    if args.nli_model:
        from .grounding import NLIVerifier
        verifier = NLIVerifier(args.nli_model)
    for model, name in zip(args.models, names):
        print(f'运行本地模型：{model}', flush=True)
        try:
            backend = backend_from_args(args, model)
            generator = ShopSmartGenerator(llm=backend, allow_fallback=False,
                                           grounding_verifier=verifier,
                                           max_facts=args.max_facts, max_claims=args.max_claims,
                                           max_context_chars=args.max_context_chars, prompt_version=args.prompt_version)
            result = generator.generate_recommendation(args.query, records,
                                                        requested_aspects=aspects, now=evaluation_time)
            audit = citation_audit(result)
            if result['backend'] != 'llm' or not result['grounding']['llm_verified_claims']:
                raise RuntimeError('模型没有产出通过证据检查的结论。')
            if audit['citation_traceability'] != 1:
                raise RuntimeError('引用未完全通过来源与原文位置核对。')
            result.update({'input_source': origin, 'mechanical_audit': audit,
                           'evidence_sha256': summary['evidence_sha256']})
            target = args.output_dir/(name+'.json')
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            target.with_suffix('.md').write_text(result['answer'], encoding='utf-8')
            row = {'model': model, 'passed': True, 'backend': result['backend'],
                   'llm_verified_claims': result['grounding']['llm_verified_claims'],
                   'final_claims': len(result['claims']), 'rejected_claims': len(result['grounding']['rejected']),
                   'citation_count': audit['citation_count'],
                   'citation_traceability': audit['citation_traceability'],
                   'pcs_by_product': {p['parent_asin']: p['pcs']['score'] for p in result['products']},
                   'total_seconds': result['timing']['total_seconds'],
                   'generation': result['llm']['last_generation'],
                   'output_json': str(target.resolve())}
            summary['results'].append(row)
            print(f"通过：{model}；模型结论 {row['llm_verified_claims']} 条，"
                  f"最终引用 {row['citation_count']} 条，耗时 {row['total_seconds']:.2f} 秒。", flush=True)
        except (ValueError, RuntimeError, OSError) as error:
            failed = True
            summary['results'].append({'model': model, 'passed': False, 'error': str(error)})
            print(f'模型 {model} 未跑通：{error}', flush=True)
        (args.output_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 两个本地模型接入结果', '', f'问题：{args.query}', '',
             f'相同输入证据：{len(records)} 条；证据 SHA-256：{summary["evidence_sha256"]}', '',
             '| 模型 | 是否跑通 | 模型结论通过检查 | 最终引用通过检查 | PCS | 总耗时（含首次模型加载） |',
             '| --- | --- | ---: | --- | --- | --- |']
    for row in summary['results']:
        if row['passed']:
            scores = ', '.join(f'{asin}: {score:.2f}' for asin, score in row['pcs_by_product'].items())
            lines.append(f"| {row['model']} | 是 | {row['llm_verified_claims']} | "
                         f"{row['citation_count']}/{row['citation_count']} | {scores} | {row['total_seconds']:.2f}s |")
        else:
            lines.append(f"| {row['model']} | 否 | — | — | — | — |")
    lines += ['', '本次仅验证真实证据进入生成、引用核对和 PCS 的接入流程。'
              '默认规则会补充代表性正反证据；模型通过的结论数单独记录。'
              'PCS 由规则计算，两个模型使用相同证据和计算时间，分数应相同。'
              '此处的引用检查率仅表示来源及原文位置一致，不代表人工评估准确率。', '',
              '默认 keep_alive=0，完成各次生成后释放模型，下载文件保留。']
    (args.output_dir/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    return 1 if failed else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as error:
        print(f'接入运行未完成：{error}\n请检查 Ollama 服务、模型名称及真实证据路径后重试。', file=sys.stderr)
        raise SystemExit(1)
