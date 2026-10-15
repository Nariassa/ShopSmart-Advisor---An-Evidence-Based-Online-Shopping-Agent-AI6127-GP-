"""Real local English NLI diagnostics and two-stage generation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from .__main__ import ROOT, read_records
from .batch import citation_audit
from .compare import safe_model_name
from .generator import ShopSmartGenerator
from .grounding import NLIVerifier, uses_latin_letters
from .llm import OllamaBackend

CASES = [
 ('setup_supported', 'The application is easy to install and the pairing process was simple.', 'Installing the application and pairing the device were easy.', True),
 ('setup_contradiction', 'The application is easy to install and the pairing process was simple.', 'Installing the application was difficult.', False),
 ('setup_unrelated', 'The application is easy to install and the pairing process was simple.', 'The battery lasts ten hours.', False),
 ('connection_supported', 'Worked brilliantly for a day and then could no longer connect to the app', 'The device stopped connecting to the app after one day.', True),
 ('connection_contradiction', 'Worked brilliantly for a day and then could no longer connect to the app', 'The device continued connecting reliably after one day.', False),
 ('connection_scope', 'My device could no longer connect to the app after one day.', 'Every device fails to connect after one day.', False),
 ('duration_supported', 'The battery lasted ten hours in my test.', 'The reviewer reported ten hours of battery life.', True),
 ('duration_changed', 'The battery lasted ten hours in my test.', 'The reviewer reported twenty hours of battery life.', False),
 ('price_supported', 'The product costs 49 dollars.', 'The price is 49 dollars.', True),
 ('price_changed', 'The product costs 49 dollars.', 'The price is 19 dollars.', False),
 ('comfort_supported', 'The headphones were comfortable during my flight.', 'The reviewer found the headphones comfortable on a flight.', True),
 ('comfort_unrelated', 'The headphones were comfortable during my flight.', 'The headphones support Bluetooth 5.3.', False),
]

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main(argv=None):
    parser = argparse.ArgumentParser(description='Real local NLI diagnostics and ID selection integration')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/id_grounding_nli_2026-10-13')
    parser.add_argument('--skip-live', action='store_true')
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verifier = NLIVerifier()
    diagnostics = []
    for name, premise, claim, expected in CASES:
        scores = verifier.score(premise, claim)
        diagnostics.append({'case_id': name, 'premise': premise, 'claim': claim,
                            'expected_supported': expected, 'scores': scores,
                            'matches_expected': scores['supported']==expected})
        print(f"NLI {name}: entailment={scores['entailment']:.4f}, accepted={scores['supported']}, expected={expected}", flush=True)
    summary = {'scope': 'development_diagnostics_not_formal_evaluation', 'nli_model': verifier.model_name,
               'threshold': verifier.threshold, 'device': str(verifier.model.device),
               'label_mapping': verifier.label_mapping, 'diagnostic_cases': len(diagnostics),
               'diagnostic_matches': sum(r['matches_expected'] for r in diagnostics),
               'false_accepts': sum(r['scores']['supported'] and not r['expected_supported'] for r in diagnostics),
               'false_rejects': sum(not r['scores']['supported'] and r['expected_supported'] for r in diagnostics),
               'label_source': 'assistant_authored_development_examples_not_independent_human_labels', 'live_results': []}
    save(args.output_dir/'nli_diagnostics.json', diagnostics)
    records = read_records(ROOT/'outputs/local_model_integration_2026-10-13/input_evidence.json')
    save(args.output_dir/'input_evidence.json', records)
    now = datetime.now(timezone.utc)
    failed = False
    if not args.skip_live:
        for model in ['qwen2.5:7b', 'llama3.1:8b']:
            for mode in ['id_only', 'id_plus_nli']:
                print(f'运行真实生成：{model} / {mode}', flush=True)
                generator = ShopSmartGenerator(llm=OllamaBackend(model, keep_alive=0),
                    grounding_verifier=verifier if mode=='id_plus_nli' else None, prompt_version='v1',
                    allow_fallback=False, max_facts=18, max_claims=8, max_context_chars=8000)
                row = {'model': model, 'mode': mode, 'passed': False}
                try:
                    result = generator.generate_recommendation(
                        'Is this key finder reliable over Bluetooth and easy to set up?', records,
                        requested_aspects=['connectivity', 'ease of use'], now=now)
                    audit = citation_audit(result)
                    if audit['citation_traceability'] != 1 or not result['grounding']['llm_verified_selections']:
                        raise RuntimeError('真实模型选择或引用核验未通过。')
                    if not all(uses_latin_letters(c['text']) for c in result['claims']):
                        raise RuntimeError('结论包含非英文文字。')
                    result['mechanical_audit'] = audit
                    target = args.output_dir/f'{safe_model_name(model)}_{mode}.json'
                    save(target, result)
                    target.with_suffix('.md').write_text(result['answer'], encoding='utf-8')
                    row.update({'passed': True, 'selected': result['grounding']['llm_verified_selections'],
                        'selection_rejected': len(result['grounding']['selection_rejected']),
                        'verified_rewrites': result['grounding']['rewrite_verified_claims'],
                        'nli_paraphrases': sum(c['grounding_method']=='nli_entailment' for c in result['claims']),
                        'rejected_proposals': len(result['grounding']['rejected']),
                        'final_citations': audit['citation_count'], 'citation_traceability': audit['citation_traceability'],
                        'total_seconds': result['timing']['total_seconds'], 'output_json': str(target.resolve())})
                    print(f"完成：{model} / {mode}，选择 {row['selected']}，改写通过 {row['verified_rewrites']}", flush=True)
                except (ValueError, RuntimeError, OSError) as error:
                    failed = True
                    row['error'] = str(error)
                    print(f'实测失败：{model} / {mode}：{error}。请检查本机服务后重试。', flush=True)
                summary['live_results'].append(row)
                save(args.output_dir/'summary.json', summary)
    save(args.output_dir/'summary.json', summary)
    lines = ['# ID、商品名称与英文 NLI 实测', '',
             f"NLI 模型：{summary['nli_model']}；本地设备：{summary['device']}；阈值：{summary['threshold']}。", '',
             f"开发诊断：{summary['diagnostic_matches']}/{summary['diagnostic_cases']} 符合预期；错误接受 {summary['false_accepts']}；错误拒绝 {summary['false_rejects']}。", '',
             '| 样例 | entailment | 预期支持 | 实际接受 |', '| --- | ---: | --- | --- |']
    for case in diagnostics:
        lines.append(f"| {case['case_id']} | {case['scores']['entailment']:.4f} | {case['expected_supported']} | {case['scores']['supported']} |")
    lines += ['', '| 模型 | 模式 | 选择通过 | 选择拒绝 | 改写通过 | 最终引用核验 | 总耗时 |',
              '| --- | --- | ---: | ---: | ---: | --- | ---: |']
    for row in summary['live_results']:
        if row['passed']:
            lines.append(f"| {row['model']} | {row['mode']} | {row['selected']} | {row['selection_rejected']} | {row['verified_rewrites']} | {row['final_citations']}/{row['final_citations']} | {row['total_seconds']:.2f}s |")
        else:
            lines.append(f"| {row['model']} | {row['mode']} | 失败 | — | — | — | — |")
    lines += ['', '选择仅返回 fact_id 与 product_name，商品名称与原始标题严格核对。原文和引用位置由程序提取。',
              '启用 NLI 后生成单事实英文改写；未通过的改写移除，原文引用保留。最终引用可能包括规则补入的证据。',
              'Llama 本例 F3 改写为 The device can connect to the app for a day.，entailment=0.0056，被拒绝并保留原文。',
              '所有结论和答案说明使用英文；商品名称及引用原文保留来源内容。',
              '12 条诊断样例由开发助手编写，不是独立人工评估或 791 条检索标注；没有用这组样例校准阈值。',
              '生成测试仅一道真实商品开发问题，每模型每模式一次。NLI 高分不等于人工认可，仍可能遗漏限定条件、失去评论归属或错误支持较宽泛改写。',
              '支持关系与需求相关性、完整保留正反体验是不同指标；本次没有测量正式生成准确率或模型优劣。', '']
    (args.output_dir/'summary.md').write_text(chr(10).join(lines), encoding='utf-8')
    return 1 if failed else 0

if __name__ == '__main__':
    raise SystemExit(main())
