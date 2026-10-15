"""Post-run comparison of frozen old/new development batches; no generation changes."""
import csv
import json
from collections import Counter
from pathlib import Path
from html import unescape
import re

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'outputs/member_c_generation_testing_2026-10-15'
NEW = ROOT/'outputs/member_c_generation_testing_revised_final_2026-10-15'


def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def report():
    old, new = rows(OLD/'results_all.jsonl'), rows(NEW/'results_all.jsonl')
    om, nm = read(OLD/'manifest.json'), read(NEW/'manifest.json')
    old_by_key = {(r['model'],r['query_id']):r for r in old}
    checks = {'same_raw_candidates':om['evidence_map_sha256']==nm['evidence_map_sha256'],
              'all_40_attempts_present':len(new)==len(old)==40,
              'same_queries':om['queries_sha256']==nm['queries_sha256'],
              'same_pcs_time':om['pcs_time_utc']==nm['pcs_time_utc'],
              'same_model_weights':read(OLD/'model_digests.json')==read(NEW/'model_digests.json')}
    pairs, summaries = [], []
    for r in new:
        before = old_by_key[(r['model'],r['query_id'])]
        old_products = {p['parent_asin']:p for p in before.get('result',{}).get('products',[])}
        new_products = {p['parent_asin']:p for p in r.get('result',{}).get('products',[])}
        common = old_products.keys() & new_products.keys()
        same_pcs = all(old_products[k]['pcs']==new_products[k]['pcs'] for k in common)
        pairs.append({'model':r['model'],'query_id':r['query_id'], 'baseline_status':before['run_status'],
            'revised_status':r['run_status'], 'same_raw_evidence':before['evidence_sha256']==r['evidence_sha256'],
            'same_actual_facts':before.get('actual_input_facts')==r.get('actual_input_facts'),
            'common_products':len(common), 'common_product_pcs_equal':same_pcs,
            'before_products':list(old_products), 'after_products':list(new_products)})
    for model in nm['models']:
        entry = {'model':model}
        for label,data in [('before',old),('after',new)]:
            subset = [r for r in data if r['model']==model]
            success = [r for r in subset if r['run_status']=='generated']
            claims = [c for r in success for c in r['result']['claims']]
            reasons = Counter(v['reason'] for r in success for v in r['result']['grounding']['rejected'])
            entry[label] = {'attempts':len(subset),'generated':len(success),
                'statuses':dict(Counter(r['run_status'] for r in subset)),
                'citations':sum(len(c['citations']) for c in claims),
                'verified_rewrites':sum(r['result']['grounding']['rewrite_verified_claims'] for r in success),
                'nli_rejections':reasons.get('nli_not_entailed',0),
                'guard_rejections':sum(v.get('stage')=='rewrite_guard' for r in success for v in r['result']['grounding']['rejected']),
                'rejection_reasons':dict(reasons),
                'product_name_mismatches':reasons.get('product_name_mismatch',0),
                'html_fragment_answers':sum(bool(re.search(r'<br|&lt;br|/\s*&gt;',r['result']['answer'],re.I)) for r in success),
                'speaker_case_in_Q003':any(p['parent_asin']=='B08HHY9FNK' for r in success if r['query_id']=='Q003' for p in r['result']['products']),
                'attributed_final_rewrites':sum(c['text'].lower().startswith('one reviewer reports that ') for c in claims if c['grounding_method']=='nli_entailment'),
                'final_rewrites':sum(c['grounding_method']=='nli_entailment' for c in claims),
                'missing_aspect_tag_flags':sum('missing_requested_aspect_tags_proxy' in r['audit']['review_flags'] for r in success)}
        summaries.append(entry)
    checks['all_raw_evidence_pairs_equal'] = all(p['same_raw_evidence'] for p in pairs)
    checks['common_product_pcs_unchanged'] = all(p['common_product_pcs_equal'] for p in pairs)
    # Audit every actual transport schema, runtime count, attributed rewrite and gate output.
    revised = [r for r in new if r['run_status']=='generated']
    checks['all_call_limits_equal_8'] = all(call['schema']['properties'][key]['maxItems']==8 and len(call.get('payload',{}).get(key,[]))<=8 for r in revised for call in r['calls'] for key in call['schema']['properties'])
    checks['excluded_products_never_reinserted'] = all(not ({d['parent_asin'] for d in r['result']['product_filter']['excluded_products']} & ({p['parent_asin'] for p in r['result']['products']} | {c['parent_asin'] for c in r['result']['claims']})) for r in revised)
    checks['all_final_review_rewrites_attributed'] = all(c['text'].startswith('One reviewer reports that ') for r in revised for c in r['result']['claims'] if c['grounding_method']=='nli_entailment' and c['citations'][0]['chunk_type']=='review')
    checks['one_final_conclusion_per_fact'] = all(len({c['fact_id'] for c in r['result']['claims']})==len(r['result']['claims']) and all(len(c['citations'])==1 for c in r['result']['claims']) for r in revised)
    comparison = {'checks':checks,'models':summaries,'pairs':pairs,
        'interpretation':'Development paired full-pipeline regression. Raw candidates identical; actual facts and products may change because filtering is part of the treatment. Not an isolated prompt ablation or human quality estimate.'}
    (NEW/'revision_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf-8')
    with (NEW/'revision_pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
    lines=['# 四项修复后的配对重测', '',
        '同一批 20 个问题、791 条原始候选、候选顺序、模型权重、解码参数和 PCS 时间。商品过滤、选择提示、Schema、改写守卫和显示层共同变化，因此是整条生成流程的修复回归，不是单独 Prompt 效果消融。过滤会改变实际给模型的事实和最终商品，详见 revision_pairs.csv。', '',
        '| 模型 | 成功生成（前→后） | 名称拒绝（前→后） | HTML 残片回答（前→后） | 改写通过（前→后） | 新增守卫拒绝 |',
        '| --- | --- | --- | --- | --- | --- |']
    for row in summaries:
        a,b=row['before'],row['after']
        lines.append(f'| {row["model"]} | {a["generated"]}/20 → {b["generated"]}/20 | {a["product_name_mismatches"]} → {b["product_name_mismatches"]} | {a["html_fragment_answers"]} → {b["html_fragment_answers"]} | {a["verified_rewrites"]} → {b["verified_rewrites"]} | {b["guard_rejections"]} |')
    lines += ['', '## 修复验证', '']
    lines += ['- '+name+': '+str(value) for name,value in checks.items()]
    lines += ['', '## 解读与限制', '',
        '- 新增守卫拒绝不等于发现同等数量的幻觉：守卫刻意保守，可能拒绝语义正确的省略或同义替换；被拒绝时保留原句及来源。',
        '- 名称只归一化 HTML 实体、Unicode NFC 和空白，保留所有词语、型号和后缀，不接受缩略名称。',
        '- 商品类型通过查询和标题规则判断，未训练分类器。未知类型不自动声称已验证，保守规则可能误排合法商品或漏掉不匹配品。功能、兼容性、耐久性等需要原文支持及人工复核。',
        '- 个人归属、数字、显式否定和条件检查是必要条件启发式，不构成完整逻辑等价判断。NLI 阈值仍为 0.9，未通过提高阈值替代这些检查。',
        '- 方面标签遗漏线索可能增加；这不是语义覆盖率。请结合商品集合变化和实际原文做独立人评，不把所有维度解释为同步改善。',
        '- 存在已保存残留反例：Qwen Q004/F1 保留否定词和 unless，但弱化 not as good as SP3 的比较含义，NLI 仍通过。详见 QUALITY_REVIEW.md。',
        '- 显示层去除 HTML，结构化 claim.text、citation.quote、sources.text 和 start/end 保持原文。',
        '- 仍用未校准情感词典和方面标签；讽刺情感错误、方面覆盖低估、PCS 校准与 方面分析与购买决策解释模块 的正式接口本次未处理。',
        '- 缺少 检索与排序评估模块 的正式检索排名、独立测试集及人工语义评分；不能将成功率或引用核验率解释为回答准确率。', '']
    (NEW/'revision_comparison.md').write_text(chr(10).join(lines),encoding='utf-8')
    print(json.dumps({'checks':checks,'models':summaries},ensure_ascii=True,indent=2))
    return 0 if all(checks.values()) else 1


if __name__=='__main__': raise SystemExit(report())
