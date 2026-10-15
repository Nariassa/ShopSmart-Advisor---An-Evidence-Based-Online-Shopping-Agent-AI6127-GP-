"""Reproducible Grounded Generation & Purchase Confidence Scoring generation testing and System Integration, User Interface & Evaluation review handoff."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
from urllib.request import urlopen

from .__main__ import ROOT, read_records
from .adapters import ParquetEvidenceStore
from .batch import citation_audit
from .compare import safe_model_name
from .evidence import canonical_aspect
from .generator import ShopSmartGenerator, infer_constraints
from .grounding import NLIVerifier, uses_latin_letters
from .llm import OllamaBackend


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows), encoding='utf-8')


def prepare_inputs(queries, annotations, store):
    """All candidates, not gold-positive candidates; no annotation fields enter evidence."""
    ids = [query['query_id'] for query in queries]
    if len(ids) != len(set(ids)):
        raise ValueError('query_id 重复，请先修正查询文件。')
    if set(row['query_id'] for row in annotations) != set(ids):
        raise ValueError('标注候选与问题集合不一致，请补齐对应 ID。')
    pairs = [(row['query_id'], row['chunk_id']) for row in annotations]
    if len(pairs) != len(set(pairs)):
        raise ValueError('存在重复的问题—片段对，请先核对输入。')
    documents = {row['chunk_id']: row for row in store.lookup([r['chunk_id'] for r in annotations])}
    matched = 0
    for row in annotations:
        actual = hashlib.sha256(documents[row['chunk_id']]['text'].encode('utf-8')).hexdigest()
        if actual != row['candidate_text_sha256']:
            raise ValueError(f"原文 SHA-256 不匹配：{row['query_id']} / {row['chunk_id']}。请核对数据版本。")
        matched += 1
    # A strict whitelist prevents future Parquet columns from carrying human labels.
    allowed = {'chunk_id', 'parent_asin', 'chunk_type', 'aspect', 'text', 'rating', 'category', 'title',
               'timestamp', 'verified_purchase', 'helpful_vote', 'source_review_id', 'price', 'currency'}
    mapping, order = {}, {}
    for query in queries:
        qid = query['query_id']
        candidates = sorted((r for r in annotations if r['query_id']==qid),
                            key=lambda r: (r['display_order'], r['chunk_id']))
        if len({r['display_order'] for r in candidates}) != len(candidates):
            raise ValueError(f'{qid} 的展示顺序重复，无法明确回放顺序。')
        mapping[qid] = [{k:v for k,v in documents[r['chunk_id']].items() if k in allowed} for r in candidates]
        order[qid] = [{'chunk_id':r['chunk_id'], 'display_order':r['display_order']} for r in candidates]
    return mapping, {'annotation_records': len(annotations), 'unique_chunks': len(documents),
                     'original_text_sha256_matches': matched, 'candidate_display_order': order}


class RecordedBackend(OllamaBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def generate_structured(self, prompt, schema):
        call = {'prompt': prompt, 'schema': schema, 'started_utc': datetime.now(timezone.utc).isoformat()}
        self.calls.append(call)
        started = time.perf_counter()
        try:
            payload = super().generate_structured(prompt, schema)
            call.update({'payload': payload, 'generation': self.last_generation, 'completed': True})
            return payload
        except Exception as error:
            call.update({'completed': False, 'error': str(error)})
            raise
        finally:
            call['wall_seconds'] = time.perf_counter()-started


def input_facts(calls):
    return json.loads(calls[0]['prompt'].splitlines()[-1].removeprefix('EVIDENCE_FACTS_JSON: ')) if calls else []


def inspect_result(result, calls):
    audit = citation_audit(result)
    facts = input_facts(calls)
    selected_ids = set(result['grounding']['selected_fact_ids'])
    final_ids = {c['fact_id'] for c in result['claims']}
    final_by_aspect = {}
    available_by_aspect = {}
    for fact in facts:
        if fact['source_type'] != 'review':
            continue
        key = (fact['product_id'], fact['aspect'])
        available_by_aspect.setdefault(key, set()).add(fact['sentiment'])
        if fact['fact_id'] in final_ids:
            final_by_aspect.setdefault(key, set()).add(fact['sentiment'])
    omissions = [{'parent_asin': key[0], 'aspect': key[1],
                  'missing_polarities': sorted({'positive','negative'}-final_by_aspect.get(key,set()))}
                 for key, polarities in available_by_aspect.items()
                 if {'positive','negative'} <= polarities
                 and not {'positive','negative'} <= final_by_aspect.get(key,set())]
    requested = set(result['requested_aspects'])
    covered = {fact['aspect'] for fact in facts if fact['fact_id'] in final_ids} & requested
    flags = []
    if omissions: flags.append('opposing_review_omission_lexicon_proxy')
    if requested-covered: flags.append('missing_requested_aspect_tags_proxy')
    if result['grounding']['selection_rejected']: flags.append('selection_proposals_rejected')
    if any(r.get('reason')=='nli_not_entailed' for r in result['grounding']['rejected']): flags.append('rewrite_rejected_original_retained')
    if any(r.get('stage')=='rewrite_guard' for r in result['grounding']['rejected']): flags.append('rewrite_safety_guard_original_retained')
    if any(w['code']=='rewrite_fallback' for w in result['warnings']): flags.append('rewrite_stage_failed_original_retained')
    if not all(uses_latin_letters(c['text']) for c in result['claims']): flags.append('non_latin_conclusion')
    if audit['citation_count'] and audit['citation_traceability'] != 1: flags.append('citation_error')
    if any(p['verdict']=='needs_sentiment_validation' for p in result['products']): flags.append('sentiment_baseline_unvalidated')
    return {**audit, 'review_flags': flags, 'opposing_omissions_lexicon_proxy': omissions,
            'requested_aspect_tag_coverage_proxy': len(covered)/len(requested) if requested else None,
            'missing_requested_aspect_tags_proxy': sorted(requested-covered),
            'final_model_selected_fact_count': len(final_ids & selected_ids),
            'rule_supplemented_fact_count': len(final_ids-selected_ids),
            'english_script_check': all(uses_latin_letters(c['text']) for c in result['claims']),
            'scope': 'mechanical_and_tag_proxies_not_human_generation_quality'}


def reusable_record(row, manifest, evidence_hash, model, qid):
    return (row.get('manifest_sha256')==sha(manifest) and row.get('evidence_sha256')==evidence_hash
            and row.get('model')==model and row.get('query_id')==qid)


def csv_file(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def export_package(directory, manifest, queries, mapping, records):
    successful = [row for row in records if row['run_status']=='generated']
    write_jsonl(directory/'results_all.jsonl', records)
    for model in manifest['models']:
        subset = [row for row in records if row['model']==model]
        write_jsonl(directory/f'{safe_model_name(model)}_results.jsonl', subset)
        pages = []
        for row in subset:
            pages += [f"# {row['query_id']}: {row['query']}", '', row.get('result', {}).get('answer',
                       'Run did not produce an LLM answer. See the structured failure record.'), '']
        (directory/f'{safe_model_name(model)}_answers.md').write_text(chr(10).join(pages), encoding='utf-8')
    ragas = []
    for row in successful:
        result = row['result']
        item = {'query_id':row['query_id'], 'model':row['model'], 'user_input':row['query'],
                'response':result['answer'], 'retrieved_contexts':[fact['quote'] for fact in row['actual_input_facts']],
                'condition':'annotation_candidate_replay_development', 'evaluation_id':row['evaluation_id']}
        query = next(q for q in queries if q['query_id']==row['query_id'])
        if query.get('reference'): item['reference'] = query['reference']
        ragas.append(item)
    write_jsonl(directory/'ragas_inputs.jsonl', ragas)
    write_jsonl(directory/'failures_and_abstentions.jsonl', [row for row in records if row['run_status']!='generated'])
    # Review packets intentionally omit model names; keep the mapping separate for System Integration, User Interface & Evaluation.
    shuffled = list(successful)
    random.Random(6127).shuffle(shuffled)
    review, keys, claim_rows = [], [], []
    for row in shuffled:
        eid, result = row['evaluation_id'], row['result']
        review.append({'evaluation_id':eid, 'query_id':row['query_id'], 'query':row['query'],
                       'requested_aspects':result['requested_aspects'], 'constraints':result['constraints'],
                       'answer':result['answer'], 'claims':result['claims'], 'sources':result['sources'],
                       'actual_input_facts':row['actual_input_facts']})
        keys.append({'evaluation_id':eid,'query_id':row['query_id'],'model':row['model']})
        for index, claim in enumerate(result['claims'], 1):
            claim_rows.append({'evaluation_id':eid,'query_id':row['query_id'],'claim_number':index,
                'claim_text':claim['text'], 'product_name':claim['product_name'],
                'source_quotes':chr(10).join(c['quote'] for c in claim['citations']),
                'chunk_ids':';'.join(c['chunk_id'] for c in claim['citations']),
                'reviewer_id':'', 'support_label':'', 'reason':''})
    write_jsonl(directory/'blinded_review_packets.jsonl', review)
    pages = []
    for packet in review:
        pages += [f"# {packet['evaluation_id']} / {packet['query_id']}", '', packet['query'], '', packet['answer'], '',
                  '## Source evidence for review', '']
        for source_id, source in packet['sources'].items():
            pages += [f"### {source_id} / {source['chunk_id']}", '', source['text'], '']
    (directory/'blinded_review_answers.md').write_text(chr(10).join(pages), encoding='utf-8')
    csv_file(directory/'model_key_for_E.csv', ['evaluation_id','query_id','model'], keys)
    fields = ['evaluation_id','query_id','reviewer_id','relevance_1_to_5','faithfulness_1_to_5',
              'completeness_1_to_5','balance_1_to_5_or_NA','english_clarity_1_to_5','notes']
    csv_file(directory/'human_review_template.csv', fields,
             [{**{field:'' for field in fields}, 'evaluation_id':r['evaluation_id'],'query_id':r['query_id']} for r in shuffled])
    csv_file(directory/'claim_support_review_template.csv', ['evaluation_id','query_id','claim_number','claim_text',
             'product_name','source_quotes','chunk_ids','reviewer_id','support_label','reason'], claim_rows)
    timing = [row['wall_seconds'] for row in successful]
    flags = Counter(flag for row in successful for flag in row['audit']['review_flags'])
    citation_total = sum(row['audit']['citation_count'] for row in successful)
    citation_valid = sum(row['audit']['valid_citation_count'] for row in successful)
    model_summaries = []
    for model in manifest['models']:
        subset = [row for row in successful if row['model']==model]
        model_summaries.append({'model':model,'generated':len(subset),
            'selected':sum(r['result']['grounding']['llm_verified_selections'] for r in subset),
            'selection_rejected':sum(len(r['result']['grounding']['selection_rejected']) for r in subset),
            'verified_rewrites':sum(r['result']['grounding']['rewrite_verified_claims'] for r in subset),
            'rejected_rewrites':sum(sum(v.get('reason')=='nli_not_entailed' for v in r['result']['grounding']['rejected']) for r in subset),
            'guard_rejected_rewrites':sum(sum(v.get('stage')=='rewrite_guard' for v in r['result']['grounding']['rejected']) for r in subset),
            'excluded_products':sum(len(r['result'].get('product_filter',{}).get('excluded_products',[])) for r in subset),
            'citation_count':sum(r['audit']['citation_count'] for r in subset),
            'valid_citations':sum(r['audit']['valid_citation_count'] for r in subset),
            'mean_wall_seconds':statistics.mean(r['wall_seconds'] for r in subset) if subset else None,
            'prompt_tokens':sum(sum((c.get('generation',{}).get('prompt_tokens') or 0) for c in r['calls']) for r in subset),
            'generated_tokens':sum(sum((c.get('generation',{}).get('generated_tokens') or 0) for c in r['calls']) for r in subset)})
    paired = []
    for query in queries:
        subset = [r for r in successful if r['query_id']==query['query_id']]
        if len(subset)==len(manifest['models']):
            paired.append({'query_id':query['query_id'],
                'same_evidence':len({r['evidence_sha256'] for r in subset})==1,
                'same_selection_prompt':len({sha(r['calls'][0]['prompt']) for r in subset})==1,
                'same_pcs':len({sha({p['parent_asin']:p['pcs'] for p in r['result']['products']}) for r in subset})==1})
    save(directory/'paired_input_checks.json', paired)
    summary = {'condition':manifest['condition'], 'planned_runs':len(queries)*len(manifest['models']),
        'recorded_runs':len(records),'generated_runs':len(successful),
        'run_status_counts':dict(Counter(row['run_status'] for row in records)),
        'citation_count':citation_total, 'valid_citations':citation_valid,
        'citation_traceability':citation_valid/citation_total if citation_total else None,
        'review_flag_counts':dict(flags),'models':model_summaries,'paired_checks':paired,
        'mean_wall_seconds':statistics.mean(timing) if timing else None,
        'human_evaluation_completed':False,'ragas_metrics_computed':False, 'independent_test_set':False}
    save(directory/'summary.json', summary)
    issue_rows = [{'evaluation_id':r['evaluation_id'],'query_id':r['query_id'],'model':r['model'],
                   'run_status':r['run_status'],'flags':';'.join(r.get('audit',{}).get('review_flags',[])),
                   'error':r.get('error','')} for r in records]
    csv_file(directory/'automatic_review_flags.csv', ['evaluation_id','query_id','model','run_status','flags','error'], issue_rows)
    lines = ['# Grounded Generation & Purchase Confidence Scoring Generation Testing', '',
        '本批为固定标注候选池回放的开发生成测试，不是新检索排名或独立正式评估。', '',
        f"计划 {summary['planned_runs']} 次；已记录 {len(records)} 次；产生真实模型回答 {len(successful)} 次。",
        f"来源及原文位置核验：{citation_valid}/{citation_total}。此比例不代表语义正确率。", '',
        '| 模型 | 生成回答 | 选择通过 | 选择拒绝 | 改写通过 | NLI 拒绝改写 | 引用核验 | 平均耗时 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for row in model_summaries:
        elapsed = f"{row['mean_wall_seconds']:.2f}s" if row['mean_wall_seconds'] is not None else '—'
        lines.append(f"| {row['model']} | {row['generated']} | {row['selected']} | {row['selection_rejected']} | {row['verified_rewrites']} | {row['rejected_rewrites']} | {row['valid_citations']}/{row['citation_count']} | {elapsed} |")
    lines += ['', '## 已完成与待 系统集成、交互与综合评估模块 完成', '',
        '- 已完成：同问题同证据生成、英文单事实改写、NLI 核验、引用核验、逐题运行记录和盲评材料。',
        '- 待系统集成、交互与综合评估模块：人工相关性、支持性、完整性、平衡及表达评分；IAA；适用的 RAGAS 指标计算。模板留空，不伪造人评分数。',
        '- 自动方面覆盖和正反保留检查使用方面标签及未校准情感词典，只是待复核线索。', '', '## 运行范围', '',
        '全部 791 条候选按原 display_order 回放，原文从 Parquet 回填并逐条核对 SHA-256。',
        '没有使用 relevance_label、confidence 或 reason 选择/重排证据，也没有把这些标注送入模型。',
        '候选池的原始检索混合方法和排序未知，display_order 只表示标注展示顺序，不能当作 检索与排序评估模块 的检索排名。',
        '没有 reference 的问题不添加参考答案。RAGAS 输入的 contexts 为实际发给选择模型的事实句，完整候选原文另行保留。',
        '模型参数、证据顺序、选择 Prompt 和 PCS 时间固定；模型后续选择与改写可能不同。',
        '性能统计包含每模型首次冷加载和后续常驻调用，不是正式系统吞吐量基准。',
        '有错误会保留逐题失败记录；程序补入的证据与模型选择分开计数。', '', '## 自动复核线索', '']
    lines += [f'- {name}: {count} 个回答' for name,count in flags.items()]
    (directory/'summary.md').write_text(chr(10).join(lines)+chr(10), encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generation Testing for System Integration, User Interface & Evaluation')
    parser.add_argument('--queries', type=Path, default=ROOT/'evaluation/formal_queries_v1.jsonl')
    parser.add_argument('--annotations', type=Path, default=ROOT.parent/'human_validated_annotations_v1.jsonl')
    parser.add_argument('--chunks', type=Path, default=ROOT.parent/'chunks.parquet')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/member_c_generation_testing_revised_final_2026-10-15')
    parser.add_argument('--baseline-dir', type=Path, default=ROOT/'outputs/member_c_generation_testing_2026-10-15')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    queries = read_records(args.queries)
    manifest_path = directory/'manifest.json'
    config = {'models':['qwen2.5:7b','llama3.1:8b'],'prompt_version':'v1','max_products':3,
              'max_facts':18,'max_claims':8,'max_context_chars':8000,'num_ctx':8192,'num_predict':2048,
              'temperature':0,'seed':42,'keep_alive':'2m','llm_timeout':180,
              'nli_model':'cross-encoder/nli-deberta-v3-small','nli_threshold':0.9,
              'generation_revision':'bounded_type_scope_display_v3_attribution_aware_nli'}
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest['config'] != config or manifest['queries_sha256'] != sha(queries):
            raise ValueError('当前配置与已冻结测试不一致，请使用新的输出目录。')
        mapping = read(directory/'evidence_map.json')
        if sha(mapping) != manifest['evidence_map_sha256']:
            raise ValueError('冻结证据发生变化，请核对文件后重试。')
    else:
        baseline = read(args.baseline_dir/'manifest.json')
        mapping = read(args.baseline_dir/'evidence_map.json')
        provenance = read(args.baseline_dir/'input_provenance.json')
        if sha(queries) != baseline['queries_sha256'] or sha(mapping) != baseline['evidence_map_sha256']:
            raise ValueError('基线问题或证据哈希不一致，无法进行配对重测。')
        print('复用基线冻结的全部 791 条候选；保持问题、顺序、原文和 PCS 时间一致。', flush=True)
        save(directory/'evidence_map.json', mapping)
        write_jsonl(directory/'queries.jsonl', queries)
        save(directory/'input_provenance.json', provenance)
        code_files = sorted((ROOT/'src/generation').glob('*.py'))
        hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files}
        for path in code_files:
            target = directory/'code_snapshot'/path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        manifest = {'condition':'annotation_candidate_replay_development_not_independent_formal_eval',
            'created_utc':datetime.now(timezone.utc).isoformat(), 'pcs_time_utc':baseline['pcs_time_utc'],
            'baseline_directory':str(args.baseline_dir.resolve()), 'baseline_manifest_sha256':sha(baseline),
            'models':config['models'], 'config':config,'queries_sha256':sha(queries),
            'evidence_map_sha256':sha(mapping),'annotations_file_sha256':hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
            'chunks_path':str(args.chunks.resolve()),'code_file_sha256':hashes,
            'annotation_labels_passed_to_llm':False,'display_order_is_retrieval_rank':False}
        save(manifest_path, manifest)
    records = []
    if args.prepare_only:
        print(f'证据准备完成：{len(queries)} 问题，{sum(len(v) for v in mapping.values())} 条问题—证据记录。', flush=True)
        return 0
    for name, expected in manifest['code_file_sha256'].items():
        if hashlib.sha256((ROOT/'src/generation'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('代码与冻结版本不一致，请新建测试目录，避免混合不同版本结果。')
    import torch
    torch.set_num_threads(4)
    verifier = NLIVerifier(config['nli_model'], config['nli_threshold'])
    now = datetime.fromisoformat(manifest['pcs_time_utc'])
    with urlopen('http://localhost:11434/api/tags', timeout=15) as response:
        tags = json.load(response)
    model_digests = {r['name']:r['digest'] for r in tags['models'] if r['name'] in config['models']}
    if set(model_digests)!=set(config['models']):
        raise ValueError('本机缺少所需模型，请核对 Ollama 模型列表。')
    if (directory/'model_digests.json').exists() and read(directory/'model_digests.json')!=model_digests:
        raise ValueError('模型权重与检查点不同，请使用新的输出目录。')
    save(directory/'model_digests.json', model_digests)
    save(directory/'ollama_models.json', tags)
    import importlib.metadata
    save(directory/'runtime_versions.json', {
        'packages':{name:importlib.metadata.version(name) for name in ['torch','transformers','pyarrow','sentencepiece']},
        'nli_model':config['nli_model'],'nli_commit':getattr(verifier.model.config,'_commit_hash',None),
        'nli_device':str(verifier.model.device),'torch_threads':torch.get_num_threads()})
    with urlopen('http://localhost:11434/api/version', timeout=15) as response:
        save(directory/'ollama_version.json', json.load(response))
    for model in config['models']:
        backend = RecordedBackend(model=model, num_ctx=config['num_ctx'], num_predict=config['num_predict'],
                                   timeout=config['llm_timeout'], keep_alive=config['keep_alive'])
        generator = ShopSmartGenerator(llm=backend, grounding_verifier=verifier, allow_fallback=False,
            prompt_version=config['prompt_version'],max_products=config['max_products'],
            max_facts=config['max_facts'],max_claims=config['max_claims'],max_context_chars=config['max_context_chars'])
        for query in queries:
            qid = query['query_id']
            target = directory/'runs'/f'{safe_model_name(model)}_{qid}.json'
            evidence_hash = sha(mapping[qid])
            if target.exists():
                row = read(target)
                if not reusable_record(row, manifest, evidence_hash, model, qid):
                    raise ValueError('检查点与当前输入不一致，请使用新的输出目录。')
                records.append(row)
                print(f'保留已完成检查点：{model} / {qid} / {row["run_status"]}', flush=True)
                continue
            backend.calls = []
            row = {'model':model,'query_id':qid,'query':query['query'],
                'evaluation_id':'E'+hashlib.sha256((sha(manifest)+model+qid).encode()).hexdigest()[:12],
                'manifest_sha256':sha(manifest),'evidence_sha256':evidence_hash,'run_status':'failed'}
            print(f'运行：{model} / {qid}', flush=True)
            started = time.perf_counter()
            try:
                constraints = {**infer_constraints(query['query']), 'target_category':query.get('target_category')}
                aspects = [canonical_aspect(a) for a in query['target_aspects'].split(';') if a.strip()]
                result = generator.generate_recommendation(query['query'], mapping[qid],
                            requested_aspects=aspects, constraints=constraints, now=now)
                row['result'] = result
                row['audit'] = inspect_result(result, backend.calls)
                row['actual_input_facts'] = input_facts(backend.calls)
                if result['backend']=='llm' and result['grounding']['llm_verified_selections']:
                    row['run_status'] = 'generated'
                elif result['status']=='insufficient_evidence':
                    row['run_status'] = 'abstained_no_evidence'
                else:
                    row['run_status'] = 'non_llm_output'
                if 'citation_error' in row['audit']['review_flags']:
                    row['run_status'] = 'failed_citation_audit'
            except Exception as error:
                row['error'] = f'{type(error).__name__}: {error}'
                print(f'本题生成失败，已记录并继续其他问题：{model} / {qid}：{error}', flush=True)
            row['calls'] = backend.calls
            row['wall_seconds'] = time.perf_counter()-started
            save(target, row)
            records.append(row)
            export_package(directory, manifest, queries, mapping, records)
            print(f"完成：{model} / {qid} / {row['run_status']} / {row['wall_seconds']:.1f}s", flush=True)
        # Unload only the model used by this run before switching models.
        from urllib.request import Request
        request = Request('http://localhost:11434/api/generate',
                          data=json.dumps({'model':model,'keep_alive':0}).encode(),
                          headers={'Content-Type':'application/json'})
        with urlopen(request, timeout=60) as response:
            response.read()
    summary = export_package(directory, manifest, queries, mapping, records)
    return 1 if any(k.startswith('failed') or k=='non_llm_output' for k in summary['run_status_counts']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
