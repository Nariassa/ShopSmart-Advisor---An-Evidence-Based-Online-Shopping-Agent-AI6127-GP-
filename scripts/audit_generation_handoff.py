"""Audit a frozen generation experiment without changing its checkpoints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def audit(directory):
    manifest = read(directory / 'manifest.json')
    queries = rows(directory / 'queries.jsonl')
    mapping = read(directory / 'evidence_map.json')
    records = rows(directory / 'results_all.jsonl')
    summary = read(directory / 'summary.json')
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)

    expected = {(m, q['query_id']) for m in manifest['models'] for q in queries}
    check('all_planned_attempts_present', {(r['model'], r['query_id']) for r in records} == expected and len(records) == len(expected))
    check('frozen_queries_unchanged', sha(queries) == manifest['queries_sha256'])
    check('frozen_evidence_unchanged', sha(mapping) == manifest['evidence_map_sha256'])
    check('frozen_code_snapshot_intact', all(hashlib.sha256((directory / 'code_snapshot' / name).read_bytes()).hexdigest() == digest for name, digest in manifest['code_file_sha256'].items()))
    check('checkpoint_manifest_and_evidence_match', all(r['manifest_sha256'] == sha(manifest) and r['evidence_sha256'] == sha(mapping[r['query_id']]) for r in records))
    checkpoints = [read(p) for p in (directory / 'runs').glob('*.json')]
    check('checkpoint_exports_identical', {sha(r) for r in records} == {sha(r) for r in checkpoints} and len(records) == len(checkpoints))
    generated = [r for r in records if r['run_status'] == 'generated']
    failed = [r for r in records if r['run_status'] != 'generated']
    check('summary_counts_match', summary['recorded_runs'] == len(records) and summary['generated_runs'] == len(generated))
    check('failures_preserved', rows(directory / 'failures_and_abstentions.jsonl') == failed)
    forbidden = {'relevance_label', 'confidence', 'reason', 'candidate_text_sha256'}
    check('no_gold_fields_in_evidence', all(not (set(doc) & forbidden) for docs in mapping.values() for doc in docs))
    check('no_gold_fields_in_model_facts', all(not (set(fact) & forbidden) for r in generated for fact in r['actual_input_facts']))
    citations = 0
    source_ok = True
    prompt_ok = True
    for record in generated:
        facts = json.loads(record['calls'][0]['prompt'].splitlines()[-1].removeprefix('EVIDENCE_FACTS_JSON: '))
        prompt_ok &= facts == record['actual_input_facts']
        source_map = {s['chunk_id']: s for s in mapping[record['query_id']]}
        by_fact = {f['fact_id']: f for f in facts}
        for claim in record['result']['claims']:
            fact = by_fact.get(claim['fact_id'])
            source_ok &= bool(fact and fact['product_id'] == claim['parent_asin'] and fact['product_name'] == claim['product_name'])
            for citation in claim['citations']:
                citations += 1
                source = record['result']['sources'][citation['source_id']]
                original = source_map.get(citation['chunk_id'])
                source_ok &= bool(original and original['text'] == source['text'] and original['parent_asin'] == claim['parent_asin'])
                source_ok &= source['text'][citation['start']:citation['end']] == citation['quote']
                source_ok &= bool(fact and fact['quote'] == citation['quote'])
    check('actual_facts_match_recorded_prompt', prompt_ok)
    check('all_citation_spans_and_product_identities_match_frozen_sources', source_ok)
    check('citation_count_matches_summary', citations == summary['citation_count'] == summary['valid_citations'])
    ragas = rows(directory / 'ragas_inputs.jsonl')
    by_id = {r['evaluation_id']: r for r in generated}
    qmap = {q['query_id']: q for q in queries}
    check('ragas_records_match_successful_runs', len(ragas) == len(generated) and {r['evaluation_id'] for r in ragas} == set(by_id))
    check('ragas_contexts_are_actual_model_facts', all(r['retrieved_contexts'] == [f['quote'] for f in by_id[r['evaluation_id']]['actual_input_facts']] and r['response'] == by_id[r['evaluation_id']]['result']['answer'] for r in ragas))
    check('no_fabricated_references', all(('reference' not in r) if not qmap[r['query_id']].get('reference') else r.get('reference') == qmap[r['query_id']]['reference'] for r in ragas))
    packets = rows(directory / 'blinded_review_packets.jsonl')
    check('blind_packet_ids_match', len(packets) == len(generated) and {r['evaluation_id'] for r in packets} == set(by_id))
    check('blind_packets_omit_model_identifiers', all('model' not in r and not any(m in json.dumps(r) for m in manifest['models']) for r in packets))
    with (directory / 'model_key_for_E.csv').open(encoding='utf-8-sig', newline='') as f:
        keys = list(csv.DictReader(f))
    check('blind_mapping_matches_runs', len(keys) == len(generated) and all(k['evaluation_id'] in by_id and k['model'] == by_id[k['evaluation_id']]['model'] and k['query_id'] == by_id[k['evaluation_id']]['query_id'] for k in keys))
    check('per_model_exports_match_combined_results', all(rows(directory / (m.replace(':', '_') + '_results.jsonl')) == [r for r in records if r['model'] == m] for m in manifest['models']))
    with (directory / 'human_review_template.csv').open(encoding='utf-8-sig', newline='') as f:
        human = list(csv.DictReader(f))
    with (directory / 'claim_support_review_template.csv').open(encoding='utf-8-sig', newline='') as f:
        claims = list(csv.DictReader(f))
    check('human_review_scores_blank', len(human) == len(generated) and all(not value for row in human for key, value in row.items() if key not in {'evaluation_id', 'query_id'}))
    check('claim_review_labels_blank', len(claims) == sum(len(r['result']['claims']) for r in generated) and all(not r['support_label'] and not r['reviewer_id'] and not r['reason'] for r in claims))
    pairs = read(directory / 'paired_input_checks.json')
    complete_pairs = sum(all((m, q['query_id']) in {(r['model'],r['query_id']) for r in generated} for m in manifest['models']) for q in queries)
    check('successful_pair_evidence_prompt_pcs_identical', len(pairs) == complete_pairs and all(p['same_evidence'] and p['same_selection_prompt'] and p['same_pcs'] for p in pairs))
    check('all_attempted_pairs_same_selection_prompt', all(len({r['calls'][0]['prompt'] for r in records if r['query_id'] == q['query_id'] and r['calls']}) == 1 for q in queries))
    if manifest.get('config',{}).get('generation_revision'):
        limit=manifest['config']['max_claims']
        check('transport_prompt_and_payload_limits_match', all(
            call['schema']==json.loads(call['prompt'].split('Return only JSON conforming to this schema:')[1].splitlines()[1])
            and call['schema']['properties'][key]['maxItems']==limit
            and len(call.get('payload',{}).get(key,[]))<=limit
            for r in generated for call in r['calls'] for key in call['schema']['properties']))
        check('filtered_products_absent_from_facts_claims_and_products', all(
            not ({p['parent_asin'] for p in r['result']['product_filter']['excluded_products']} &
                 ({p['product_id'] for p in r['actual_input_facts']} | {c['parent_asin'] for c in r['result']['claims']} | {p['parent_asin'] for p in r['result']['products']}))
            for r in generated))
        check('final_rewrites_preserve_attribution', all(c['text'].startswith('One reviewer reports that ')
            for r in generated for c in r['result']['claims'] if c['grounding_method']=='nli_entailment' and c['citations'][0]['chunk_type']=='review'))
    result = {'checks': checks, 'all_checks_passed': all(checks.values()), 'planned_attempts':len(expected), 'generated_answers': len(generated), 'other_outcomes': len(failed), 'successful_model_pairs':complete_pairs, 'citations_checked': citations, 'claim_review_rows':len(claims), 'scope':'Artifact integrity and provenance only; not human semantic quality.'}
    (directory / 'artifact_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    raise SystemExit(audit(parser.parse_args().directory))
