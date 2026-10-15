"""English handoff README rendering shared by the local packaging command."""
import json


def render(directory, summary):
    failures = [json.loads(s) for s in (directory/'failures_and_abstentions.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    failed_ids = ', '.join(r['model']+' / '+r['query_id'] for r in failures) or 'None'
    revised = 'generation_revision' in json.loads((directory/'manifest.json').read_text(encoding='utf-8-sig'))['config']
    lines = ['# Grounded Generation & Purchase Confidence Scoring: Generation Testing Handoff', '',
             'Completed: 2026-10-15.', '',
             f'{summary["recorded_runs"]}/{summary["planned_runs"]} attempts recorded; {summary["generated_runs"]} real model answers; {len(failures)} failed or other outcomes.', '',
             '| Model | Generated answers | Citation checks | NLI-rejected rewrites |',
             '| --- | ---: | ---: | ---: |']
    for row in summary['models']:
        lines.append(f'| {row["model"]} | {row["generated"]}/20 | {row["valid_citations"]}/{row["citation_count"]} | {row["rejected_rewrites"]} |')
    lines += ['', f'Attempts without an answer: {failed_ids}. Original responses and errors are retained; saved outcomes were not replaced with retries.', '',
        '## Start here: System Integration, User Interface & Evaluation', '',
        '1. Read [the summary](summary.md) and [known quality issues](QUALITY_REVIEW.md) for the experiment scope and limitations.',
        '2. Follow [the review guide](E_REVIEW_GUIDE.md). Distribute blinded_review_answers.md or blinded_review_packets.jsonl and independent copies of both scoring templates. Keep model_key_for_E.csv separately. Do not give annotators the complete package. Show automatic flags and known issues after independent scoring to reduce anchoring.',
        '3. Use human_review_template.csv for answer-level assessment and claim_support_review_template.csv for claim support. All scores and support labels are currently blank.',
        '4. ragas_inputs.jsonl contains actual selection-model fact contexts and final answers, without fabricated references. Formal human evaluation, IAA, and RAGAS remain the responsibility of System Integration, User Interface & Evaluation.', '',
        '## Traceability and reproduction', '',
        '- results_all.jsonl and runs/ preserve attempts, structured model outputs, prompts, NLI outcomes, and citations.',
        '- artifact_audit.json records provenance checks. SHA256SUMS.txt lists package file hashes except its own.',
        '- manifest.json, model_digests.json, runtime_versions.json, and code_snapshot/ record the frozen configuration and versions. Reproduction still requires the project environment.',
        '- Run this command from the repository root:', '',
        '~~~powershell',
        '& ../.venv/Scripts/python.exe scripts/audit_generation_handoff.py outputs/'+directory.name,
        '~~~', '',
        '- See GENERATION_TESTING.md for execution and RUN_NOTES.md for runtime and recovery details.',
        '- Historical directory names, filenames, code snapshots, and serialized identifiers retain their original values for integrity and compatibility. Current module display names are used in documentation.', '',
        '## Interpretation limits', '',
        'This development batch replays 791 annotation candidates for 20 questions. All candidate text hashes matched. Human relevance labels did not filter or reorder inputs. The selection model receives a program-built subset of facts; see actual_input_facts.',
        'Formal rankings from Retrieval, Reranking & Evaluation were not used. The questions have been used during development, so this is neither an independent test set nor an end-to-end retrieval evaluation. Citation traceability does not measure semantic accuracy or human faithfulness.',
        'PCS is uncalibrated, and formal polarization and coverage signals from Aspect Analysis & Purchase Decision Explanation were not integrated. Human scoring, IAA, and RAGAS have not been completed.']
    if revised:
        lines += ['This revision uses the v1 base with four fixes, not unchanged historical v1 or a v1/v2/v3 prompt ablation. Rewrite guards and title filtering are conservative heuristics with false-positive and false-negative risks. See QUALITY_REVIEW.md and revision_comparison.md. These automatic checks do not establish a model quality ranking.']
    else:
        lines += ['This baseline uses v1. Product-type, qualification, sentiment, and rendering issues remain; see QUALITY_REVIEW.md. The later revised experiment is saved separately. These automatic checks do not establish a model quality ranking.']
    return chr(10).join(lines)+chr(10)
