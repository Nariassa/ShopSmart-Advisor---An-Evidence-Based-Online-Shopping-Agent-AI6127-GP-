import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from src.generation.testing import prepare_inputs, reusable_record, sha, export_package


class Store:
    def lookup(self, ids):
        return [{'chunk_id':key,'parent_asin':'A','chunk_type':'review','aspect':'comfort',
                 'text':key,'reason':'must not leak','relevance_label':2} for key in set(ids)]


class GenerationTestingTests(unittest.TestCase):
    def test_all_candidates_retained_ordered_and_gold_fields_never_passed(self):
        queries = [{'query_id':'Q1'}]
        annotations = [{'query_id':'Q1','chunk_id':key,'display_order':order,
                        'candidate_text_sha256':hashlib.sha256(key.encode()).hexdigest(),
                        'relevance_label':label,'reason':'do not send to model','confidence':'High'}
                       for key,order,label in [('b',2,0),('a',1,2)]]
        mapping, provenance = prepare_inputs(queries, annotations, Store())
        self.assertEqual([r['chunk_id'] for r in mapping['Q1']], ['a','b'])
        self.assertEqual(provenance['original_text_sha256_matches'],2)
        for row in mapping['Q1']:
            self.assertTrue(set(row).isdisjoint({'reason','relevance_label','confidence'}))
        annotations[0]['candidate_text_sha256'] = 'invalid'
        with self.assertRaises(ValueError):
            prepare_inputs(queries, annotations, Store())

    def test_resume_rejects_changed_configuration_or_evidence(self):
        manifest = {'config':'frozen'}
        row = {'manifest_sha256':sha(manifest),'evidence_sha256':'evidence','model':'m','query_id':'q'}
        self.assertTrue(reusable_record(row,manifest,'evidence','m','q'))
        self.assertFalse(reusable_record(row,manifest,'changed','m','q'))
        self.assertFalse(reusable_record(row,{'config':'changed'},'evidence','m','q'))

    def test_failure_is_exported_without_fake_ragas_answers_or_human_scores(self):
        manifest = {'models':['m'],'condition':'development'}
        queries = [{'query_id':'Q1','query':'Question'}]
        row = {'run_status':'failed','model':'m','query_id':'Q1','query':'Question',
               'evaluation_id':'E1','error':'test failure'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = export_package(root, manifest, queries, {'Q1':[]}, [row])
            self.assertEqual(summary['generated_runs'],0)
            self.assertIsNone(summary['citation_traceability'])
            self.assertEqual((root/'ragas_inputs.jsonl').read_text(encoding='utf-8'),'')
            self.assertEqual(json.loads((root/'failures_and_abstentions.jsonl').read_text(encoding='utf-8'))['error'],'test failure')
            self.assertFalse(summary['human_evaluation_completed'])

if __name__ == '__main__':
    unittest.main()
