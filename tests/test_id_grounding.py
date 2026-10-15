import json
import unittest
from unittest.mock import patch

from src.generation.evidence import normalize_evidence
from src.generation.generator import ShopSmartGenerator
from src.generation.grounding import extract_facts, verify_selections, verify_claims
from src.generation.llm import OllamaBackend
from src.generation.prompts import SELECTION_SCHEMA
from src.generation.sentiment import LexiconSentiment, evidence_sentiment


class IDGroundingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'chunk_id': 'A1', 'parent_asin': 'A', 'title': 'Example Tracker',
                      'chunk_type': 'review', 'aspect': 'connectivity',
                      'text': 'The connection failed after one day.'}]
        self.items = normalize_evidence(self.rows)
        classifier = LexiconSentiment()
        sentiments = {i.chunk_id: evidence_sentiment(i, classifier) for i in self.items}
        self.facts = extract_facts(self.items, 'connection', ['connectivity'], classifier, sentiments)

    def test_id_name_hydrates_exact_original_and_rejects_wrong_name_unknown_id_and_text(self):
        valid = {'fact_id': 'F1', 'product_name': 'Example Tracker'}
        payload = {'selections': [valid, {**valid, 'product_name': 'Another Tracker'},
                                  {**valid, 'fact_id': 'F99'}, {**valid, 'text': 'invented'}, valid]}
        accepted, rejected = verify_selections(payload, self.facts, self.items)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]['text'], self.rows[0]['text'])
        self.assertEqual(accepted[0]['product_name'], 'Example Tracker')
        self.assertEqual({r['reason'] for r in rejected},
                         {'product_name_mismatch', 'unknown_fact', 'invalid_selection_fields', 'duplicate_fact'})
        with self.assertRaises(ValueError):
            verify_selections({'claims': []}, self.facts, self.items)

    def test_non_latin_rewrite_never_reaches_english_nli(self):
        class Verifier:
            def supports(self, *args):
                raise AssertionError('Chinese must not reach English NLI')
        accepted, rejected = verify_claims({'claims': [{'fact_id': 'F1', 'text': '连接失败。'}]},
                                          self.facts, self.items, Verifier(), english_only=True)
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]['reason'], 'non_english_script')

    def test_two_stages_keep_one_fact_per_rewrite_and_fallback_to_quote(self):
        class Verifier:
            def supports(self, premise, claim):
                return ('failed' in claim, 0.99 if 'failed' in claim else 0.01)
        class Backend:
            def __init__(self, rewrite):
                self.rewrite = rewrite
                self.calls = []
            def generate_structured(self, prompt, schema):
                if 'claims' in schema['properties']:
                    return self.generate(prompt)
                self.calls.append(schema)
                return {'selections': [{'fact_id': 'F1', 'product_name': 'Example Tracker'}]}
            def generate(self, prompt):
                self.calls.append(prompt)
                return {'claims': [{'fact_id': 'F1', 'text': self.rewrite},
                                   {'fact_id': 'F999', 'text': 'Connection failed.'}]}
        for supported in [True, False]:
            with self.subTest(supported=supported):
                backend = Backend('One reviewer reports that the connection failed after one day.' if supported else 'One reviewer reports that the connection is perfect.')
                result = ShopSmartGenerator(llm=backend, grounding_verifier=Verifier(), allow_fallback=False).generate_recommendation(
                    'connection', self.rows, requested_aspects=['connectivity'])
                self.assertEqual(len(backend.calls), 2)
                self.assertEqual(backend.calls[0], SELECTION_SCHEMA)
                self.assertEqual(result['grounding']['llm_verified_selections'], 1)
                self.assertEqual(result['grounding']['rewrite_verified_claims'], int(supported))
                self.assertEqual(len(result['claims']), 1)
                claim = result['claims'][0]
                self.assertEqual(claim['fact_id'], 'F1')
                self.assertEqual(len(claim['citations']), 1)
                self.assertEqual(claim['citations'][0]['quote'], self.rows[0]['text'])
                self.assertEqual(claim['text'], backend.rewrite if supported else self.rows[0]['text'])

    def test_ollama_transport_uses_selection_schema(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size):
                return json.dumps({'done': True, 'response': json.dumps({'selections': []})}).encode()
        with patch('src.generation.llm.urlopen', return_value=Response()) as transport:
            backend = OllamaBackend()
            self.assertEqual(backend.generate_structured('select', SELECTION_SCHEMA), {'selections': []})
            self.assertEqual(json.loads(transport.call_args.args[0].data)['format'], SELECTION_SCHEMA)
            self.assertEqual(backend.last_generation['proposed_selections'], [])

    def test_non_english_source_is_not_sent_to_english_nli(self):
        from dataclasses import replace
        rows = [{**self.rows[0], 'text': '连接失败。'}]
        items = normalize_evidence(rows)
        facts = [replace(self.facts[0], quote=rows[0]['text'], start=0, end=len(rows[0]['text']))]
        class Verifier:
            def supports(self, *args):
                raise AssertionError('Chinese source must not reach English NLI')
        accepted, rejected = verify_claims({'claims': [{'fact_id': 'F1', 'text': 'The connection failed.'}]},
                                          facts, items, Verifier(), english_only=True)
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]['reason'], 'nli_source_non_english')


if __name__ == '__main__':
    unittest.main()
