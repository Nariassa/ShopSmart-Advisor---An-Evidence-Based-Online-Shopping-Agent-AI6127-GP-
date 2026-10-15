"""Check controlled inputs and honest pre-supplementation evaluation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.generation.generator import ShopSmartGenerator
from src.generation.prompt_compare import main, selection_metrics, validate_rubric

ROOT = Path(__file__).resolve().parents[1]


class SelectingBackend:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        facts = json.loads(prompt.split('EVIDENCE_FACTS_JSON: ', 1)[1])
        if 'product_name' in facts[0]:
            return {'selections': [{'fact_id': facts[0]['fact_id'], 'product_name': facts[0]['product_name']}]}
        return {'claims': [{'fact_id': facts[0]['fact_id'], 'text': facts[0]['quote']}]}

    def describe(self):
        return {'last_generation': {'generation_seconds': 1, 'prompt_tokens': 100}}


class PromptComparisonTests(unittest.TestCase):
    def test_comparison_freezes_facts_and_does_not_supplement_model_selection(self):
        backend = SelectingBackend()
        with tempfile.TemporaryDirectory() as directory:
            with patch('src.generation.prompt_compare.backend_from_args', return_value=backend):
                code = main(['--evidence', str(ROOT/'examples/generation_evidence.json'),
                             '--query', 'battery and comfort', '--aspects', 'battery life;comfort',
                             '--models', 'test-model', '--output-dir', directory])
            summary = json.loads((Path(directory)/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(code, 0)
            self.assertEqual(len(backend.prompts), 2)
            self.assertEqual(backend.prompts[0].split('EVIDENCE_FACTS_JSON: ', 1)[1],
                             backend.prompts[1].split('EVIDENCE_FACTS_JSON: ', 1)[1])
            for row in summary['results']:
                self.assertEqual(row['facts_sha256'], summary['facts_sha256'])
                self.assertEqual(len(row['accepted_claims']), 1)
                self.assertNotIn('direct_relevance_rate', row['metrics'])

    def test_relevance_metrics_do_not_count_rejected_or_background_as_direct_coverage(self):
        labels = {'F1': {'relevance': 2, 'aspects': ['battery life']},
                  'F2': {'relevance': 1, 'aspects': ['comfort']},
                  'F3': {'relevance': 0, 'aspects': []}}
        payload = {'claims': [{'fact_id': f} for f in ['F1', 'F2', 'F3']]}
        accepted = [{'fact_id': 'F1'}, {'fact_id': 'F2'}]
        metrics = selection_metrics(payload, accepted, [{'reason': 'bad_quote'}],
                                    ['battery life', 'comfort'], labels)
        self.assertEqual(metrics['direct_relevance_rate'], 0.5)
        self.assertEqual(metrics['useful_relevance_rate'], 1)
        self.assertEqual(metrics['requested_aspect_coverage'], 0.5)
        self.assertEqual(metrics['irrelevant_claims'], 0)
        self.assertEqual(metrics['rejected_claims'], 1)
        with self.assertRaises(ValueError):
            validate_rubric({'query': 'other', 'requested_aspects': []}, 'query', [], [])

    def test_generator_exposes_actual_requested_aspects_and_constraints_to_model(self):
        rows = json.loads((ROOT/'examples/generation_evidence.json').read_text(encoding='utf-8'))['evidence']
        backend = SelectingBackend()
        ShopSmartGenerator(llm=backend, prompt_version='v2').generate_recommendation(
            'battery and comfort', rows, requested_aspects=['comfort'],
            constraints={'max_price': 200, 'currency': 'USD', 'target_category': 'earbuds'})
        prompt = backend.prompts[0]
        requirements = json.loads(prompt.split('USER_REQUIREMENTS_JSON: ', 1)[1].splitlines()[0])
        self.assertEqual(requirements['requested_aspects'], ['comfort'])
        self.assertEqual(requirements['constraints']['max_price'], 200)
        self.assertEqual(requirements['constraints']['target_category'], 'earbuds')


if __name__ == '__main__':
    unittest.main()
