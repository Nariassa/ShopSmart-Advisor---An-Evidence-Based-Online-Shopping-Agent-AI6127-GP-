"""Local model selection, completion checks and two-model pipeline handoff."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.generation.compare import main as compare_main
from src.generation.llm import LLMUnavailable, OllamaBackend

ROOT = Path(__file__).resolve().parents[1]


class Response:
    def __init__(self, envelope):
        self.envelope = envelope

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size):
        return json.dumps(self.envelope).encode()


class SelectingBackend:
    def __init__(self, model):
        self.model = model

    def generate(self, prompt):
        facts = json.loads(prompt.split('EVIDENCE_FACTS_JSON: ')[1])
        return {'selections': [{'fact_id': fact['fact_id'], 'product_name': fact['product_name']} for fact in facts[:3]]}

    def describe(self):
        return {'model': self.model, 'last_generation': {'proposed_claims': []}}


class LocalModelIntegrationTests(unittest.TestCase):
    def test_both_model_names_options_and_usage_are_preserved(self):
        for model in ['qwen2.5:7b', 'llama3.1:8b']:
            with self.subTest(model=model):
                envelope = {'done': True, 'done_reason': 'stop', 'model': model,
                            'response': json.dumps({'claims': []}),
                            'eval_count': 20, 'eval_duration': 1_000_000_000,
                            'prompt_eval_count': 40, 'total_duration': 2_000_000_000}
                with patch('src.generation.llm.urlopen', return_value=Response(envelope)) as transport:
                    backend = OllamaBackend(model, num_ctx=2048, num_predict=256, keep_alive=0)
                    backend.generate('Test prompt')
                    body = json.loads(transport.call_args.args[0].data)
                    self.assertEqual(body['model'], model)
                    self.assertEqual(body['options']['num_ctx'], 2048)
                    self.assertEqual(body['options']['num_predict'], 256)
                    self.assertEqual(body['keep_alive'], 0)
                    metadata = backend.describe()
                    self.assertEqual(metadata['model'], model)
                    self.assertEqual(metadata['last_generation']['tokens_per_second'], 20)

    def test_incomplete_or_truncated_output_is_not_reported_as_success(self):
        for completion in [{'done': False}, {'done': True, 'done_reason': 'length'}]:
            with self.subTest(completion=completion):
                envelope = {'response': json.dumps({'claims': []}), **completion}
                with patch('src.generation.llm.urlopen', return_value=Response(envelope)):
                    backend = OllamaBackend()
                    with self.assertRaises(LLMUnavailable):
                        backend.generate('Test prompt')
                    self.assertIsNone(backend.last_generation)

    def test_comparison_uses_identical_evidence_and_writes_separate_model_results(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with patch('src.generation.compare.backend_from_args',
                       side_effect=lambda args, model: SelectingBackend(model)):
                code = compare_main(['--evidence', str(ROOT/'examples/generation_evidence.json'),
                                     '--query', 'battery and comfort', '--aspects', 'battery life;comfort',
                                     '--output-dir', str(target)])
            self.assertEqual(code, 0)
            summary = json.loads((target/'summary.json').read_text(encoding='utf-8'))
            results = [json.loads((target/name).read_text(encoding='utf-8'))
                       for name in ['qwen2.5_7b.json', 'llama3.1_8b.json']]
            self.assertTrue(all(row['passed'] for row in summary['results']))
            self.assertTrue(all(result['backend'] == 'llm' for result in results))
            self.assertEqual(results[0]['evidence_sha256'], results[1]['evidence_sha256'])
            self.assertEqual(results[0]['products'][0]['pcs'], results[1]['products'][0]['pcs'])
            self.assertEqual(results[0]['mechanical_audit']['citation_traceability'], 1)
            self.assertEqual(results[1]['mechanical_audit']['citation_traceability'], 1)

    def test_one_model_failure_is_reported_without_extractive_success(self):
        class FailedBackend:
            def generate(self, prompt):
                raise LLMUnavailable('模拟模型不可用')

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with patch('src.generation.compare.backend_from_args',
                       side_effect=lambda args, model: FailedBackend() if model.startswith('qwen')
                       else SelectingBackend(model)):
                code = compare_main(['--evidence', str(ROOT/'examples/generation_evidence.json'),
                                     '--query', 'battery', '--aspects', 'battery life', '--output-dir', str(target)])
            self.assertEqual(code, 1)
            summary = json.loads((target/'summary.json').read_text(encoding='utf-8'))
            self.assertFalse(summary['results'][0]['passed'])
            self.assertTrue(summary['results'][1]['passed'])
            self.assertFalse((target/'qwen2.5_7b.json').exists())


if __name__ == '__main__':
    unittest.main()
