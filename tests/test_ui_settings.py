import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.ui.settings import load_local_settings

class LocalSettingsTests(unittest.TestCase):
    def test_loads_project_file(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            (Path(folder)/'.env').write_text('OPENAI_API_KEY=test-only-placeholder\n',encoding='utf-8-sig')
            load_local_settings(folder)
            self.assertEqual(os.getenv('OPENAI_API_KEY'),'test-only-placeholder')

    def test_explicit_environment_takes_precedence(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'OPENAI_API_KEY':'explicit-test-placeholder'}, clear=True):
            (Path(folder)/'.env').write_text('OPENAI_API_KEY=file-test-placeholder\n',encoding='utf-8')
            load_local_settings(folder)
            self.assertEqual(os.getenv('OPENAI_API_KEY'),'explicit-test-placeholder')

    def test_no_paid_smoke_check_can_skip_local_credentials(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'SHOPSMART_IGNORE_DOTENV':'1'}, clear=True):
            (Path(folder)/'.env').write_text('OPENAI_API_KEY=test-only-placeholder\n',encoding='utf-8')
            load_local_settings(folder)
            self.assertNotIn('OPENAI_API_KEY',os.environ)
