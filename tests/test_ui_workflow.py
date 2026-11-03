import os
from pathlib import Path
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

class ShoppingWorkflowTests(unittest.TestCase):
    def test_search_text_survives_navigation(self):
        root=Path(__file__).resolve().parents[1]
        with patch.dict(os.environ, {'SHOPSMART_MODE':'saved','SHOPSMART_IGNORE_DOTENV':'1'}):
            app=AppTest.from_file(str(root/'app.py')).run(timeout=30)
            app.button(key='example_0').click().run(timeout=30)
            query=app.session_state['query']
            app.radio(key='theme').set_value('Night').run(timeout=30)
            self.assertFalse(list(app.exception))
            app.radio(key='page').set_value('Compare').run(timeout=30)
            app.radio(key='page').set_value('Discover').run(timeout=30)
            self.assertFalse(list(app.exception))
            self.assertEqual(app.text_input[0].value,query)
            app.button(key='team_info').click().run(timeout=30)
            roster='\n'.join(m.value for m in app.markdown)
            for name in ['Balamurugan Harini', 'Cen Qinxuan', 'Li Ka Lam', 'Wang Zhongda', 'Zheng Haoyang']:
                self.assertIn(name,roster)
            self.assertFalse(list(app.exception))
            self.assertEqual(app.session_state['query'],query)
