"""Load optional local credentials without replacing explicit process settings."""
import os
from pathlib import Path

from dotenv import load_dotenv


def load_local_settings(project_root):
    if os.getenv('SHOPSMART_IGNORE_DOTENV') == '1':
        return
    load_dotenv(Path(project_root) / '.env', override=False, encoding='utf-8-sig')
