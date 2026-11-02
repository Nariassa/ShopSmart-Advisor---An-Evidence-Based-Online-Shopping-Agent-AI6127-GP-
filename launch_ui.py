"""Start the UI against local data; no copying or rebuilding the database."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='Start ShopSmart Advisor')
parser.add_argument('--mode', choices=['full','saved'], default='full')
parser.add_argument('--port', type=int, default=8502)
parser.add_argument('--retriever',choices=['shards','chroma'],default='shards')
args = parser.parse_args()
python = root / '.venv' / ('Scripts/python.exe' if os.name=='nt' else 'bin/python')
if args.mode == 'saved':
    python = Path(sys.executable)
if not python.is_file():
    parser.error('Project environment not found. Follow UI_README.md to install it.')
env = os.environ.copy()
env['SHOPSMART_MODE'] = args.mode
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    env.setdefault(name, '4')
env.setdefault('TOKENIZERS_PARALLELISM', 'false')
if args.mode == 'full':
    kb = Path(env.get('SHOPSMART_DATA_DIR',str(root.parent)))
    env.setdefault('CHUNKS_PATH',str(kb/'chunks.parquet'))
    env.setdefault('CHROMA_PATH',str(kb/'chroma_db'))
    env['SHOPSMART_RETRIEVER'] = args.retriever
    env.setdefault('EMBEDDING_SHARDS_PATH',str(kb/'embedding_shards'))
    if args.retriever == 'shards' and not Path(env['EMBEDDING_SHARDS_PATH']).is_dir():
        parser.error('Embedding shards missing. Set EMBEDDING_SHARDS_PATH or use --retriever chroma.')
    if not Path(env['CHUNKS_PATH']).is_file():
        parser.error('Corpus missing. Set SHOPSMART_DATA_DIR or CHUNKS_PATH.')
    if args.retriever == 'chroma' and not (Path(env['CHROMA_PATH'])/'chroma.sqlite3').is_file():
        parser.error('Chroma database missing. Set CHROMA_PATH or use --retriever shards.')
if args.mode == 'full':
    try:
        check = subprocess.run([str(python), '-c', 'import pandas, torch, sentence_transformers, transformers, streamlit'], capture_output=True, text=True, errors='replace', timeout=60)
    except subprocess.TimeoutExpired:
        (root/'runtime-check.log').write_text('Native dependency import did not complete within 60 seconds. Earlier direct checks returned Windows Application Control WinError 4551.', encoding='utf-8')
        parser.exit(1, 'Full-mode runtime check timed out. See runtime-check.log and UI_README.md.\n')
    if check.returncode:
        (root/'runtime-check.log').write_text(check.stderr, encoding='utf-8')
        parser.exit(1, 'Full-mode runtime could not load. See runtime-check.log and UI_README.md. The database has not been changed.\n')
print(f'ShopSmart: http://127.0.0.1:{args.port} ({args.mode})',flush=True)
raise SystemExit(subprocess.call([str(python),'-m','streamlit','run',str(root/'app.py'),'--server.address','127.0.0.1','--server.port',str(args.port),'--server.headless','true','--browser.gatherUsageStats','false'],cwd=root,env=env))
