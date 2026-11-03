"""Exercise the real full-data UI without paid answer generation."""
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
root=Path(__file__).resolve().parents[1]
os.environ['SHOPSMART_MODE']='full'
os.environ['SHOPSMART_RETRIEVER']='shards'
os.environ['CHUNKS_PATH']=str(root.parent/'chunks.parquet')
os.environ['CHROMA_PATH']=str(root.parent/'chroma_db')
os.environ['EMBEDDING_SHARDS_PATH']=str(root.parent/'embedding_shards')
os.environ.pop('OPENAI_API_KEY',None)
os.environ['SHOPSMART_IGNORE_DOTENV']='1'
from streamlit.testing.v1 import AppTest
app=AppTest.from_file(str(root/'app.py')).run(timeout=180)
assert not list(app.exception),[e.message for e in app.exception]
app.text_input[0].set_value('wireless earbuds with long battery life for flights')
[b for b in app.button if b.label=='Find my options'][0].click().run(timeout=180)
assert not list(app.exception),[e.message for e in app.exception]
assert len(app.metric)==3
print('FULL_UI_QUERY_OK',len(app.metric),flush=True)
for key in [b.key for b in app.button if b.key and b.key.startswith('add_')]:
    app.button(key=key).click().run(timeout=180)
app.radio(key='page').set_value('Compare').run(timeout=180)
assert not list(app.exception) and len(app.metric)==3
print('FULL_UI_COMPARE_OK',flush=True)
app.radio(key='theme').set_value('Night').run(timeout=180)
assert not list(app.exception)
for key in [b.key for b in app.button if b.key and b.key.startswith('remove_')]:
    app.button(key=key).click().run(timeout=180)
assert len(app.session_state['shortlist'])==0
print('FULL_UI_NIGHT_AND_REMOVE_OK',flush=True)
print('FULL_UI_CHECK_COMPLETE',flush=True)
