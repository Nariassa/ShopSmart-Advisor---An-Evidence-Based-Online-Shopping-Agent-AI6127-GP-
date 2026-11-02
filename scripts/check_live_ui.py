"""Local live-data check. Does not call paid answer generation."""
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.ui.backend import ShopSmartBackend
root=Path(__file__).resolve().parents[1]
kb=root.parent
os.environ.setdefault('SHOPSMART_RETRIEVER','shards')
started=time.perf_counter()
print('Loading complete knowledge base and models...',flush=True)
b=ShopSmartBackend.load(chunks_path=str(kb/'chunks.parquet'),chroma_path=str(kb/'chroma_db'))
print('LOADED',round(time.perf_counter()-started,2),'seconds',flush=True)
print('RETRIEVAL',b.retrieval_mode,'CORPUS',b.dense.corpus_size if b.dense else None,flush=True)
report=[]
for q in ['wireless earbuds with long battery life for flights','Bluetooth headphones with a clear microphone for work calls','portable speaker with strong bass','lightweight laptop for travel','noise cancelling headphones comfortable with glasses']:
    hits,elapsed=b.search(q,3)
    rows=[]
    for hit in hits:
        a=b.analyse(hit,q)
        risk=b.explain(hit.asin,q,with_alternatives=True)
        rows.append(dict(asin=hit.asin,title=hit.title,evidence=len(hit.evidence),score=a['pcs']['score'],coverage=a['coverage'],risk=risk.get('headline')))
    report.append(dict(query=q,search_seconds=elapsed,products=rows))
    print('QUERY',q,'PRODUCTS',len(hits),'SEARCH_SECONDS',round(elapsed,3),flush=True)
(root/'live-validation.json').write_text(json.dumps(dict(mode=b.mode,corpus_size=b.dense.corpus_size,checks=report),indent=2),encoding='utf-8')
print('LIVE CHECK COMPLETE',flush=True)
