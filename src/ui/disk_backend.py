"""Bounded-memory access to the same full corpus and precomputed BGE vectors."""
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from src.why_not.why_not import ReviewStore

class DiskReviewStore(ReviewStore):
    def __init__(self, path):
        self.dataset = ds.dataset(str(path),format='parquet')
        self._cache = OrderedDict()

    def _matching(self, field, values, columns=None):
        scanner = self.dataset.scanner(columns=columns,filter=ds.field(field).isin(values),
                                       batch_size=2048,batch_readahead=1,fragment_readahead=1,use_threads=False)
        rows=[]
        for batch in scanner.to_batches():
            rows.extend(batch.to_pylist())
        return rows

    def has(self, asin):
        if asin in self._cache:return bool(self._cache[asin])
        return bool(self._matching('parent_asin',[asin],['parent_asin']))

    def product_chunks(self, asin):
        if asin not in self._cache:
            self._cache[asin] = self._matching('parent_asin',[asin],self.COLUMNS)
            if len(self._cache)>100:self._cache.popitem(last=False)
        return self._cache[asin]

    def prefetch(self, asins):
        pending=[a for a in dict.fromkeys(asins) if a not in self._cache]
        if not pending:return
        grouped={a:[] for a in pending}
        for row in self._matching('parent_asin',pending,self.COLUMNS):
            grouped[row['parent_asin']].append(row)
        self._cache.update(grouped)
        while len(self._cache)>100:self._cache.popitem(last=False)

class ShardDenseRetriever:
    """Exact cosine search over every vector, using small memory-mapped batches."""
    def __init__(self,path,encoder=None,batch_size=4096):
        self.path=Path(path)
        self.encoder=encoder
        self.batch_size=batch_size
        self.shards=sorted(self.path.glob('shard_*_emb.npy'))
        if not self.shards:raise FileNotFoundError(f'No embedding shards at {path}')
        self.corpus_size=0
        for f in self.shards:
            a=np.load(f,mmap_mode='r')
            ids=np.load(f.with_name(f.name.replace('_emb.npy','_ids.npy')),mmap_mode='r')
            if len(a)!=len(ids):raise ValueError(f'Mismatched vectors and IDs: {f.name}')
            self.corpus_size+=len(a)
    def load(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder=SentenceTransformer('BAAI/bge-base-en-v1.5',device='cpu')
        return self
    def search(self,query,top_k=10):
        if not query.strip() or top_k<1:raise ValueError('Provide a query and positive top_k.')
        started=time.perf_counter()
        q=np.asarray(self.encoder.encode([query],normalize_embeddings=True),dtype=np.float32)[0]
        q=q/max(float(np.linalg.norm(q)),1e-12)
        best=[]
        for f in self.shards:
            vectors=np.load(f,mmap_mode='r')
            ids=np.load(f.with_name(f.name.replace('_emb.npy','_ids.npy')),mmap_mode='r')
            local=[]
            for start in range(0,len(vectors),self.batch_size):
                block=np.asarray(vectors[start:start+self.batch_size],dtype=np.float32)
                scores=(block@q)/np.maximum(np.linalg.norm(block,axis=1),1e-12)
                k=min(top_k,len(scores))
                selected=np.argpartition(scores,-k)[-k:]
                local.extend((float(scores[i]),str(ids[start+i])) for i in selected)
                local=sorted(local,reverse=True)[:top_k]
            best=sorted(best+local,reverse=True)[:top_k]
        results=[SimpleNamespace(chunk_id=cid,score=score,distance=1-score,rank=i,
                                 metadata={'parent_asin':cid.split('_',1)[0]})
                 for i,(score,cid) in enumerate(best,1)]
        return results,SimpleNamespace(total_seconds=time.perf_counter()-started)
