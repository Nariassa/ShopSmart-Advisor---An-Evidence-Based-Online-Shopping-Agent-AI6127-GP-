import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from src.ui.disk_backend import DiskReviewStore, ShardDenseRetriever
from src.ui.pcs_adapter import product_pcs
from src.why_not.why_not import ReviewStore

class FullIntegrationTests(unittest.TestCase):
    def test_disk_search_matches_exact_cosine_across_shards_and_batches(self):
        vectors=np.array([[1,0],[.5,.5],[0,1],[-1,0],[.9,.1]],dtype=np.float16)
        encoder=SimpleNamespace(encode=lambda *a,**k:np.array([[1.,0.]],dtype=np.float32))
        with tempfile.TemporaryDirectory() as d:
            for i,part in enumerate([vectors[:3],vectors[3:]]):
                np.save(Path(d)/f'shard_{i:04d}_emb.npy',part)
                start=0 if i==0 else 3
                np.save(Path(d)/f'shard_{i:04d}_ids.npy',np.array([f'P{j}_review_0' for j in range(start,start+len(part))]))
            retriever=ShardDenseRetriever(d,encoder=encoder,batch_size=2)
            hits,_=retriever.search('battery',3)
            self.assertEqual(retriever.corpus_size,5)
            self.assertEqual([h.metadata['parent_asin'] for h in hits],['P0','P4','P1'])

    def test_lazy_product_store_preserves_sources(self):
        rows=[dict(chunk_id='A_review_0',parent_asin='A',chunk_type='review',aspect='battery life',text='[Product: Headphones | ASIN: A] Great battery.',rating=4.,category='Audio')]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'chunks.parquet';pd.DataFrame(rows).to_parquet(path)
            store=DiskReviewStore(path);store.prefetch(['A','missing'])
            self.assertTrue(store.has('A'));self.assertFalse(store.has('missing'))
            self.assertEqual(store.product_info('A')['title'],'Headphones')
            self.assertEqual(store.product_chunks('A')[0]['chunk_id'],'A_review_0')
            scorer=lambda texts:[SimpleNamespace(label='positive',score=.9) for _ in texts]
            score=product_pcs(store,scorer,'A','battery',[{'aspect':'battery life'}])
            self.assertEqual(score['version'],'team-pcs')
            self.assertTrue(0<score['score']<100)
            self.assertEqual(score['missing_metadata']['timestamp'],1)

if __name__=='__main__':unittest.main()
