"""Adapt team PCS to the existing UI contract without changing its algorithm."""
from src.generation.evidence import normalize_evidence
from src.generation.pcs import calculate_pcs
from src.generation.sentiment import Sentiment


def product_pcs(store, sentiment, asin, query, coverage):
    items = normalize_evidence(store.product_chunks(asin))
    reviews = [item for item in items if item.chunk_type == 'review']
    values = sentiment([item.body for item in reviews]) if reviews else []
    sentiments = {
        item.chunk_id: Sentiment(value.label, {'positive':1.,'neutral':0.,'negative':-1.}[value.label], value.score, 'transformer')
        for item,value in zip(reviews, values)
    }
    aspects = [c['aspect'] for c in coverage] or ['general']
    result = calculate_pcs(items, sentiments, aspects)
    return {**result, 'weights': {'quality':.4,'consistency':.35,'recency':.25},
            'version':'team-pcs', 'scope':'available_product_reviews',
            'deal_breakers':0,
            'explanation':'Evidence confidence based on review quality, consistency, recency, volume and requirement coverage. Missing review dates use a neutral recency value.'}
