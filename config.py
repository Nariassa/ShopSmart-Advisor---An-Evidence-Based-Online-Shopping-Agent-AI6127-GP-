"""Shared configuration for the Electronics KB pipeline."""

# ---- Input / checkpoint paths ----
REVIEWS_CLEAN_PATH = "reviews_clean.parquet"        # output of step2
CHUNKS_PATH = "chunks.parquet"                      # output of step3
CORPUS_STATS_PATH = "corpus_stats.json"             # output of step5

# ---- Cleaning thresholds ----
LANG_KEEP = "en"

# ---- PII entities to mask ----
PII_ENTITIES = ["PHONE_NUMBER", "EMAIL_ADDRESS"]

# ---- Aspect taxonomy for semantic chunking ----
ASPECTS = [
    "price/value", "build quality", "specifications/performance",
    "battery life", "screen/display", "connectivity",
    "customer service", "ease of use", "sound quality", "design/appearance",
]

# ---- Indexing ----
ES_HOST = "http://localhost:9200"
ES_INDEX_NAME = "electronics_kb"
CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "electronics_kb"
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# ---- Retrieval fusion ----
RRF_K = 60

TOP_K_DEFAULT = 10