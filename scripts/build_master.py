"""
Build the sentiment annotation master file from your groupmate's retrieval files,
then (after annotation) compute agreement and export eval.xls.

Usage:
    python build_master.py build       # makes master_annotation.csv
    python build_master.py topup       # (old) appends extra chunks from your knowledge base until N_TARGET is reached
    python build_master.py rebalance   # NEW: drops spec rows, adds review chunks from chunks.parquet so
                                       #      the polarity classes are balanced (see rebalance() below)
    python build_master.py finalize    # reads the filled master, prints agreement, writes eval.xls

Edit the CONFIG block if your files or field names differ.
"""
import json
import sys
import pandas as pd

# ---------------- CONFIG ----------------
QUERIES_PATH = "formal_queries_v1.jsonl"
POOL_PATH = "formal_annotation_pool.jsonl"
QRELS_PATH = "qrels_human_validated_v1.txt"   # TREC: query_id 0 chunk_id relevance (set None to skip)
MASTER_PATH = "master_annotation.csv"   # for rebalance/finalize use master_annotation_completed.csv
N_TARGET = 1000
MIN_RELEVANCE = 1        # only sample chunks the retriever judged relevant (needs QRELS_PATH)
SEED = 42

# Field-name guesses for the pool file; the first match is used.
CHUNK_ID_CANDIDATES = ["chunk_id", "doc_id", "docid", "id"]
TEXT_CANDIDATES = ["text", "chunk_text", "passage", "content"]

# --- used only by "topup" ---
CHUNKS_PATH = "chunks.parquet"      # your knowledge-base chunk store (.parquet, .jsonl or .csv)
ASPECT_CANDIDATES = ["aspect", "aspects", "aspect_label", "aspect_tags"]
RATING_CANDIDATES = ["rating", "star_rating", "stars", "overall"]
CATEGORY_CANDIDATES = ["category", "main_category", "product_category"]
MIN_WORDS, MAX_WORDS = 8, 150       # skip fragments and very long chunks
MAX_POOL_ROWS = 300_000             # pre-sample very large stores to keep memory sane

# --- used only by "rebalance" ---
TARGET_SHARE = {"positive": 0.40, "negative": 0.30, "neutral": 0.30}   # desired class mix of the final set
MAX_PER_ASIN = 3                    # at most this many new rows from one product (avoids near-duplicates)
SCAN_ROWS = 600_000                 # review chunks scanned (random subset of the store) before filtering

ASPECTS = ["battery life", "comfort", "sound quality", "connectivity",
           "specifications/performance", "screen/display", "ease of use",
           "design/appearance", "noise", "build quality", "image quality",
           "customer service"]
# ----------------------------------------


def pick(cols, candidates, what):
    for c in candidates:
        if c in cols:
            return c
    sys.exit(f"Could not find a {what} field. Pool columns are: {list(cols)}. "
             f"Add the right name to the CANDIDATES list in CONFIG.")


def build():
    queries = pd.read_json(QUERIES_PATH, lines=True)
    pool = pd.read_json(POOL_PATH, lines=True)
    cid = pick(pool.columns, CHUNK_ID_CANDIDATES, "chunk ID")
    txt = pick(pool.columns, TEXT_CANDIDATES, "text")
    pool = pool.rename(columns={cid: "chunk_id", txt: "text"})
    print(f"Pool: {len(pool)} pairs, {pool['chunk_id'].nunique()} unique chunks")

    if QRELS_PATH:
        qrels = pd.read_csv(QRELS_PATH, sep=r"\s+", header=None,
                            names=["query_id", "_zero", "chunk_id", "retrieval_relevance"],
                            dtype={"query_id": str, "chunk_id": str})
        pool["chunk_id"] = pool["chunk_id"].astype(str)
        pool = pool.merge(qrels[["query_id", "chunk_id", "retrieval_relevance"]],
                          on=["query_id", "chunk_id"], how="left")
        pool = pool[pool["retrieval_relevance"].fillna(0) >= MIN_RELEVANCE]
        print(f"After relevance >= {MIN_RELEVANCE}: {len(pool)} pairs")
    else:
        pool["retrieval_relevance"] = ""

    # add query fields only where the pool does not already carry them (avoids _x/_y duplicates)
    qcols = [c for c in ["intent", "target_aspects", "target_category", "difficulty"]
             if c not in pool.columns]
    if qcols:
        pool = pool.merge(queries[["query_id"] + qcols], on="query_id", how="left")

    # one row per distinct text so the eval set has no duplicates
    before = len(pool)
    pool = pool.drop_duplicates(subset="text")
    print(f"After removing duplicate texts: {len(pool)} (dropped {before - len(pool)})")

    if len(pool) > N_TARGET:
        # sample evenly across queries so no single query dominates
        per_q = max(1, N_TARGET // pool["query_id"].nunique())
        sampled = pd.concat([g.sample(min(len(g), per_q), random_state=SEED)
                             for _, g in pool.groupby("query_id")])
        if len(sampled) < N_TARGET:
            rest = pool.drop(sampled.index).sample(N_TARGET - len(sampled), random_state=SEED)
            sampled = pd.concat([sampled, rest])
        pool = sampled
    elif len(pool) < N_TARGET:
        print(f"WARNING: only {len(pool)} unique chunks available, short of {N_TARGET} by "
              f"{N_TARGET - len(pool)}. Sample extra chunks from the knowledge base.")

    pool = pool.sample(frac=1, random_state=SEED).reset_index(drop=True)
    pool.insert(0, "id", [f"R{i+1:04d}" for i in range(len(pool))])
    for col in ["aspect", "subjectivity", "annotator_1", "annotator_2", "agree", "label", "notes"]:
        pool[col] = ""
    keep = ["id", "query_id", "chunk_id", "text", "intent", "target_category", "target_aspects",
            "difficulty", "retrieval_relevance", "aspect", "subjectivity",
            "annotator_1", "annotator_2", "agree", "label", "notes"]
    for col in keep:
        if col not in pool.columns:
            pool[col] = ""
    pool[keep].to_csv(MASTER_PATH, index=False)
    print(f"Wrote {MASTER_PATH} with {len(pool)} rows. Aspect labels to use: {ASPECTS}")


def balanced_sample(df, n, by, seed=SEED):
    """Take about n rows spread evenly across the values of column `by`."""
    if n <= 0 or df.empty:
        return df.iloc[0:0]
    groups = [g for _, g in df.groupby(by)]
    per = -(-n // len(groups))
    picked = pd.concat([g.sample(min(len(g), per), random_state=seed) for g in groups])
    if len(picked) > n:
        picked = picked.sample(n, random_state=seed)
    elif len(picked) < n:
        rest = df.drop(picked.index)
        picked = pd.concat([picked, rest.sample(min(len(rest), n - len(picked)), random_state=seed)])
    return picked


def first_match(cols, candidates):
    return next((c for c in candidates if c in cols), None)


def topup():
    master = pd.read_csv(MASTER_PATH, dtype=str, keep_default_na=False)
    need = N_TARGET - len(master)
    if need <= 0:
        sys.exit(f"Master already has {len(master)} rows, nothing to add.")
    print(f"Master has {len(master)} rows; adding {need} from {CHUNKS_PATH}")

    # read only the columns we need when the store is a parquet file
    if CHUNKS_PATH.endswith(".parquet"):
        import pyarrow.parquet as pq
        names = pq.read_schema(CHUNKS_PATH).names
        cid = pick(names, CHUNK_ID_CANDIDATES, "chunk ID")
        txt = pick(names, TEXT_CANDIDATES, "text")
        extra = [first_match(names, c) for c in (ASPECT_CANDIDATES, RATING_CANDIDATES, CATEGORY_CANDIDATES)]
        chunks = pd.read_parquet(CHUNKS_PATH, columns=list(dict.fromkeys([cid, txt] + [e for e in extra if e])))
    elif CHUNKS_PATH.endswith(".jsonl"):
        chunks = pd.read_json(CHUNKS_PATH, lines=True)
    else:
        chunks = pd.read_csv(CHUNKS_PATH)
    cid = pick(chunks.columns, CHUNK_ID_CANDIDATES, "chunk ID")
    txt = pick(chunks.columns, TEXT_CANDIDATES, "text")
    asp = first_match(chunks.columns, ASPECT_CANDIDATES)
    rat = first_match(chunks.columns, RATING_CANDIDATES)
    cat = first_match(chunks.columns, CATEGORY_CANDIDATES)
    print(f"Fields: chunk_id={cid}, text={txt}, aspect={asp}, rating={rat}, category={cat}")

    chunks = chunks.rename(columns={cid: "chunk_id", txt: "text"})
    chunks["chunk_id"] = chunks["chunk_id"].astype(str)
    chunks = chunks[~chunks["chunk_id"].isin(set(master["chunk_id"]))]
    chunks = chunks[~chunks["text"].isin(set(master["text"]))]
    words = chunks["text"].str.split().str.len()
    chunks = chunks[(words >= MIN_WORDS) & (words <= MAX_WORDS)].drop_duplicates(subset="text")
    if len(chunks) > MAX_POOL_ROWS:
        chunks = chunks.sample(MAX_POOL_ROWS, random_state=SEED)
    print(f"Candidate chunks after filtering: {len(chunks)}")

    if rat:
        # star rating is only a sampling proxy so the set is not all positive; annotators never see it
        r = pd.to_numeric(chunks[rat], errors="coerce")
        chunks = chunks[r.notna()].copy()
        chunks["_g"] = pd.cut(r[r.notna()], [-1, 2, 3, 99], labels=["low", "mid", "high"])
        parts = []
        for _, g in chunks.groupby("_g", observed=True):
            n_g = -(-need // chunks["_g"].nunique())
            parts.append(balanced_sample(g, n_g, asp) if asp else g.sample(min(len(g), n_g), random_state=SEED))
        sample = pd.concat(parts)
        if len(sample) > need:
            sample = sample.sample(need, random_state=SEED)
        elif len(sample) < need:
            rest = chunks.drop(sample.index)
            sample = pd.concat([sample, rest.sample(min(len(rest), need - len(sample)), random_state=SEED)])
    else:
        sample = balanced_sample(chunks, need, asp) if asp else chunks.sample(min(len(chunks), need), random_state=SEED)

    if len(sample) < need:
        print(f"WARNING: only found {len(sample)} usable chunks, still {need - len(sample)} short.")

    start = len(master) + 1
    new = pd.DataFrame({c: "" for c in master.columns}, index=range(len(sample)))
    new["id"] = [f"R{start + i:04d}" for i in range(len(sample))]
    new["chunk_id"] = sample["chunk_id"].values
    new["text"] = sample["text"].values
    if cat:
        new["target_category"] = sample[cat].astype(str).values
    if asp:
        new["target_aspects"] = sample[asp].astype(str).values   # keyword hint for annotators
    pd.concat([master, new], ignore_index=True).to_csv(MASTER_PATH, index=False)
    print(f"Appended {len(new)} rows. Master now has {len(master) + len(new)} rows. "
          f"These rows have no query_id or retrieval_relevance (they are not in the retrieval pool).")


def vader_band(texts):
    """Rough polarity pre-screen used ONLY to decide which chunks to sample (never as a label)."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        sys.exit("pip install vaderSentiment")
    sia = SentimentIntensityAnalyzer()
    out = []
    for t in texts:
        c = sia.polarity_scores(t)["compound"]
        out.append("positive" if c >= 0.35 else "negative" if c <= -0.05 else "neutral")
    return out


def rebalance():
    """Replace the spec rows with review chunks so the polarity classes are balanced.

    1. keeps every existing REVIEW row (with its labels), drops the *_spec rows;
    2. samples new REVIEW chunks from chunks.parquet (4.45M-chunk store), excluding anything already used;
    3. picks them to fill the gap to TARGET_SHARE, using a VADER pre-screen to find negative/neutral-looking text
       (a sampling aid only: every new row gets a blank label and must be annotated by humans);
    4. assigns each new row to the most suitable of the 20 queries (same category + aspect), else leaves query blank.
    """
    import pyarrow.parquet as pq
    m = pd.read_csv(MASTER_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    tcol = "text_clean" if "text_clean" in m.columns else "text"
    spec = m["chunk_id"].str.contains("_spec", regex=False)
    keep = m[~spec].copy()
    print(f"Existing master: {len(m)} rows; dropping {spec.sum()} spec rows; keeping {len(keep)} review rows")
    need = N_TARGET - len(keep)

    lab = keep["label"].where(keep["label"] != "", keep.get("annotator_1", "")) if "label" in keep else None
    have = lab.value_counts().to_dict() if lab is not None else {}
    gap = {c: max(0, round(TARGET_SHARE[c] * N_TARGET) - have.get(c, 0)) for c in TARGET_SHARE}
    tot = sum(gap.values()) or 1
    quota = {c: int(round(need * g / tot)) for c, g in gap.items()}
    quota["negative"] += need - sum(quota.values())          # absorb rounding
    print(f"Current review labels: {have}\nTo add ({need}) by VADER band: {quota}")

    names = pq.read_schema(CHUNKS_PATH).names
    cid = pick(names, CHUNK_ID_CANDIDATES, "chunk ID"); txt = pick(names, TEXT_CANDIDATES, "text")
    asp = first_match(names, ASPECT_CANDIDATES); cat = first_match(names, CATEGORY_CANDIDATES)
    typ = "chunk_type" if "chunk_type" in names else None
    asn = "parent_asin" if "parent_asin" in names else None
    cols = list(dict.fromkeys([c for c in (cid, txt, asp, cat, typ, asn) if c]))
    ch = pd.read_parquet(CHUNKS_PATH, columns=cols, filters=[(typ, "=", "review")] if typ else None)
    ch = ch.rename(columns={cid: "chunk_id", txt: "text"})
    ch["chunk_id"] = ch["chunk_id"].astype(str)
    if not typ:
        ch = ch[ch["chunk_id"].str.contains("_review", regex=False)]
    ch = ch.sample(min(len(ch), SCAN_ROWS), random_state=SEED)

    used_ids, used_txt = set(m["chunk_id"]), set(m[tcol])
    ch = ch[~ch["chunk_id"].isin(used_ids)]
    # strip the "[Product: ... | ASIN: ...]" prefix so the text is just the review
    body = ch["text"].str.replace(r"^\[Product:.*?\|\s*ASIN:[^\]]*\]\s*", "", regex=True)
    body = body.str.replace(r"<br\s*/?>", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    ch["product"] = ch["text"].str.extract(r"^\[Product:\s*(.*?)\s*\|\s*ASIN", expand=False).fillna("")
    ch["text_clean"] = body
    w = body.str.split().str.len()
    ch = ch[(w >= MIN_WORDS) & (w <= MAX_WORDS)]
    ch = ch[~ch["text_clean"].isin(used_txt)].drop_duplicates(subset="text_clean")
    ch["asin"] = ch[asn] if asn else ch["chunk_id"].str.split("_").str[0]
    print(f"Candidate review chunks after filtering: {len(ch)}")
    ch["band"] = vader_band(ch["text_clean"].tolist())

    picked = []
    for band, n in quota.items():
        pool = ch[ch["band"] == band]
        if len(pool) < n:
            print(f"WARNING: only {len(pool)} {band}-looking candidates for a quota of {n}; raise SCAN_ROWS")
        pool = pool.sample(frac=1, random_state=SEED)
        pool = pool.groupby("asin", sort=False).head(MAX_PER_ASIN)          # cap near-duplicate products
        if asp:                                                           # spread across aspects
            pool = balanced_sample(pool, n, asp)
        else:
            pool = pool.head(n)
        picked.append(pool.head(n))
    new = pd.concat(picked)
    if len(new) < need:                      # a band ran dry: fill from the remaining candidates, non-positive first
        rest = ch[~ch["chunk_id"].isin(set(new["chunk_id"]))]
        rest = rest.assign(_p=(rest["band"] == "positive").astype(int)).sort_values("_p", kind="stable")
        new = pd.concat([new, rest.groupby("asin", sort=False).head(MAX_PER_ASIN).head(need - len(new))])
    print(f"Selected {len(new)} new review chunks")

    # assign to a query: same category and sampled aspect appears in the query's target_aspects
    q = pd.read_json(QUERIES_PATH, lines=True)
    def assign(row):
        for _, qq in q.iterrows():
            aspects = [a.strip() for a in qq["target_aspects"].split(";")]
            if row.get(cat, "") == qq["target_category"] and row.get(asp, "") in aspects:
                return qq
        return None
    start = len(keep) + 1
    rows = []
    for i, (_, r) in enumerate(new.iterrows()):
        qq = assign(r) if (cat and asp) else None
        rows.append({
            "id": f"R{start + i:04d}", "query_id": qq["query_id"] if qq is not None else "",
            "chunk_id": r["chunk_id"], "product": r["product"], "text_clean": r["text_clean"],
            "intent": qq["intent"] if qq is not None else "",
            "target_category": r[cat] if cat else "", "target_aspects": r[asp] if asp else "",
            "difficulty": qq["difficulty"] if qq is not None else "", "retrieval_relevance": "",
            "aspect": "", "subjectivity": "", "annotator_1": "", "annotator_2": "", "agree?": "",
            "label": "", "llm_confidence": "", "notes": f"new_kb_sample; vader_band={r['band']}"})
    new_df = pd.DataFrame(rows)
    if "text" in keep.columns and "text_clean" not in keep.columns:
        keep["text_clean"] = keep["text"]
    out = pd.concat([keep, new_df], ignore_index=True)
    out = out[[c for c in new_df.columns if c in out.columns]]
    # shuffle the ORDER shown to annotators so new rows are not obviously a block, keep ids stable
    out.to_csv("master_annotation_rebalanced.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote master_annotation_rebalanced.csv ({len(out)} rows). The {len(new_df)} new rows are unlabeled: "
          f"annotate them (aspect, subjectivity, annotator_1, annotator_2, label), then run finalize.")


def finalize():
    from sklearn.metrics import cohen_kappa_score
    m = pd.read_csv(MASTER_PATH, dtype=str, encoding="utf-8-sig").fillna("")
    if "agree?" in m.columns:
        m = m.rename(columns={"agree?": "agree"})
    if "text_clean" in m.columns:
        m["text"] = m["text_clean"]
    done = m[(m["annotator_1"] != "") & (m["annotator_2"] != "")].copy()
    done["agree"] = (done["annotator_1"] == done["annotator_2"]).map({True: "yes", False: "no"})
    raw = (done["agree"] == "yes").mean()
    kappa = cohen_kappa_score(done["annotator_1"], done["annotator_2"])
    print(f"Annotated rows: {len(done)} / {len(m)}")
    print(f"Raw agreement: {raw:.1%}   Cohen's kappa: {kappa:.3f}   (target: raw >= 80%)")
    missing = done[done["label"] == ""]
    if len(missing):
        print(f"{len(missing)} rows still need an adjudicated 'label' (disagreements first).")
    m.update(done)
    m.to_csv(MASTER_PATH, index=False)

    final = m[m["label"] != ""][["id", "text", "subjectivity", "label"]]
    try:
        import xlwt  # written directly because recent pandas versions dropped .xls export
    except ImportError:
        print("Install xlwt (pip install xlwt), or open master_annotation.csv in Excel and Save As .xls.")
        return
    book = xlwt.Workbook()
    sheet = book.add_sheet("eval")
    for c, name in enumerate(final.columns):
        sheet.write(0, c, name)
    for r, row in enumerate(final.itertuples(index=False), start=1):
        for c, val in enumerate(row):
            sheet.write(r, c, str(val)[:32000])   # .xls cells cap at 32,767 characters
    book.save("eval.xls")
    print(f"Wrote eval.xls with {len(final)} rows. Label counts:\n{final['label'].value_counts()}")


if __name__ == "__main__":
    {"build": build, "topup": topup, "rebalance": rebalance, "finalize": finalize}.get(sys.argv[1] if len(sys.argv) > 1 else "",
                                              lambda: sys.exit(__doc__))()
