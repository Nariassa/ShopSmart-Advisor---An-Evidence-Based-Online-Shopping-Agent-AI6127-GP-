"""Full classification evaluation for the ShopSmart eval set.
Usage: python eval_full.py master_annotation_final.csv outdir [roberta]
Tasks: (A) polarity (positive/neutral/negative) on `label`; (B) subjectivity (subjective/objective).
Models: VADER, TF-IDF+LogReg, TF-IDF+LinearSVC (5-fold CV grouped by product), optional RoBERTa zero-shot.
"""
import sys, json, time, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, cohen_kappa_score)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); use_rob = len(sys.argv) > 3 and sys.argv[3] == "roberta"
d = pd.read_csv(src, dtype=str, keep_default_na=False, encoding="utf-8-sig")
import re, html
def clean(t):
    t = re.sub(r"^\[Product:.*?\|\s*ASIN:[^\]]*\]\s*", "", t); t = html.unescape(t)
    return re.sub(r"\s+", " ", re.sub(r"<br\s*/?>", " ", t)).strip()
d["text_clean"] = d.text_clean.map(clean)
X = d.text_clean.values; g = d.chunk_id.str.split("_").str[0].values
POL = ["negative", "neutral", "positive"]; SUB = ["objective", "subjective"]
rng = np.random.default_rng(0)

def cv_model(make, y, tgt):
    p = np.empty(len(d), dtype=object); t = time.time()
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        m = make().fit(v.fit_transform(X[tr]), y[tr]); p[te] = m.predict(v.transform(X[te]))
    return p, (time.time() - t) / len(d) * 1000

sia = SentimentIntensityAnalyzer()
def vader(th=0.05):
    t = time.time(); c = np.array([sia.polarity_scores(s)["compound"] for s in X])
    return c, (time.time() - t) / len(X) * 1000
comp, vms = vader()
vpol = np.where(comp >= .05, "positive", np.where(comp <= -.05, "negative", "neutral"))
vsub = np.where(np.abs(comp) >= .05, "subjective", "objective")

def boot(y, p, labs, n=1000):
    f = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); f.append(f1_score(y[i], p[i], labels=labs, average="macro", zero_division=0))
    return np.percentile(f, [2.5, 97.5])

def score(y, p, labs):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labs, zero_division=0)
    lo, hi = boot(y, p, labs)
    return dict(acc=accuracy_score(y, p), macro_f1=f1.mean(), f1_lo=lo, f1_hi=hi, per_class={l: dict(p=pr[i], r=rc[i], f1=f1[i], n=int(sup[i])) for i, l in enumerate(labs)},
                cm=confusion_matrix(y, p, labels=labs).tolist())

R = {}
for task, ycol, labs, vp in [("polarity", "label", POL, vpol), ("subjectivity", "subjectivity", SUB, vsub)]:
    y = d[ycol].values; preds = {"VADER (rules)": (vp, vms)}
    preds["TF-IDF + LogReg"] = cv_model(lambda: LogisticRegression(max_iter=3000, class_weight="balanced"), y, task)
    preds["TF-IDF + LinearSVC"] = cv_model(lambda: LinearSVC(class_weight="balanced"), y, task)
    if use_rob and task == "polarity":
        from transformers import pipeline
        pp = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", truncation=True, max_length=512)
        t = time.time(); r = np.array([x["label"].lower() for x in pp(list(X), batch_size=16)])
        preds["RoBERTa zero-shot (cardiffnlp)"] = (r, (time.time() - t) / len(X) * 1000)
    R[task] = {n: dict(score(y, p, labs), ms=ms) for n, (p, ms) in preds.items()}
    for n, (p, _) in preds.items(): d[f"pred_{task}_{n.split()[0].lower()}{'_' + n.split()[-1].lower() if 'TF' in n else ''}"] = p

# agreement
R["agreement"] = dict(raw=float((d.annotator_1 == d.annotator_2).mean()), kappa=float(cohen_kappa_score(d.annotator_1, d.annotator_2)),
                      n_disagree=int((d.annotator_1 != d.annotator_2).sum()))
# per-aspect accuracy of best polarity model
best = max(R["polarity"], key=lambda k: R["polarity"][k]["macro_f1"]); R["best_polarity"] = best
bp = d[[c for c in d.columns if c.startswith("pred_polarity")]].iloc[:, 0]
json.dump(R, open(out + "/results.json", "w"), indent=1, default=float)
d.to_csv(out + "/predictions.csv", index=False, encoding="utf-8-sig")

# figures
fig, ax = plt.subplots(1, len(R["polarity"]), figsize=(4 * len(R["polarity"]), 3.6))
for a, (n, r) in zip(np.atleast_1d(ax), R["polarity"].items()):
    cm = np.array(r["cm"]); a.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3): a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_xticks(range(3), ["neg", "neu", "pos"]); a.set_yticks(range(3), ["neg", "neu", "pos"])
    a.set_title(n, fontsize=9); a.set_xlabel("predicted"); a.set_ylabel("gold")
plt.tight_layout(); plt.savefig(out + "/confusion_polarity.png", dpi=160); plt.close()
fig, a = plt.subplots(figsize=(6, 3.4)); names = list(R["polarity"])
a.bar(np.arange(len(names)) - .2, [R["polarity"][n]["macro_f1"] for n in names], .4, label="polarity")
a.bar(np.arange(len(names)) + .2, [R["subjectivity"].get(n, {"macro_f1": 0})["macro_f1"] for n in names], .4, label="subjectivity")
a.set_xticks(range(len(names)), [n.replace(" (", "\n(") for n in names], fontsize=7); a.set_ylabel("macro-F1"); a.set_ylim(0, 1); a.legend()
plt.tight_layout(); plt.savefig(out + "/macro_f1.png", dpi=160); plt.close()
print(json.dumps({t: {n: (round(r["acc"], 3), round(r["macro_f1"], 3)) for n, r in R[t].items()} for t in ["polarity", "subjectivity"]}, indent=1)); print(R["agreement"])
