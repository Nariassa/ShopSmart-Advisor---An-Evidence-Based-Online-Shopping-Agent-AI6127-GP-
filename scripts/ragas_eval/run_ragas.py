"""RAGAS + latency/throughput/cost evaluation for ShopSmart Advisor.

Pipeline per query: take the Dense top-K chunks (the recommended retriever) -> LLM writes an evidence-grounded
answer with [C#] citations -> RAGAS scores faithfulness, answer relevancy, context precision, context recall.

Run from the repo root (needs results/ and evaluation/ from the project):
    pip install ragas langchain-openai openai pandas
    export OPENAI_API_KEY=...            # or point OPENAI_BASE_URL at any OpenAI-compatible endpoint
    python run_ragas.py --dry-run        # builds the dataset only, no API calls
    python run_ragas.py --k 5 --gen-model gpt-4o-mini --judge-model gpt-4o-mini --n-queries 20

Outputs (in --out): answers.jsonl, ragas_per_query.csv, ragas_summary.csv, efficiency.csv, ragas_report.md
"""
import argparse, json, os, time, statistics as st
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--results", default="results/formal_dense_results.jsonl")
p.add_argument("--qrels", default="evaluation/qrels_human_validated_v1.txt")
p.add_argument("--pool", default="evaluation/formal_annotation_pool.jsonl")
p.add_argument("--out", default="ragas_out")
p.add_argument("--k", type=int, default=5, help="contexts given to the generator")
p.add_argument("--n-queries", type=int, default=20)
p.add_argument("--gen-model", default="gpt-4o-mini")
p.add_argument("--judge-model", default="gpt-4o-mini")
p.add_argument("--embed-model", default="text-embedding-3-small")
p.add_argument("--price-in", type=float, default=0.15, help="USD per 1M input tokens (generator)")
p.add_argument("--price-out", type=float, default=0.60, help="USD per 1M output tokens (generator)")
p.add_argument("--dry-run", action="store_true")
p.add_argument("--reuse-answers", action="store_true", help="skip generation; load answers.jsonl and efficiency.csv from --out")
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

def strip(t):  # remove the "[Product: ... | ASIN: ...]" prefix for the evidence shown to the LLM, keep title separately
    if t.startswith("[Product:") and "]" in t:
        head, body = t.split("]", 1); return head.split("|")[0].replace("[Product:", "").strip(), body.strip()
    return "", t.strip()

runs = [json.loads(l) for l in open(a.results, encoding="utf-8")][: a.n_queries]
qrels = {}
for l in open(a.qrels):
    q, _, c, g = l.split(); qrels.setdefault(q, {})[c] = int(g)
pool = {}
for l in open(a.pool, encoding="utf-8"):
    r = json.loads(l); pool[(r["query_id"], r["chunk_id"])] = r["text"]

SYSTEM = ("You are ShopSmart Advisor. Answer the shopper's question using ONLY the numbered evidence. "
          "Recommend at most 3 products, name them, say why with specifics, mention one drawback if the evidence shows one, "
          "and cite evidence as [C1], [C2]. If the evidence is insufficient, say so. Do not use outside knowledge.")

rows = []
for r in runs:
    ctx_chunks = r["results"][: a.k]
    contexts = [f"{strip(c['text'])[0]}: {strip(c['text'])[1]}" for c in ctx_chunks]
    # reference for context recall: text of the human-validated grade-2 chunks for this query (proxy for a gold answer)
    gold = [pool[(r["query_id"], c)] for c, g in qrels.get(r["query_id"], {}).items() if g == 2 and (r["query_id"], c) in pool]
    reference = " ".join(f"{strip(t)[0]}: {strip(t)[1]}" for t in gold[:8]) or "(no grade-2 evidence)"
    rows.append(dict(query_id=r["query_id"], user_input=r["query"], retrieved_contexts=contexts, reference=reference,
                     retrieval_latency_s=r.get("latency_seconds")))
print(f"{len(rows)} queries; avg contexts {st.mean(len(x['retrieved_contexts']) for x in rows):.1f}; "
      f"queries without a gold reference: {sum(x['reference'].startswith('(no') for x in rows)}")
if a.dry_run:
    json.dump(rows[:2], open(out / "dry_run_sample.json", "w"), indent=1); print("dry run ok ->", out / "dry_run_sample.json"); raise SystemExit

if a.reuse_answers:
    rows = json.load(open(out / "answers.jsonl"))
    eff = pd.read_csv(out / "efficiency.csv").to_dict("records")
    print(f"Reusing {len(rows)} saved answers")
else:
    from openai import OpenAI
    client = OpenAI()
    eff = []
    for x in rows:
        ev = "\n".join(f"[C{i+1}] {c}" for i, c in enumerate(x["retrieved_contexts"]))
        t = time.perf_counter()
        resp = client.chat.completions.create(model=a.gen_model, temperature=0, messages=[
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Question: {x['user_input']}\n\nEvidence:\n{ev}"}])
        dt = time.perf_counter() - t
        x["response"] = resp.choices[0].message.content
        u = resp.usage
        eff.append(dict(query_id=x["query_id"], generation_s=dt, in_tokens=u.prompt_tokens, out_tokens=u.completion_tokens,
                        cost_usd=u.prompt_tokens * a.price_in / 1e6 + u.completion_tokens * a.price_out / 1e6,
                        retrieval_s=x["retrieval_latency_s"]))
        print(x["query_id"], f"{dt:.1f}s", x["response"][:80].replace("\n", " "))
    json.dump(rows, open(out / "answers.jsonl", "w"), indent=1)
    pd.DataFrame(eff).to_csv(out / "efficiency.csv", index=False)   # saved now so a later RAGAS error loses nothing

from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
llm = LangchainLLMWrapper(ChatOpenAI(model=a.judge_model, temperature=0))
emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=a.embed_model))
ds = EvaluationDataset.from_list([{k: x[k] for k in ("user_input", "retrieved_contexts", "response", "reference")} for x in rows])
res = evaluate(ds, metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()], llm=llm, embeddings=emb)
df = res.to_pandas(); df.insert(0, "query_id", [x["query_id"] for x in rows]); df.to_csv(out / "ragas_per_query.csv", index=False)
metric_cols = [c for c in df.columns if c in ("faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall")]
summ = df[metric_cols].agg(["mean", "std", "min", "max"]).T.round(3); summ.to_csv(out / "ragas_summary.csv")

e = pd.DataFrame(eff); e.to_csv(out / "efficiency.csv", index=False)
total = e.generation_s.sum() + e.retrieval_s.fillna(0).sum()
ef = {"queries": len(e), "retrieval_s_mean": e.retrieval_s.mean(), "generation_s_mean": e.generation_s.mean(),
      "generation_s_p95": e.generation_s.quantile(.95), "end_to_end_s_mean": (e.generation_s + e.retrieval_s.fillna(0)).mean(),
      "throughput_queries_per_min_sequential": len(e) / total * 60,
      "tokens_in_mean": e.in_tokens.mean(), "tokens_out_mean": e.out_tokens.mean(),
      "cost_per_query_usd": e.cost_usd.mean(), "cost_total_usd": e.cost_usd.sum()}
pd.Series(ef).round(4).to_csv(out / "efficiency_summary.csv", header=["value"])
(out / "ragas_report.md").write_text(
    f"# RAGAS and efficiency results\n\nGenerator `{a.gen_model}`, judge `{a.judge_model}`, k={a.k}, {len(rows)} queries.\n\n"
    f"## RAGAS (mean over queries)\n\n{summ.to_markdown()}\n\n## Efficiency\n\n{pd.Series(ef).round(4).to_markdown()}\n\n"
    "Notes: context recall uses the human-validated grade-2 chunks as the reference (a proxy for a gold answer). "
    "Retrieval latency comes from the project's formal retrieval log. Throughput is sequential, single-process.\n", encoding="utf-8")
print(summ); print(pd.Series(ef).round(3))
