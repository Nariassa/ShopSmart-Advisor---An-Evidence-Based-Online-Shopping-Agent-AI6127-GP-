# AI6127 Amazon Electronics: Grounded Generation & Purchase Confidence Scoring

This repository converts Amazon Electronics product specifications and review evidence into traceable English shopping answers and calculates a Purchase Confidence Score (PCS).

It runs Qwen2.5 7B and Llama 3.1 8B locally through Ollama and provides product-type filtering, evidence selection, single-fact rewriting, rule-based and NLI validation, source citations, and evaluation-ready generation outputs.

The language model selects evidence and rewrites conclusions. Application code controls product identity, citation offsets, answer structure, PCS, and recommendation wording. Model weights and the complete knowledge base remain local.

## Project scope

The generation work covers the following tasks:

1. Integrate local Qwen2.5 7B and Llama 3.1 8B through Ollama.
2. Design evidence-constrained prompts with explicit product types, requirements, and exclusions.
3. Select fact IDs and product names, then resolve original quotations and sources in code.
4. Generate one English conclusion per fact and validate it with rules and local NLI.
5. Calculate PCS and expose interfaces for aspect sentiment, polarization, and coverage.
6. Run two-model generation tests with per-query outputs, failures, and reproducibility records.
7. Export blinded review packets, blank scoring templates, and RAGAS inputs.

The project uses the following module names throughout its documentation:

| Module | Responsibility |
| --- | --- |
| Knowledge Base Construction & Data Quality | Corpus cleaning, chunking, indexing, and corpus statistics |
| Retrieval, Reranking & Evaluation | BM25, dense and hybrid retrieval, reranking, and retrieval evaluation |
| Grounded Generation & Purchase Confidence Scoring | Local models, prompts, grounding, citations, PCS, and generation testing |
| Aspect Analysis & Purchase Decision Explanation | Aspect sentiment, polarization, coverage, deal-breakers, and alternatives |
| System Integration, User Interface & Evaluation | UI, annotation coordination, human evaluation, IAA, RAGAS, ablations, performance, and report compilation |

## Main result

The revised development generation test used 20 questions with two models, completing all 40 attempts.

| Model | Generated answers | Citation checks | Accepted rewrite proposals | NLI-rejected proposals | Guard-rejected proposals | Mean question time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5 7B | 20/20 | 109/109 | 44 | 9 | 45 | 13.87 s |
| Llama 3.1 8B | 20/20 | 124/124 | 46 | 9 | 69 | 15.87 s |

A generated answer means that the model completed valid evidence selection and the pipeline produced a checked answer. It does not mean that the answer passed human quality evaluation. Final answers combine verified model rewrites with original quotations retained or supplemented by application rules.

Accepted rewrite proposals are counted at the validation stage. Final merging can remove duplicates or apply answer-length limits, so these counts differ from the number of rewrites displayed. Rejection counts describe proposals, not failed answers or hallucination rates.

All 233 citations match the frozen sources' product identity, original text, and character offsets. All 20 two-model pairs use the same candidate evidence, selection prompt, and PCS. Citation validation establishes mechanical traceability, not semantic accuracy.

Both local models can run this pipeline. The current results do not establish which model produces better answers. Timings include local loading and processing and are not a controlled performance benchmark.

## Before-and-after comparison

| Check | Before | After |
| --- | --- | --- |
| Qwen generated answers | 20/20 | 20/20 |
| Llama generated answers | 18/20 | 20/20 |
| Product-name mismatch rejections | 2 | 0 |
| Answers containing HTML fragments | 13 | 0 |
| Speaker carrying case included for Q003 | Both models | Removed from both |
| Missing requested-aspect tag flags | 20 answers | 24 answers |

The two original Llama failures came from inconsistent output limits: the prompt requested at most eight selections, but the schema allowed 24. Both selection and rewriting now use the configured limit of eight in prompts, schemas, and runtime validation.

The revision also filters product types before ranking, context construction, and rule supplements; checks attribution and qualifications in rewrites; and removes HTML from rendered text only.

Questions, all 791 raw candidates, candidate order, model weights, and the PCS timestamp remain fixed. Product filtering changes the facts and products actually used. This is a combined pipeline regression, not an isolated prompt ablation. Missing-aspect flags use labels and cannot be interpreted directly as semantic coverage changes.

## Data and test design

The upstream knowledge base contains 4,450,749 specification and review chunks. This test uses 20 existing questions and 791 query–chunk candidate pairs covering 790 unique chunks.

Candidate text was hydrated from Parquet, and all 791 text SHA-256 checks matched. Candidates remain in their original display_order. Human relevance_label, confidence, and reason fields are not used to filter or rerank candidates and are not passed to the model. Product filtering, fact extraction, and context limits subsequently determine the facts in the selection prompt.

Upstream relevance judgments assess whether retrieved chunks answer questions; they are not human support labels for generated claims. Generation scoring sheets remain blank.

This batch did not consume the Retrieval, Reranking & Evaluation module's frozen Dense top-k rankings. display_order is annotation presentation order. The questions have been used during generation development, so the batch is a fixed candidate-pool replay development test, not an independent test set or formal end-to-end evaluation.

## Generation pipeline

1. **Normalize evidence:** validate chunk_id, parent_asin, text, and offsets while distinguishing specifications from reviews.
2. **Identify requirements:** extract product types and explicit conditions while retaining the full question. Missing feature or compatibility evidence does not establish satisfaction.
3. **Filter products:** conservative title rules exclude mismatched or unverified product types and record excluded_products. Rule supplements use only the retained evidence.
4. **Select facts:** the model returns only fact_id and the complete product_name. Code verifies their relationship and resolves source text and citations.
5. **Rewrite individual facts:** each English conclusion corresponds to one fact. Review rewrites start with “One reviewer reports that” and retain personal scope, negation, quantities, units, and conditions.
6. **Validate and retain evidence:** guards run before NLI checks the proposition after the attribution prefix. Rejected rewrites retain verified original quotations.
7. **Build the answer:** code supplements available opposing evidence, calculates PCS and recommendation wording, and renders the answer with sources.

Product-name comparison normalizes only Unicode NFC, HTML entities, and whitespace. Words, model numbers, and suffixes remain significant. HTML cleaning affects the rendered view only; citation.quote, sources.text, start, and end remain unchanged.

## PCS design

PCS is an experimental evidence-confidence measure, not a probability of purchase satisfaction. Its weights and thresholds are implementation choices that remain uncalibrated:

~~~text
PCS = 100 × V × C × (0.40Q + 0.35S + 0.25R)
V   = n / (n + 5)
S   = 1 − P
P   = Σ_a w_a × 4 × p_positive,a × p_negative,a
C   = Σ_a w_a × min(n_a / 2, 1)
~~~

| Symbol | Meaning |
| --- | --- |
| Q | Evidence quality from text substance, purchase verification, and helpful votes |
| S / P | Opinion consistency / polarization within requested aspects |
| R | Recency with a default 180-day half-life |
| V | Evidence-volume gate that limits confidence from small samples |
| C | Evidence coverage of the requested aspects |
| n / n_a | Total deduplicated review evidence units / units for aspect a |
| w_a | Normalized weight for requested aspect a |

Missing review timestamps use a disclosed recency value of 0.5. Missing purchase verification or helpful votes earn no bonus. Without original review IDs, deduplicated text units are not independent user counts. Star ratings do not substitute for aspect sentiment.

Consistently negative reviews can have high PCS. Separate fit_score, evidence-sufficiency, and budget checks therefore control recommendation wording. The Aspect Analysis & Purchase Decision Explanation interface is available but its formal outputs were not used in this batch. See [PCS_DESIGN.md](docs/PCS_DESIGN.md).

## Repository layout

~~~text
.
|-- src/generation/
|   |-- generator.py              Pipeline, supplements, and rendering
|   |-- prompts.py                Selection/rewrite prompts and dynamic schemas
|   |-- product_match.py          Requirements and product-type filtering
|   |-- grounding.py              Quotes, guards, NLI, and citation validation
|   |-- pcs.py                    PCS and aspect-analysis signal integration
|   |-- sentiment.py              Sentiment baseline and external labels
|   |-- llm.py                    Local Ollama interface
|   |-- evidence.py               Evidence normalization and source offsets
|   |-- adapters.py               Parquet hydration and retrieval integration
|   |-- batch.py                  Retrieval-to-generation evaluation exports
|   |-- testing.py                Frozen inputs, checkpoints, and test exports
|   '-- __main__.py               Single-question CLI
|-- scripts/                      Audits, revision comparison, and challenge cases
|-- tests/                        55 automated tests
|-- examples/                     Explicitly synthetic demonstration fixtures
|-- outputs/*/queries.jsonl       Frozen generation-test query snapshots
|-- docs/                         PCS, generation, and test documentation
|-- outputs/                      Frozen local experiments
|-- requirements-generation.txt
'-- README.md
~~~

Upstream retrieval code, retrieval requirements.txt, the standalone formal query set, and human annotations are intentionally omitted from this generation-only upload. Live retrieval requires src/retrieval/, its dependencies, and local indexes from the Retrieval, Reranking & Evaluation repository. Generating from complete evidence files does not require loading those indexes. Historical directory names and serialized identifiers remain unchanged for compatibility and experiment traceability; they are not module display names.

## Requirements

- Python 3.10 or a compatible newer release; the tested environment uses Python 3.12.
- Ollama with qwen2.5:7b and llama3.1:8b.
- torch, transformers, and sentencepiece for NLI.
- pyarrow for Parquet access; complete JSON evidence avoids full-corpus reads.
- Internet access for first-time model downloads. GPU execution is optional; this batch used an RTX 4060 Laptop GPU for Ollama and CPU NLI.

These commands create a repository-local environment. In the existing development setup, the environment is in the parent directory, so replace the interpreter path with ../.venv/Scripts/python.exe.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-generation.txt
.\.venv\Scripts\python.exe -m pip install torch transformers sentencepiece
~~~

Install and start Ollama separately, then download the models:

~~~powershell
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama list
~~~

Transformers downloads cross-encoder/nli-deberta-v3-small on first use unless it is cached. Actual package versions, Ollama versions, and model digests are recorded with the results; the installation commands do not provide a fully locked environment.

## Quick start

Check the pipeline with explicitly synthetic evidence:

~~~powershell
.\.venv\Scripts\python.exe -m src.generation --demo --query "Compare these headphones for battery life and comfort."
~~~

Run local Qwen with NLI:

~~~powershell
.\.venv\Scripts\python.exe -m src.generation --demo --query "Compare these headphones for battery life and comfort." --backend ollama --model qwen2.5:7b --nli-model cross-encoder/nli-deberta-v3-small --max-claims 8 --require-llm --output outputs/demo_qwen.json
~~~

Change --model to llama3.1:8b to use the other model. Synthetic demonstrations are not empirical Amazon quality measurements.

Use real evidence exported by the Retrieval, Reranking & Evaluation module:

~~~powershell
.\.venv\Scripts\python.exe -m src.generation --evidence path/to/retrieved_evidence.json --query "Which portable speaker has strong bass?" --backend ollama --model qwen2.5:7b --nli-model cross-encoder/nli-deberta-v3-small --max-claims 8 --require-llm --output outputs/real_answer.json
~~~

For ID-only evidence, add --chunks-path path/to/chunks.parquet to hydrate source text.

## Reproducing the generation test

The final batch is stored in outputs/member_c_generation_testing_revised_final_2026-10-15/.

The runner needs the old baseline's manifest.json, evidence_map.json, and input_provenance.json plus the same query file. Code and summary statistics alone cannot reproduce the batch.

~~~powershell
.\.venv\Scripts\python.exe -m src.generation.testing --baseline-dir outputs/member_c_generation_testing_2026-10-15 --output-dir outputs/member_c_generation_testing_reproduction --queries outputs/member_c_generation_testing_2026-10-15/queries.jsonl --annotations path/to/human_validated_annotations_v1.jsonl
~~~

On initial preparation, --annotations supplies the source file hash; candidate text comes from the frozen baseline map, without label-based selection. Obtain the same annotation version from the Retrieval, Reranking & Evaluation repository and pass its path explicitly. The command uses the retained frozen query snapshot rather than an omitted upstream query file. Each question has a checkpoint; repeating the command resumes matching checkpoints without overwriting saved results.

| Setting | Value |
| --- | --- |
| Models | qwen2.5:7b; llama3.1:8b |
| generation_revision | bounded_type_scope_display_v3_attribution_aware_nli |
| Prompt base | v1 with the current fixes, not unchanged historical v1 |
| temperature / seed | 0 / 42 |
| num_ctx / num_predict | 8192 / 2048 |
| max_products / max_facts / max_claims | 3 / 18 / 8 |
| max_context_chars | 8000 |
| NLI / threshold | cross-encoder/nli-deberta-v3-small / 0.9 |
| NLI device / threads | CPU / 4 |
| Ollama keep_alive | 2m; unload in-memory weights after each model's batch |

Changes to code, inputs, or model digests cannot be mixed with saved checkpoints. Use a new output directory for changed code. Archived snapshots and records retain their original bytes and identifiers; current documentation changes do not rewrite experimental history.

## Tests and audits

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/audit_generation_handoff.py outputs/member_c_generation_testing_revised_final_2026-10-15
.\.venv\Scripts\python.exe scripts/compare_generation_revision.py
.\.venv\Scripts\python.exe scripts/run_rewrite_challenge.py
~~~

The 55 automated tests cover output contracts, filtering and supplements, fabricated facts, citation offsets, scope and negation, NLI fallback, PCS, model calls, and export consistency. The final artifact audit passed 27 checks.

All 15 developer-authored challenge cases matched expectations with real local NLI and guards. These small development diagnostics are not a formal accuracy estimate. Comparison and challenge scripts use the documented local result directories.

## Key outputs

The final result directory contains:

| File | Purpose |
| --- | --- |
| summary.md / summary.json | Batch and model statistics |
| revision_comparison.md / revision_comparison.json | Before-and-after comparison |
| revision_pairs.csv | Per-query outcomes, products, and actual fact changes |
| results_all.jsonl / runs/ | Raw calls, claims, citations, PCS, and checkpoints |
| qwen2.5_7b_answers.md / llama3.1_8b_answers.md | English answers by model |
| evidence_map.json / queries.jsonl | Frozen candidates and questions |
| ragas_inputs.jsonl | Actual selection-model facts and final answers, without invented references |
| blinded_review_packets.jsonl / blinded_review_answers.md | Model-anonymized review materials |
| human_review_template.csv / claim_support_review_template.csv | Blank answer-level and claim-level scoring sheets |
| model_key_for_E.csv | Model mapping, held separately by System Integration, User Interface & Evaluation |
| manifest.json / code_snapshot/ | Frozen configuration, hashes, and code |
| model_digests.json / runtime_versions.json | Model and software versions |
| artifact_audit.json / SHA256SUMS.txt | Integrity audit and file checksums |
| rewrite_challenge.json | Results for the 15 diagnostic cases |
| QUALITY_REVIEW.md / E_REVIEW_GUIDE.md | Remaining issues and review instructions |

The complete final handoff is also available as outputs/member_c_generation_testing_revised_final_2026-10-15.zip. GitHub publication remains a separate local packaging step.

## Limitations

- The 20 development questions are not a broad benchmark or independent generation test set.
- This batch replays a candidate pool, not formal Dense top-k rankings.
- Title-based type filtering can exclude valid products or miss accessories; matching a type does not verify suitability or compatibility.
- Guards check explicit words and numbers, not complete negation scope, comparisons, written-out number words, complex units, or pronoun resolution.
- Revised Qwen Q004/F1 weakens “not as good as SP3” despite passing NLI.
- Conservative guards can reject correct rewrites and increase quotation fallback. The 114 guard rejections are not 114 established hallucinations.
- Lexicon sentiment can misread sarcasm, and single-aspect tags can undercount multi-aspect evidence. Formal aspect-analysis signals were not used.
- PCS weights, recommendation thresholds, and missing-metadata defaults are uncalibrated.
- Formal human scoring, IAA, and RAGAS metrics have not been computed. Exported scoring sheets remain blank.
- These results do not establish model quality rankings or controlled speed comparisons.

## Downstream handoff

Retrieval, Reranking & Evaluation recommends Dense as the default evidence source. Grounded Generation & Purchase Confidence Scoring should consume real Dense top-k evidence while preserving chunk_id, parent_asin, chunk_type, aspect, text, rank, and retrieval score, and run it under a separate experiment condition.

Aspect Analysis & Purchase Decision Explanation supplies validated aspect sentiment, polarization, and coverage signals. Grounded Generation & Purchase Confidence Scoring validates those interfaces and calculates PCS. System Integration, User Interface & Evaluation receives answers, contexts, and templates for independent human evaluation, IAA, and applicable RAGAS metrics. Keep model_key_for_E.csv away from blinded annotators until independent scoring is complete.

Do not combine new retrieval-based results with this candidate-pool replay when reporting metrics.
