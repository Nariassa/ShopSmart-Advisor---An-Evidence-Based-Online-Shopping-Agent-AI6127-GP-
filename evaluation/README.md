# Human Relevance Evaluation Package

## Status

The baseline pipeline-validation stage and the formal 20-query retrieval runs
are complete. This directory contains the human-evaluation package for replacing
silver labels with graded manual relevance judgments.

On October 7, 2026, a project member completed a full review of all 791 labels
and confirmed the final judgments. The current workbook and qrels use the
identifier `Human_Reviewer_01`. The labels are ready for formal single-reviewer
retrieval evaluation. A second independent annotator was not used, so
inter-annotator agreement is not reported.

## Files

- `formal_queries_v1.jsonl` contains 20 frozen natural-language Amazon
  Electronics queries covering recommendation, aspect, and support-quality
  intents.
- `QRELS_ANNOTATION_GUIDE.md` defines the 0/1/2 relevance rubric, annotation
  rules, and quality-control process.
- `Amazon_Electronics_Qrels_Annotation.xlsx` is the earlier nine-query pilot
  annotation workbook.
- `Amazon_Electronics_Final_Qrels_Double_Annotation.xlsx` is the final
  20-query double-annotation workbook.
- `human_validated_annotations_v1.jsonl` contains the current 791 human-confirmed
  labels, reasons, confidence values, identifiers, and source-text hashes.
- `human_validation_manifest.json` records full review scope, counts, and hashes.
- `qrels_human_validated_v1.txt` contains the current graded qrels in TREC format.
- Earlier annotation ledgers are retained as internal version history and are
  not the current evaluation source.
- `formal_annotation_pool.jsonl` contains 791 blinded query--chunk pairs.
- `formal_annotation_pool_manifest.json` records pooling statistics and lineage
  checks.
- `annotation_pool_manifest.json` records the earlier pilot workbook scope.
- `bm25_pilot_queries.jsonl` is the original nine-query pipeline-validation set.

## Final Workbook Structure

- `Overview` shows annotation, disagreement, adjudication, and final-label
  progress.
- `Annotations` contains the blinded 791-row pool, two independent judgment
  blocks, adjudication fields, and calculated final labels.
- `Queries` contains the frozen query set and candidate-pool statistics.
- `Provenance` stores all four baseline ranks and scores separately to reduce
  anchoring bias during annotation.

## Annotation Procedure

1. Read `QRELS_ANNOTATION_GUIDE.md` before editing the workbook.
2. Give each annotator an independent copy or controlled view of the workbook.
3. Filter `Annotations` by `query_id` and label each query--chunk pair.
4. Enter a relevance label, annotator identifier, and confidence value.
5. Use the adjudication fields only when `adjudication_required` is `Yes`.
6. Preserve both original labels. Use `final_label` for final metrics.

## Current Scope and Next Step

The formal pool is the deduplicated union of the top 20 candidates from BM25,
dense, hybrid RRF, and cross-encoder results. It contains 791 pairs across 20
queries. The next step is independent annotation and adjudication. Final report
metrics must use `final_label`, not the silver labels used for pipeline
validation.
