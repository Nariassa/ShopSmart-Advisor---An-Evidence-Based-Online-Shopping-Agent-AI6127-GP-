# Qrels Annotation Guide

## Purpose

This guide defines the manual relevance labels used to evaluate retrieval quality for the Amazon Electronics knowledge base. The annotation unit is a query--chunk pair. Labels must reflect whether the chunk itself provides evidence that helps answer the query.

## Relevance Scale

- **2 -- Highly relevant:** The chunk directly addresses the requested product type and the important aspect or constraint. It contains specific evidence that could support a recommendation or answer.
- **1 -- Partially relevant:** The chunk is related to the requested product type or aspect, but the evidence is incomplete, indirect, generic, or only addresses part of a multi-aspect query.
- **0 -- Not relevant:** The chunk concerns the wrong product type, does not address the requested need, is too vague to be useful, or is otherwise unrelated.

## Core Rules

1. Judge only the text shown in the candidate row. Do not infer missing properties from outside knowledge.
2. Product-type match is necessary for recommendation queries. A strong aspect match for the wrong product type is still label 0.
3. For multi-aspect queries, label 2 requires useful evidence for the product type and at least one central aspect. Evidence can support or contradict the desired attribute, as specified in rule 4. Use label 1 when only a secondary aspect is addressed.
4. Positive and negative evidence can both be relevant. For example, a review reporting frequent connection drops is relevant to a connectivity query even though the sentiment is negative.
5. A specification chunk can receive label 2 when its fields directly establish the requested capability. A vague review such as "works great" should normally receive label 1 or 0 depending on context.
6. Duplicate or near-duplicate chunks must be judged independently but consistently.
7. Ignore baseline ranks and scores during judgment. They are stored on a separate provenance sheet to reduce anchoring bias.

## Annotation Workflow

1. Open the `Annotations` sheet and filter by `query_id`.
2. Read the query, then inspect each candidate chunk.
3. Enter `0`, `1`, or `2` in your assigned `annotator_1_label` or `annotator_2_label` column.
4. Enter your identifier in the matching annotator ID column and select a confidence value.
5. Use the matching notes column for supporting evidence, ambiguity, suspected data errors, or adjudication needs.
6. Do not edit identifiers, query text, candidate text, or provenance fields.

## Quality Control

- Double-annotate at least 20% of the query--chunk pairs.
- Resolve disagreements through adjudication and record the final label separately before computing metrics.
- Report inter-annotator agreement before adjudication.
- Keep the original individual labels so agreement can be reproduced.
- Freeze the query set and relevance guidelines before comparing final systems.

## Current Package Scope

The formal workbook contains 791 unique query--chunk pairs pooled from all four completed retrieval systems for the 20 formal queries. The earlier 180-row pilot workbook is retained separately. The formal workbook supports two judgments per candidate and adjudication of disagreements.

As of October 7, 2026, a project member has reviewed and confirmed all 791 first-annotator judgments. The current identifier is `Human_Reviewer_01`, and these labels are the formal single-reviewer qrels. A second independent annotator was not used, so inter-annotator agreement is not reported.
