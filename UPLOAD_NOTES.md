# Generation-only GitHub upload

This folder contains Grounded Generation & Purchase Confidence Scoring code, adapters, tests, synthetic examples, selected documentation, and two complete development experiment archives. The original project is unchanged.

## Upstream files omitted

The retrieval implementation (src/retrieval/), its requirements.txt, evaluation/formal_queries_v1.jsonl, and evaluation/human_validated_annotations_v1.jsonl belong to the upstream Retrieval, Reranking & Evaluation work and are not duplicated here. Obtain them from that repository when needed. Live retrieval requires the upstream implementation and indexes. Evidence-file generation, synthetic demonstrations, and generation unit tests do not require live retrieval.

For reproduction, use the command in README.md: --queries points to the retained baseline queries.jsonl; --annotations must point to the same upstream annotation version. Defaults in the historical runner assume the original combined workspace.

## Retained experiment records

Frozen queries.jsonl, evidence_map.json, input_provenance.json, code snapshots, and archived requirements files are retained inside outputs/ to preserve generation experiment provenance and integrity. These are experiment records, not additional upstream datasets or active retrieval code. Their original bytes and internal checksums are unchanged. Local paths and historical identifiers remain in frozen records.

Complete experiments include model_key_for_E.csv and model-labelled outputs. Keep them separate from blinded annotators until independent scoring is complete.

Models, indexes, caches, interrupted runs, and old smoke runs are excluded. The existing outputs.zip is a separate local archive; .gitignore excludes ZIP files. Archived references to other handoff ZIP files refer to local archives, not required Git inputs.

COPY_MANIFEST.json lists retained source copies, hashes, intentional publication edits, and removed upstream files. This folder has not been uploaded to GitHub.
