# Reproducibility corrections — 2026-09-08

The code and documentation were repaired after checking the project rules, saved metrics, current notebooks and local experiment artifacts. Historical datasets, checkpoints and numerical run reports were retained. These changes are local until committed and published.

| Problem | Correction |
|---|---|
| Smoke results could be reused and relabeled by a full request | Immutable run identities; verify before writing; reject incompatible or legacy resume |
| Training could select an epoch on test | Train/select with validation; test is evaluation-only for trainable models |
| Partial resume and QLoRA adapter provenance were incomplete | Check checkpoint/history/config hashes, restore RNG, verify adapter identity |
| Direct dependencies and research setup were not reproducible | Pinned CPU/research/VLM requirements, tested Python 3.11 reference freeze |
| Registry/executor referred to obsolete notebook architectures | Register the actual 18 notebooks, configure isolated copies and record seeds/settings/execution |
| Raw AI2D preparation and OCR recipe were absent | Audited raw-to-OCR-to-manifest command with provenance and explicit missing-file handling |
| Blank distractors shifted an AI2D answer index | Preserve official option slots; rebuild as a distinct data version |
| Paths depended on a particular working directory | Explicit dataset/run/cache roots and portable image/OCR resolution |
| Main tables used incomplete/stale run counts and omitted OCR retrieval | Generate all current rows from checked evidence; include the deterministic OCR baseline |
| Saved attention evaluation files were ignored by Git | Add compact portable evaluation evidence with canonical JSON hashes |
| Evaluation/export depended on local notebook positions and paths | AST-based saved-model loading; scoped audit; verified portable evidence archive |

The [step-by-step guide](reproducibility.md) documents installation, data preparation, training, saved-checkpoint evaluation, aggregation, archive export and known limits. [environment_reference.json](environment_reference.json) records the tested CPU environment. [results.md](results.md) distinguishes current aggregates from historical snapshots and limits baseline/resource claims.

Validation includes a new Python 3.11 CPU environment, dependency and import checks, a real GATv2 forward pass, regression tests, source-data OCR and full saved-model retrieval. The full AI2D «Граф с общим отбором связей» seed-42 retrieval evaluation matched all three saved recalls exactly and passed its independent audit. Additional final checks are recorded in [reproduction_validation.json](reproduction_validation.json).

The missing document-only AI2D image 4325, incomplete historical per-run hyperparameters, full GPU retraining and external publication remain unresolved inputs/tasks. Existing scores are not relabeled as results of training on the corrected AI2D manifest.
