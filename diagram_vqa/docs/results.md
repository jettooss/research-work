# Evaluation results and provenance

Current source: [versioned full-test evidence](../reports/reproducibility/evidence.json), updated 2026-09-08, with 36 graph/CLIP runs and three deterministic OCR baseline runs. The [research presentation](https://docs.google.com/presentation/d/1RnsHrDhAxOaTre7VCdBh67QbfDaOExJKicDK9aOw374/edit) supplies the current display names (revision `89PTcrzvEx09IQ`). The [original slide-text snapshot](presentation_snapshot_2026-09-06.json), revision `TCRNCD2aGY1zmQ`, is a historical snapshot and contains earlier incomplete aggregates. Section references below use the presentation's displayed slide labels (the title slide is unnumbered).

## AI2D answer accuracy — slide 10

Current local models: 3,088 test questions, seeds 42–44, mean ± population standard deviation (`ddof=0`). Percentages:

| Model | Accuracy, % | Runs |
|---|---:|---:|
| Qwen2.5-VL-3B + QLoRA | 73.19 (archived, not revalidated) | — |
| Граф с общим отбором связей | 48.54 ± 0.44 | 3 |
| Граф с отбором связей по типам | 47.83 ± 0.44 | 3 |
| GATv2 + kNN | 46.24 ± 0.46 | 3 |
| CLIP | 44.52 ± 0.86 | 3 |
| Random | 24.35 ± 0.21 | 3 |

The «Граф с общим отбором связей» model improves over CLIP by 4.02 percentage points using the displayed rounded means.

## Document retrieval — slides 11–12

`R@K = (question-to-document R@K + document-to-question R@K) / 2`.

The candidate pool contains all unique documents in the evaluation split. Reverse retrieval searches the associated questions; any associated question is a correct match. A hit requires at least one relevant item in the top K.

<!-- BEGIN GENERATED RETRIEVAL TABLE -->
| Dataset | Model | R@1, % | R@5, % | R@10, % | n |
|---|---|---:|---:|---:|---:|
| AI2D | CLIP | 4.72 ± 0.29 | 18.46 ± 0.55 | 29.20 ± 1.11 | 3 |
| AI2D | GATv2 + kNN | 6.64 ± 0.63 | 20.78 ± 0.45 | 31.82 ± 0.81 | 3 |
| AI2D | Граф с отбором связей по типам | 6.30 ± 0.42 | 20.85 ± 0.79 | 31.15 ± 1.35 | 3 |
| AI2D | Граф с общим отбором связей | 6.13 ± 0.42 | 20.28 ± 0.58 | 30.90 ± 0.39 | 3 |
| DocVQA | CLIP | 5.09 ± 0.22 | 14.82 ± 0.81 | 20.98 ± 0.73 | 3 |
| DocVQA | GATv2 + kNN | 18.51 ± 0.56 | 36.73 ± 0.30 | 45.80 ± 0.26 | 3 |
| DocVQA | Граф с отбором связей по типам | 7.62 ± 0.36 | 20.20 ± 0.35 | 29.52 ± 0.42 | 3 |
| DocVQA | Граф с общим отбором связей | 7.54 ± 0.43 | 20.79 ± 0.48 | 29.50 ± 0.51 | 3 |
| InfographicVQA | CLIP | 11.35 ± 0.08 | 25.76 ± 0.74 | 34.85 ± 0.87 | 3 |
| InfographicVQA | GATv2 + kNN | 24.58 ± 1.73 | 45.92 ± 1.36 | 56.27 ± 0.93 | 3 |
| InfographicVQA | Граф с отбором связей по типам | 13.85 ± 2.04 | 32.70 ± 2.77 | 43.53 ± 3.20 | 3 |
| InfographicVQA | Граф с общим отбором связей | 14.32 ± 2.16 | 32.14 ± 3.10 | 43.03 ± 3.61 | 3 |
<!-- END GENERATED RETRIEVAL TABLE -->

Mean ± population standard deviation across the three completed seeds 42–44. Historical manifest byte identity is not recorded by the original runs; current observed fingerprints are in the evidence manifest.

The displayed graph models improve over CLIP on the reported retrieval metrics. Architecture comparisons change features and training budgets as well as edges; they do not isolate the contribution of attention.

## Accuracy by question type — slide 13

| Type | Questions | CLIP | GATv2 + kNN | Граф с отбором связей по типам | Граф с общим отбором связей |
|---|---:|---:|---:|---:|---:|
| Counting | 122 | 42.9 | 50.8 | 50.0 | 52.5 |
| Causal | 230 | 45.8 | 52.3 | 51.7 | 53.2 |
| Spatial | 387 | 50.3 | 51.0 | 52.9 | 53.2 |
| Labels/OCR | 554 | 29.2 | 31.4 | 30.4 | 30.9 |

Values are the displayed mean accuracy percentages. The slide describes mean ± std across three runs, but does not provide numerical std values in its text; none are inferred here. Groups are assigned by keywords and cover 1,293 selected questions, not the entire test set. The «Граф с общим отбором связей» model gains 9.6 percentage points on counting and 7.4 on causal questions relative to CLIP. This is descriptive subgroup analysis.

## Pilot resource measurements — slide 15

AI2D, 32 questions, batch=1, RTX 4060 Laptop GPU.

| Model / mode | Peak VRAM, GiB | Time, ms/question |
|---|---:|---:|
| Qwen2.5-VL-3B + QLoRA | 2.695 | 992.86 |
| Граф с общим отбором связей (cached features) | 0.018 | 8.24 |
| Граф с отбором связей по типам (cached features) | 0.019 | 9.63 |
| GATv2 + kNN (cached features) | 0.014 | 8.67 |
| CLIP answer head (cached features) | 0.011 | 2.02 |
| Random (CPU) | 0 | < 0.001 |

Time is the mean after warmup. VRAM is peak PyTorch tensor allocation, excluding reserved memory and the CUDA context. Graph/CLIP measurements use prepared features, seed-42 weights, and three repeats; OCR, SAM2, DINOv2, and text/CLIP encoders are excluded. Qwen uses NF4, at most 32 generated tokens, max_pixels=262144, and one repeat; its visual encoder is included. GPU use was shared with other processes.

Compute boundaries differ, so this does not measure end-to-end speedup or training memory requirements. The three visually checked correspondences on slide 14 are a selected best-case example, not a dataset-wide edge-quality metric.

## Evidence

- [Generated verification snapshot](readme_metrics_verification.json): per-seed Accuracy and all retrieval aggregates. Regenerate with `python diagram_vqa/scripts/reproduce_results.py --write`; verify both Markdown tables with `--check`.
- [Retrieval audit](../reports/presentation_metrics/retrieval_matrix_audit.json): protocol `bidirectional_mean_unique_documents_v1`, 36 verified runs out of 36, seeds, historical result paths, and weight hashes. Local checkpoint/history validation was repeated successfully for all 36 runs on 2026-09-08.
- [Portable evaluation evidence](../reports/reproducibility/evidence.json): relative paths, directional retrieval measurements and file checksums. All 18 attention retrieval reports needed to check table aggregation are now versioned outside ignored run directories.
- Accuracy: `runs/ai2d/<architecture>/seed{42,43,44}_full/metrics.json`, field `test.accuracy` or `test.vqa.score`. Architectures are `hybrid_v4_multipos_training`, `sam2_dinov2_heterogeneous_balanced_evidence_graph`, and `sam2_dinov2_dual_branch_evidence_graph`. CLIP/Random use `runs/model_matrix/ai2d/{clip,random}/seed{42,43,44}/test/metrics.json`, field `vqa.score`.
- Resource measurements were checked against `resource-slide/{qwen,dual,attention,knn,clip,random}.json` in the workspace root. These pilot measurements are separate from the generated Accuracy/retrieval verification snapshot.

## Source discrepancies and limitations

1. The old snapshot contains InfographicVQA attention R@1=11.01% in its conclusion and an n=2 table. The current result is **13.85 ± 2.04%, n=3**; the current live presentation has also been corrected. Historical snapshots are retained as provenance, not current tables.
2. `reports/presentation_metrics/tables_20260906.json` is an older snapshot: for example, AI2D GATv2 + kNN Accuracy 46.57%, n=2. Three completed runs give **46.24 ± 0.46%**, matching the current slide table. The earlier file is not treated as the current table.
3. All 36 principal runs are complete with three distinct weight hashes per architecture/dataset. Three training seeds and population SD do not by themselves establish statistical significance. The OCR comparison uses a deterministic full evaluation rather than fabricated repeated seeds.
4. Qwen's archived 73.19% has not been confirmed by a new full evaluation. Small-sample evaluations or bootstrap resampling of fixed predictions do not constitute independent training runs.
5. DocVQA/InfographicVQA retrieval values are not answer-quality ANLS scores. The model matrix stores answer quality and retrieval separately.
6. New results require rechecking split, protocol, sample count, and completed seeds before updating the published table. Full model training and a complete Qwen reevaluation were outside this documentation update.
