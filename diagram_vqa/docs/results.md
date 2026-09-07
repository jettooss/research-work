# Evaluation results and provenance

Source: [research presentation](https://docs.google.com/presentation/d/1a3m6Goh3P5MZFHDC1UjNX7AyD_vt7G5VJsvcX-Syvyc/edit), revision `TCRNCD2aGY1zmQ`. The [original slide-text snapshot](presentation_snapshot_2026-09-06.json) preserves the source language, values, and caveats. Snapshot date: 2026-09-06. Slide numbers below include the title slide.

## AI2D answer accuracy — slide 10

Current local models: 3,088 test questions, seeds 42–44, mean ± population standard deviation (`ddof=0`). Percentages:

| Model | Accuracy, % | Runs |
|---|---:|---:|
| Qwen2.5-VL-3B + QLoRA | 73.19 (archived, not revalidated) | — |
| Dual-Branch + top-k Attention | 48.54 ± 0.44 | 3 |
| Heterogeneous Graph + Attention | 47.83 ± 0.44 | 3 |
| Hybrid GATv2 + kNN | 46.24 ± 0.46 | 3 |
| CLIP | 44.52 ± 0.86 | 3 |
| Random | 24.35 ± 0.21 | 3 |

Dual-Branch improves over CLIP by 4.02 percentage points using the displayed rounded means.

## Document retrieval — slides 11–12

`R@K = (question-to-document R@K + document-to-question R@K) / 2`.

The candidate pool contains all unique documents in the evaluation split. Reverse retrieval searches the associated questions; any associated question is a correct match. A hit requires at least one relevant item in the top K.

| Dataset | Model | R@1, % | R@5, % | R@10, % | n |
|---|---|---:|---:|---:|---:|
| DocVQA | CLIP | 5.09 ± 0.22 | 14.82 ± 0.81 | 20.98 ± 0.73 | 3 |
| DocVQA | Hybrid GATv2 + kNN | 17.76 | 36.77 | 45.47 | 1 |
| DocVQA | Heterogeneous Graph + Attention | 7.62 ± 0.36 | 20.20 ± 0.35 | 29.52 ± 0.42 | 3 |
| DocVQA | Dual-Branch + top-k Attention | 7.54 ± 0.43 | 20.79 ± 0.48 | 29.50 ± 0.51 | 3 |
| InfographicVQA | CLIP | 11.35 ± 0.08 | 25.76 ± 0.74 | 34.85 ± 0.87 | 3 |
| InfographicVQA | Hybrid GATv2 + kNN | 26.94 | 47.75 | 57.52 | 1 |
| InfographicVQA | Heterogeneous Graph + Attention | 12.92 ± 1.91 | 31.27 ± 2.32 | 41.98 ± 2.85 | 2 |
| InfographicVQA | Dual-Branch + top-k Attention | 14.32 ± 2.16 | 32.14 ± 3.10 | 43.03 ± 3.61 | 3 |
| AI2D | CLIP | 4.72 ± 0.29 | 18.46 ± 0.55 | 29.20 ± 1.11 | 3 |
| AI2D | Hybrid GATv2 + kNN | 6.64 ± 0.63 | 20.78 ± 0.45 | 31.82 ± 0.81 | 3 |
| AI2D | Heterogeneous Graph + Attention | 6.30 ± 0.42 | 20.85 ± 0.79 | 31.15 ± 1.35 | 3 |
| AI2D | Dual-Branch + top-k Attention | 6.13 ± 0.42 | 20.28 ± 0.58 | 30.90 ± 0.39 | 3 |

Mean ± population standard deviation across completed runs. `n` counts actual completed seeds from 42–44; spread is not estimated for n=1. Missing runs are excluded.

## Accuracy by question type — slide 13

| Type | Questions | CLIP | Hybrid GATv2 + kNN | Heterogeneous Graph + Attention | Dual-Branch + top-k Attention |
|---|---:|---:|---:|---:|---:|
| Counting | 122 | 42.9 | 50.8 | 50.0 | 52.5 |
| Causal | 230 | 45.8 | 52.3 | 51.7 | 53.2 |
| Spatial | 387 | 50.3 | 51.0 | 52.9 | 53.2 |
| Labels/OCR | 554 | 29.2 | 31.4 | 30.4 | 30.9 |

Values are the displayed mean accuracy percentages. The slide describes mean ± std across three runs, but does not provide numerical std values in its text; none are inferred here. Groups are assigned by keywords and cover 1,293 selected questions, not the entire test set. Dual-Branch gains 9.6 percentage points on counting and 7.4 on causal questions relative to CLIP. This is descriptive subgroup analysis.

## Pilot resource measurements — slide 15

AI2D, 32 questions, batch=1, RTX 4060 Laptop GPU.

| Model / mode | Peak VRAM, GiB | Time, ms/question |
|---|---:|---:|
| Qwen2.5-VL-3B + QLoRA | 2.695 | 992.86 |
| Dual-Branch (cached features) | 0.018 | 8.24 |
| Graph attention (cached features) | 0.019 | 9.63 |
| Graph kNN (cached features) | 0.014 | 8.67 |
| CLIP answer head (cached features) | 0.011 | 2.02 |
| Random (CPU) | 0 | < 0.001 |

Time is the mean after warmup. VRAM is peak PyTorch tensor allocation, excluding reserved memory and the CUDA context. Graph/CLIP measurements use prepared features, seed-42 weights, and three repeats; OCR, SAM2, DINOv2, and text/CLIP encoders are excluded. Qwen uses NF4, at most 32 generated tokens, max_pixels=262144, and one repeat; its visual encoder is included. GPU use was shared with other processes.

Compute boundaries differ, so this does not measure end-to-end speedup or training memory requirements. The three visually checked correspondences on slide 14 are a selected best-case example, not a dataset-wide edge-quality metric.

## Evidence

- [Verification snapshot](readme_metrics_verification.json): per-seed Accuracy, all retrieval aggregates, resource values, and local source paths.
- [Retrieval audit](../reports/presentation_metrics/retrieval_matrix_audit.json): protocol `bidirectional_mean_unique_documents_v1`, 31 verified runs out of 36 in the cited snapshot, seeds, result paths, and weight hashes. All 36 means and 30 available standard deviations match the presentation after rounding.
- Accuracy: `runs/ai2d/<architecture>/seed{42,43,44}_full/metrics.json`, field `test.accuracy` or `test.vqa.score`. Architectures are `hybrid_v4_multipos_training`, `sam2_dinov2_heterogeneous_balanced_evidence_graph`, and `sam2_dinov2_dual_branch_evidence_graph`. CLIP/Random use `runs/model_matrix/ai2d/{clip,random}/seed{42,43,44}/test/metrics.json`, field `vqa.score`.
- Resource measurements were checked against `resource-slide/{qwen,dual,attention,knn,clip,random}.json` in the workspace root; the compact verification snapshot retains the numeric values.

## Source discrepancies and limitations

1. The conclusion on slide 16 still quotes InfographicVQA attention R@1=11.01%. The table and audit give **12.92 ± 1.91%, n=2**. The README follows the table. The presentation itself was not edited as part of the README work.
2. `reports/presentation_metrics/tables_20260906.json` is an older snapshot: for example, AI2D Hybrid Accuracy 46.57%, n=2. Three completed runs give **46.24 ± 0.46%**, matching the current slide table. The earlier file is not treated as the current table.
3. Hybrid results for DocVQA/InfographicVQA have one completed run. Larger means do not demonstrate robustness or statistical significance. Missing seeds are not replaced with duplicates.
4. Qwen's archived 73.19% has not been confirmed by a new full evaluation. Small-sample evaluations or bootstrap resampling of fixed predictions do not constitute independent training runs.
5. DocVQA/InfographicVQA retrieval values are not answer-quality ANLS scores. The model matrix stores answer quality and retrieval separately.
6. New results require rechecking split, protocol, sample count, and completed seeds before updating the published table. Full model training and a complete Qwen reevaluation were outside this documentation update.
