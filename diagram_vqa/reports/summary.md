# Experiment Summary

Historical snapshot (May 2026). The acceptance threshold below belongs to that development stage; it is not the current scientific comparison or a rule for selecting results on test. Current verified three-seed results and limitations are in [results.md](../docs/results.md).

This folder stores compact reports only. Large checkpoints, model weights, datasets, caches, and full prediction dumps stay outside this clean project.

## Included reports

- `baselines/ai2d_baselines_q_only_test_metrics.json`
- `hybrid/ai2d_hybrid_v4_multipos_metrics.json`
- `hybrid/ai2d_hybrid_v5_vqa06_lr1e4_metrics.json`
- `hybrid/ai2d_hybrid_manual_full_20260510_metrics.json`
- `public_vqa/comparison_manual_20260510.md`
- `public_vqa/ai2d_hybrid_manual_full_20260510_metrics.json`
- `public_vqa/ai2d_hybrid_manual_full_20260510_error_slices.json`

## Acceptance threshold

The recorded best local AI2D test accuracy at that stage was `0.4035`, with a proposed development target of `0.4535`. Select models on validation; reserve test for final reporting.
