# Experiment Summary

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

The current best local AI2D test accuracy is `0.4035`. Treat a new experiment as meaningful only if it reaches at least `0.4535` on the same test split.
