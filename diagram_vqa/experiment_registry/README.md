# Experiment Registry

This registry includes more than AI2D hybrid v4/v5: graph checkpoints, baselines, base VLM, QLoRA adapter, SFT data, VLM eval, public VQA reports, and multi-dataset summaries.

| Kind | Name | Role | Score / acc | Path |
|---|---|---|---:|---|
| graph_checkpoint | ai2d_hybrid | trained GraphEncoder/TextProj checkpoint | 0.2464 | `model_registry/checkpoints/ai2d_hybrid/checkpoint_best.pt` |
| graph_checkpoint | ai2d_hybrid_manual_full_20260510 | trained GraphEncoder/TextProj checkpoint | 0.4035 | `model_registry/checkpoints/ai2d_hybrid_manual_full_20260510/checkpoint_best.pt` |
| graph_checkpoint | ai2d_hybrid_manual_full_clean | trained GraphEncoder/TextProj checkpoint | 0.3912 | `model_registry/checkpoints/ai2d_hybrid_manual_full_clean/checkpoint_best.pt` |
| graph_checkpoint | ai2d_hybrid_manual_smoke_20260510 | trained GraphEncoder/TextProj checkpoint | 0.5000 | `model_registry/checkpoints/ai2d_hybrid_manual_smoke_20260510/checkpoint_best.pt` |
| graph_checkpoint | ai2d_hybrid_v3 | trained GraphEncoder/TextProj checkpoint | 0.4048 | `model_registry/checkpoints/ai2d_hybrid_v3/checkpoint_best.pt` |
| graph_checkpoint | ai2d_hybrid_v4_multipos | trained GraphEncoder/TextProj checkpoint | 0.4048 | `model_registry/checkpoints/ai2d_hybrid_v4_multipos/checkpoint_best.pt` |
| graph_checkpoint | ai2d_hybrid_v5_vqa06_lr1e4 | trained GraphEncoder/TextProj checkpoint | 0.3993 | `model_registry/checkpoints/ai2d_hybrid_v5_vqa06_lr1e4/checkpoint_best.pt` |
| base_vlm_external | Qwen2.5-VL-3B-Instruct | local base vision-language model used for direct/rerank experiments | - | `../models/Qwen2.5-VL-3B-Instruct` |
| vlm_adapter | vlm_qlora_smoke3_adapter | QLoRA adapter checkpoint for VLM SFT smoke experiment | - | `experiment_registry/vlm_adapters/vlm_qlora_smoke3_adapter/adapter_model.safetensors` |
| baseline_metrics | ai2d_baselines_q_only_test_metrics | baseline metrics | - | `experiment_registry/baselines/ai2d_baselines_q_only_test_metrics.json` |
| baseline_metrics | ai2d_baselines_test_metrics | baseline metrics | - | `experiment_registry/baselines/ai2d_baselines_test_metrics.json` |
| public_vqa_report | ai2d_hybrid_manual_full_20260510_error_slices | public VQA metrics/comparison/error report | - | `experiment_registry/public_vqa/ai2d_hybrid_manual_full_20260510_error_slices.json` |
| public_vqa_report | ai2d_hybrid_manual_full_20260510_metrics | public VQA metrics/comparison/error report | - | `experiment_registry/public_vqa/ai2d_hybrid_manual_full_20260510_metrics.json` |
| public_vqa_report | ai2d_hybrid_v5_error_slices | public VQA metrics/comparison/error report | - | `experiment_registry/public_vqa/ai2d_hybrid_v5_error_slices.json` |
| public_vqa_report | ai2d_hybrid_v5_metrics | public VQA metrics/comparison/error report | - | `experiment_registry/public_vqa/ai2d_hybrid_v5_metrics.json` |
| public_vqa_report | comparison | public VQA metrics/comparison/error report | - | `experiment_registry/public_vqa/comparison.json` |
| vlm_eval_report | vlm_sft_eval_smoke_512_error_slices | VLM SFT evaluation report | - | `experiment_registry/vlm_eval/vlm_sft_eval_smoke_512_error_slices.json` |
| vlm_eval_report | vlm_sft_eval_smoke_512_metrics | VLM SFT evaluation report | - | `experiment_registry/vlm_eval/vlm_sft_eval_smoke_512_metrics.json` |
| vlm_eval_report | vlm_sft_eval_smoke_error_slices | VLM SFT evaluation report | - | `experiment_registry/vlm_eval/vlm_sft_eval_smoke_error_slices.json` |
| vlm_eval_report | vlm_sft_eval_smoke_metrics | VLM SFT evaluation report | - | `experiment_registry/vlm_eval/vlm_sft_eval_smoke_metrics.json` |
| sft_dataset_summary | vlm_sft_data_ai2d_summary | SFT dataset generation summary | - | `experiment_registry/sft_data/vlm_sft_data_ai2d_summary.json` |
| sft_dataset_summary | vlm_sft_data_smoke_summary | SFT dataset generation summary | - | `experiment_registry/sft_data/vlm_sft_data_smoke_summary.json` |
| sft_dataset_summary | vlm_sft_data_summary | SFT dataset generation summary | - | `experiment_registry/sft_data/vlm_sft_data_summary.json` |
| multi_dataset_summary | ai2d_tesseract_summary | MLflow/multi-dataset checkpoint summary | - | `experiment_registry/mlflow_summaries/ai2d_tesseract_summary.json` |
| multi_dataset_summary | docvqa_auto_summary | MLflow/multi-dataset checkpoint summary | - | `experiment_registry/mlflow_summaries/docvqa_auto_summary.json` |
| graph_reeval_report | ai2d_hybrid_v3_reeval_multipos_metrics | graph model reevaluation report | - | `experiment_registry/graph_reeval/ai2d_hybrid_v3_reeval_multipos_metrics.json` |
