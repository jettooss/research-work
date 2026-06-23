# AI2D notebook metrics recomputed

Created: 2026-06-23T09:13:42

## Recomputed from prediction rows

| Run | Correct/Total | Accuracy | Stored | Match |
|---|---:|---:|---:|:---:|
| `diagram_vqa\runs\ai2d_graph_transformer_full` | 1224/3088 | 0.3964 | 0.39637306332588196 | True |
| `diagram_vqa\runs\ai2d_graph_transformer_smoke` | 3/4 | 0.75 | 0.75 | True |
| `diagram_vqa\runs\ai2d_hybrid_manual_full` | 1208/3088 | 0.3912 | 0.3911917209625244 | True |
| `diagram_vqa\runs\ai2d_hybrid_smoke` | 2/4 | 0.5 | 0.5 | True |
| `diagram_vqa\runs\ai2d_sam2_graph_transformer_full` | 761/3088 | 0.2464 | 0.24643781781196594 | True |
| `diagram_vqa\runs\progress_sam_smoke` | 1/1 | 1.0 | 1.0 | True |
| `diagram_vqa\runs\smoke_graph_transformer_train_cuda` | 1/1 | 1.0 | 1.0 | True |
| `diagram_vqa\runs\smoke_late_graph_transformer_train_cuda` | 1/1 | 1.0 | 1.0 | True |
| `diagram_vqa\runs\smoke_sam_graph_transformer_train_cuda` | 1/1 | 1.0 | 1.0 | True |
| `runs\ai2d_hybrid` | 761/3088 | 0.2464 | 0.24643781781196594 | True |
| `runs\ai2d_hybrid_manual_full_20260510` | 1246/3088 | 0.4035 | 0.4034973978996277 | True |
| `runs\ai2d_hybrid_manual_smoke_20260510` | 2/4 | 0.5 | 0.5 | True |
| `runs\ai2d_hybrid_v3` | 1250/3088 | 0.4048 | 0.40479275584220886 | True |
| `runs\ai2d_hybrid_v3_reeval_multipos` | 1250/3088 | 0.4048 | None | None |
| `runs\ai2d_hybrid_v4_multipos` | 1250/3088 | 0.4048 | 0.40479275584220886 | True |
| `runs\ai2d_hybrid_v5_vqa06_lr1e4` | 1233/3088 | 0.3993 | 0.39928755164146423 | True |

## Standalone metrics files

```json
[
  {
    "file": "diagram_vqa\\runs\\ai2d_graphcolbert_film\\metrics_all.json",
    "n_test": 3088,
    "split": "test",
    "graph_only": 0.3119,
    "plain_colbert": 0.2895
  },
  {
    "file": "diagram_vqa\\runs\\ai2d_llm_reasoner\\metrics_all.json",
    "n_eval": 3088,
    "split": "test",
    "per_model": {
      "openai/gpt-oss-20b": {
        "accuracy": 0.6153,
        "parsed_ok": 2446,
        "n_eval": 3088,
        "seconds": 6320.6
      },
      "deepseek/deepseek-v4-pro": {
        "accuracy": 0.1852,
        "parsed_ok": 615,
        "n_eval": 3088,
        "seconds": 1030.6
      }
    }
  },
  {
    "file": "diagram_vqa\\runs\\vlm_qlora_mcq\\metrics.json",
    "model": "Qwen2.5-VL-3B QLoRA MCQ"
  },
  {
    "file": "runs\\ai2d_baselines\\test_metrics.json",
    "vqa_accuracy": {
      "random": 0.2548575129533679,
      "majority_correct_index": 0.25971502590673573,
      "ocr_option_overlap": 0.28950777202072536
    },
    "split": "test"
  },
  {
    "file": "runs\\ai2d_baselines_q_only\\test_metrics.json",
    "vqa_accuracy": {
      "random": 0.2548575129533679,
      "majority_correct_index": 0.25971502590673573,
      "ocr_option_overlap": 0.28950777202072536
    },
    "split": "test"
  },
  {
    "file": "diagram_vqa\\reports\\hybrid\\ai2d_hybrid_v4_multipos_metrics.json",
    "test": {
      "split": "test",
      "i2t": {
        "1": 0.014572538435459137,
        "5": 0.10330311208963394,
        "10": 0.16483160853385925
      },
      "t2i": {
        "1": 0.014572538435459137,
        "5": 0.028821242973208427,
        "10": 0.04922279715538025
      },
      "mean": {
        "1": 0.014572538435459137,
        "5": 0.06606217753142118,
        "10": 0.10702720284461975
      },
      "vqa_acc": 0.40479275584220886,
      "composite": 0.2559099793434143
    }
  },
  {
    "file": "diagram_vqa\\reports\\hybrid\\ai2d_hybrid_v5_vqa06_lr1e4_metrics.json",
    "test": {
      "split": "test",
      "i2t": {
        "1": 0.019430052489042282,
        "5": 0.09196890890598297,
        "10": 0.1596502661705017
      },
      "t2i": {
        "1": 0.02234455943107605,
        "5": 0.03950777277350426,
        "10": 0.06088082864880562
      },
      "mean": {
        "1": 0.020887305960059166,
        "5": 0.06573834083974361,
        "10": 0.11026554740965366
      },
      "vqa_acc": 0.39928755164146423,
      "composite": 0.25477654952555895
    }
  },
  {
    "file": "diagram_vqa\\reports\\hybrid\\ai2d_hybrid_manual_full_20260510_metrics.json",
    "test": {
      "split": "test",
      "i2t": {
        "1": 0.025259068235754967,
        "5": 0.09164507687091827,
        "10": 0.17389896512031555
      },
      "t2i": {
        "1": 0.018458548933267593,
        "5": 0.03465025871992111,
        "10": 0.05699481815099716
      },
      "mean": {
        "1": 0.02185880858451128,
        "5": 0.06314766779541969,
        "10": 0.11544689163565636
      },
      "vqa_acc": 0.4034973978996277,
      "composite": 0.259472144767642
    }
  }
]
```