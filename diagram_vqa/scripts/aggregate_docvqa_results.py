from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("text_only", "coords_no_edges", "knn_k3", "knn_k5", "typed_spatial")
SEEDS = (42, 43, 44)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the controlled DocVQA graph experiment.")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs" / "docvqa_gnn")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "docvqa_gnn" / "aggregate_metrics.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "docvqa_results.md")
    parser.add_argument("--model-registry", type=Path, default=ROOT / "model_registry" / "docvqa_models.json")
    args = parser.parse_args()

    variants: list[dict[str, Any]] = []
    for variant in VARIANTS:
        runs = []
        for seed in SEEDS:
            path = args.runs_dir / variant / f"seed{seed}" / "metrics.json"
            if not path.exists():
                runs.append({"seed": seed, "status": "missing", "metrics_file": str(path.resolve())})
                continue
            metrics = read_json(path)
            runs.append(
                {
                    "seed": seed,
                    "status": metrics.get("status"),
                    "metrics_file": str(path.resolve()),
                    "checkpoint": metrics.get("checkpoint"),
                    "best_epoch": metrics.get("best_epoch"),
                    "val_mean_recall_at_k": metrics.get("best", {}).get("mean_recall_at_k", {}),
                    "val_q2d_mrr": metrics.get("best", {}).get("question_to_document", {}).get("mrr"),
                    "test_mean_recall_at_k": (metrics.get("test") or {}).get("mean_recall_at_k", {}),
                    "wall_time_seconds": metrics.get("wall_time_seconds"),
                    "peak_ram_mb": metrics.get("peak_ram_mb"),
                    "peak_vram_mb": metrics.get("peak_vram_mb"),
                    "parameter_count": metrics.get("parameter_count"),
                }
            )
        complete_runs = [run for run in runs if run.get("status") == "available_full"]
        val_r1 = [float(run["val_mean_recall_at_k"]["1"]) for run in complete_runs]
        test_r1 = [float(run["test_mean_recall_at_k"]["1"]) for run in complete_runs if run.get("test_mean_recall_at_k")]
        variants.append(
            {
                "variant": variant,
                "status": "available_full" if len(complete_runs) == len(SEEDS) else "incomplete",
                "runs": runs,
                "seed_count": len(complete_runs),
                "val_mean_r1_mean": mean(val_r1),
                "val_mean_r1_std": statistics.stdev(val_r1) if len(val_r1) > 1 else None,
                "test_mean_r1_mean": mean(test_r1),
                "test_mean_r1_std": statistics.stdev(test_r1) if len(test_r1) > 1 else None,
                "wall_time_seconds_mean": mean([float(run["wall_time_seconds"]) for run in complete_runs]),
                "peak_ram_mb_max": max((float(run["peak_ram_mb"]) for run in complete_runs), default=None),
                "peak_vram_mb_max": max((float(run["peak_vram_mb"]) for run in complete_runs), default=None),
                "parameter_count": complete_runs[0].get("parameter_count") if complete_runs else None,
            }
        )
    aggregate = {
        "dataset": "docvqa",
        "metric_family": "retrieval",
        "seeds": list(SEEDS),
        "status": "available_full" if all(row["status"] == "available_full" for row in variants) else "incomplete",
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")

    def optional_metrics(path: Path) -> dict[str, Any]:
        return read_json(path) if path.exists() else {}

    open_answer_rows = [
        ("Most frequent", optional_metrics(ROOT / "runs/docvqa_baselines/open_answer/most_frequent/val/metrics.json")),
        ("Random train answer", optional_metrics(ROOT / "runs/docvqa_baselines/open_answer/random_train_answer/val/metrics.json")),
        ("Question lexical retrieval", optional_metrics(ROOT / "runs/docvqa_baselines/open_answer/question_retrieval/val/metrics.json")),
        ("Qwen2.5-VL-3B", optional_metrics(ROOT / "runs/docvqa_qwen25_vl/val/metrics.json")),
    ]
    retrieval_rows = [
        ("Random", optional_metrics(ROOT / "runs/docvqa_baselines/retrieval/random/val/metrics.json"), optional_metrics(ROOT / "runs/docvqa_baselines/retrieval/random/test/metrics.json")),
        ("OCR text", optional_metrics(ROOT / "runs/docvqa_baselines/retrieval/ocr_text/val/metrics.json"), optional_metrics(ROOT / "runs/docvqa_baselines/retrieval/ocr_text/test/metrics.json")),
        ("CLIP ViT-B/32", optional_metrics(ROOT / "runs/docvqa_clip_retrieval/val/metrics.json"), optional_metrics(ROOT / "runs/docvqa_clip_retrieval/test/metrics.json")),
    ]

    lines = [
        "# DocVQA: воспроизводимые результаты",
        "",
        "ANLS открытых ответов и retrieval-метрики публикуются раздельно.",
        "Старый checkpoint `docvqa-auto` с Mean R@1=0.04156 считается legacy и не входит в таблицу.",
        "",
        "## Открытые ответы на official val",
        "",
        "| Модель | ANLS | Exact accuracy | Latency, ms/question | Статус |",
        "|---|---:|---:|---:|---|",
    ]
    for name, metrics in open_answer_rows:
        anls = "—" if metrics.get("anls") is None else f"{float(metrics['anls']):.4f}"
        accuracy = "—" if metrics.get("accuracy") is None else f"{float(metrics['accuracy']):.4f}"
        latency = "—" if metrics.get("latency_ms_per_question") is None else f"{float(metrics['latency_ms_per_question']):.2f}"
        lines.append(f"| {name} | {anls} | {accuracy} | {latency} | {metrics.get('status', 'missing')} |")
    lines += [
        "",
        "## Поиск документа",
        "",
        "| Модель | Val Mean R@1 | Test Mean R@1 | Val Mean R@5 | Статус |",
        "|---|---:|---:|---:|---|",
    ]
    for name, val, test in retrieval_rows:
        val_r1 = val.get("mean_recall_at_k", {}).get("1")
        val_r5 = val.get("mean_recall_at_k", {}).get("5")
        test_r1 = test.get("mean_recall_at_k", {}).get("1")
        lines.append(
            f"| {name} | {'—' if val_r1 is None else f'{float(val_r1):.4f}'} | "
            f"{'—' if test_r1 is None else f'{float(test_r1):.4f}'} | "
            f"{'—' if val_r5 is None else f'{float(val_r5):.4f}'} | {val.get('status', 'missing')} |"
        )
    lines += [
        "",
        "## OCR-графовые абляции",
        "",
        "| Вариант | Seeds | Val Mean R@1, mean±std | Test Mean R@1, mean±std | Статус |",
        "|---|---:|---:|---:|---|",
    ]
    for row in variants:
        val = "—" if row["val_mean_r1_mean"] is None else f"{row['val_mean_r1_mean']:.4f} ± {row['val_mean_r1_std'] or 0:.4f}"
        test = "—" if row["test_mean_r1_mean"] is None else f"{row['test_mean_r1_mean']:.4f} ± {row['test_mean_r1_std'] or 0:.4f}"
        lines.append(f"| {row['variant']} | {row['seed_count']} | {val} | {test} | {row['status']} |")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    model_registry = {
        "dataset": "docvqa",
        "generated_from": str(args.output.resolve()),
        "legacy": {
            "checkpoint": str((ROOT.parent / "runs/mlflow_checkpoints/docvqa-auto/gnn_best.pt").resolve()),
            "status": "legacy",
            "included_in_primary_table": False,
        },
        "qwen": {
            "model": str((ROOT.parent / "models/Qwen2.5-VL-3B-Instruct").resolve()),
            "metrics": str((ROOT / "runs/docvqa_qwen25_vl/val/metrics.json").resolve()),
            "status": open_answer_rows[-1][1].get("status", "missing"),
        },
        "graph_models": [run for variant in variants for run in variant["runs"]],
    }
    args.model_registry.parent.mkdir(parents=True, exist_ok=True)
    args.model_registry.write_text(json.dumps(model_registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
