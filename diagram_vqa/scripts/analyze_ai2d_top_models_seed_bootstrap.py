from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MODELS = {
    "CLIP baseline": Path("runs/ai2d/clip_vqa_baseline/seed42_full/test/predictions.jsonl"),
    "GraphColBERT+FiLM": Path("runs/ai2d/graphcolbert_film/seed42_full/test/predictions.jsonl"),
    "Граф с общим отбором связей": Path(
        "runs/ai2d_top_models_100_seed_analysis/dual_branch/predictions.jsonl"
    ),
    "Qwen2.5-VL QLoRA": Path(
        "runs/ai2d_top_models_100_seed_analysis/qwen25_vl_qlora/"
        "predictions.jsonl"
    ),
}


def _load_predictions(path: Path) -> tuple[list[str], dict[str, float]]:
    order: list[str] = []
    scores: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        if "is_correct" in row:
            score = float(bool(row["is_correct"]))
        elif "exact" in row:
            score = float(row["exact"])
        else:
            score = float(row["score"])
        if sample_id not in scores:
            order.append(sample_id)
        scores[sample_id] = score
    return order, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/ai2d_top_models_100_seed_analysis"),
    )
    args = parser.parse_args()

    loaded = {name: _load_predictions(path) for name, path in DEFAULT_MODELS.items()}
    reference_order = loaded["Qwen2.5-VL QLoRA"][0]
    common_ids = [
        sample_id
        for sample_id in reference_order
        if all(sample_id in scores for _, scores in loaded.values())
    ][: args.samples]
    if len(common_ids) != args.samples:
        counts = {name: len(scores) for name, (_, scores) in loaded.items()}
        raise RuntimeError(
            f"Expected {args.samples} common samples, found {len(common_ids)}; "
            f"prediction counts={counts}"
        )

    names = list(loaded)
    score_matrix = np.asarray(
        [[loaded[name][1][sample_id] for sample_id in common_ids] for name in names],
        dtype=float,
    )
    observed = score_matrix.mean(axis=1)

    iteration_rows: list[dict[str, object]] = []
    bootstrap_scores = np.empty((args.iterations, len(names)), dtype=float)
    for iteration in range(args.iterations):
        seed = args.seed_start + iteration
        indices = np.random.default_rng(seed).integers(0, len(common_ids), len(common_ids))
        values = score_matrix[:, indices].mean(axis=1)
        bootstrap_scores[iteration] = values
        best = values.max()
        winners = np.flatnonzero(np.isclose(values, best))
        ranks = np.asarray([1 + int(np.sum(values > value)) for value in values])
        for model_index, name in enumerate(names):
            iteration_rows.append(
                {
                    "iteration": iteration + 1,
                    "seed": seed,
                    "model": name,
                    "accuracy": values[model_index],
                    "rank": int(ranks[model_index]),
                    "is_winner": int(model_index in winners),
                    "winner_credit": (
                        1.0 / len(winners) if model_index in winners else 0.0
                    ),
                }
            )

    clip_index = names.index("CLIP baseline")
    summary_rows: list[dict[str, object]] = []
    for model_index, name in enumerate(names):
        values = bootstrap_scores[:, model_index]
        delta = values - bootstrap_scores[:, clip_index]
        ranks = [row["rank"] for row in iteration_rows if row["model"] == name]
        win_credit = [
            row["winner_credit"] for row in iteration_rows if row["model"] == name
        ]
        summary_rows.append(
            {
                "model": name,
                "samples": len(common_ids),
                "iterations": args.iterations,
                "observed_accuracy": observed[model_index],
                "bootstrap_mean": values.mean(),
                "bootstrap_std": values.std(ddof=1),
                "ci95_low": np.percentile(values, 2.5),
                "ci95_high": np.percentile(values, 97.5),
                "min_accuracy": values.min(),
                "max_accuracy": values.max(),
                "first_place_rate": np.mean(win_credit),
                "average_rank": np.mean(ranks),
                "mean_delta_vs_clip": delta.mean(),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["bootstrap_mean", "observed_accuracy"], ascending=False
    )
    iterations = pd.DataFrame(iteration_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    iterations.to_csv(
        args.output_dir / "bootstrap_iterations.csv", index=False, encoding="utf-8-sig"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "sample_ids.json").write_text(
        json.dumps(common_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.boxplot(
        [bootstrap_scores[:, names.index(name)] * 100 for name in summary["model"]],
        tick_labels=summary["model"],
        showmeans=True,
    )
    axis.set_ylabel("Accuracy, %")
    axis.set_title(f"AI2D: {args.iterations} paired bootstrap seed iterations")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=18)
    figure.tight_layout()
    figure.savefig(args.output_dir / "bootstrap_accuracy.png", dpi=180)
    plt.close(figure)

    report_lines = [
        "# AI2D: анализ устойчивости лучших моделей",
        "",
        f"Одинаковые {len(common_ids)} тестовых вопросов; {args.iterations} парных bootstrap-итераций.",
        "Это анализ устойчивости по seed-выборкам, а не 100 повторных обучений каждой модели.",
        "",
        "Qwen запущен через совместимую копию LoRA-checkpoint: старые пути слоёв "
        "переназначены под текущую версию Transformers, исходный adapter не изменён.",
        "",
        "| Модель | Accuracy | Среднее | Std | 95% интервал | Победы |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        report_lines.append(
            "| {model} | {observed_accuracy:.1%} | {bootstrap_mean:.1%} | "
            "{bootstrap_std:.1%} | {ci95_low:.1%}–{ci95_high:.1%} | "
            "{first_place_rate:.0%} |".format(**row)
        )
    (args.output_dir / "REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
