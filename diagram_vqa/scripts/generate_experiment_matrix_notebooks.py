from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vqa_retrieval.experiment_matrix import ARCHITECTURES, ARCHITECTURE_SPECS, DATASETS


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks" / "experiments"


def code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def notebook(dataset: str, architecture: str) -> dict[str, Any]:
    spec = ARCHITECTURE_SPECS[architecture]
    title = f"{dataset.upper()} - {spec.title}"
    cells = [
        markdown_cell(
            f"# {title}\n\n"
            "One notebook = one dataset + one architecture. The cells below run the full "
            "experiment path end to end: data preview, parsing, architecture, training, "
            "validation, prediction, metrics, and overfitting check."
        ),
        code_cell(
            "from pathlib import Path\n\n"
            "from vqa_retrieval.experiment_matrix import (\n"
            "    ExperimentConfig,\n"
            "    build_metrics_table,\n"
            "    check_overfitting,\n"
            "    describe_dataset_format,\n"
            "    describe_model_architecture,\n"
            "    evaluate_experiment,\n"
            "    load_dataset_sample,\n"
            "    plot_training_curves,\n"
            "    predict_sample,\n"
            "    prepare_experiment_data,\n"
            "    show_parsed_sample,\n"
            "    show_sample_image,\n"
            "    train_experiment,\n"
            ")\n"
        ),
        code_cell(
            f"DATASET = {dataset!r}\n"
            f"ARCHITECTURE = {architecture!r}\n"
            "SEED = 42\n"
            "EPOCHS = 40\n"
            "\n"
            "\n"
            "def make_config() -> ExperimentConfig:\n"
            "    return ExperimentConfig(\n"
            "        dataset=DATASET,\n"
            "        architecture=ARCHITECTURE,\n"
            "        seed=SEED,\n"
            "        epochs=EPOCHS,\n"
            "        force_retrain=False,\n"
            "        reuse_existing=True,\n"
            "    )\n"
            "\n"
            "\n"
            "config = make_config()\n"
            "config.as_dict()\n"
        ),
        markdown_cell("## Dataset Format"),
        code_cell(
            "def load_data_preview(split: str = 'train'):\n"
            "    return describe_dataset_format(config, split=split, limit=3)\n"
            "\n"
            "\n"
            "format_preview = load_data_preview('train')\n"
            "format_preview\n"
        ),
        markdown_cell("## Data Example"),
        code_cell(
            "def load_example(split: str = 'train', index: int = 0):\n"
            "    return load_dataset_sample(config, split=split, index=index)\n"
            "\n"
            "\n"
            "sample = load_example('train', 0)\n"
            "sample\n"
        ),
        code_cell(
            "def show_example_image(sample_row):\n"
            "    return show_sample_image(sample_row)\n"
            "\n"
            "\n"
            "image = show_example_image(sample)\n"
            "image\n"
        ),
        markdown_cell("## Parsed Image"),
        code_cell(
            "def prepare_data_for_approach():\n"
            "    return prepare_experiment_data(config)\n"
            "\n"
            "\n"
            "prepared = prepare_data_for_approach()\n"
            "prepared['summary']\n"
        ),
        code_cell(
            "def visualize_parsing(split: str = 'train', index: int = 0):\n"
            "    return show_parsed_sample(config, split=split, index=index, limit=24)\n"
            "\n"
            "\n"
            "parsed = visualize_parsing('train', 0)\n"
            "parsed\n"
        ),
        markdown_cell("## Architecture"),
        code_cell(
            "def define_architecture():\n"
            "    return describe_model_architecture(config)\n"
            "\n"
            "\n"
            "architecture = define_architecture()\n"
            "architecture\n"
        ),
        markdown_cell("## Training And Validation"),
        code_cell(
            "def train_model_with_validation():\n"
            "    return train_experiment(config, show_progress=True)\n"
            "\n"
            "\n"
            "metrics = train_model_with_validation()\n"
            "metrics\n"
        ),
        markdown_cell("## Prediction"),
        code_cell(
            "def run_prediction(split: str = 'val', index: int = 0):\n"
            "    return predict_sample(config, metrics, split=split, index=index)\n"
            "\n"
            "\n"
            "prediction = run_prediction('val', 0)\n"
            "prediction\n"
        ),
        markdown_cell("## Metrics"),
        code_cell(
            "def validate_model():\n"
            "    return evaluate_experiment(config, metrics)\n"
            "\n"
            "\n"
            "evaluation = validate_model()\n"
            "evaluation\n"
        ),
        code_cell(
            "test_metrics = metrics['test']\n"
            "test_metrics\n"
        ),
        code_cell(
            "def summarize_overfitting():\n"
            "    return check_overfitting(metrics)\n"
            "\n"
            "\n"
            "overfitting = summarize_overfitting()\n"
            "overfitting\n"
        ),
        code_cell(
            "def metrics_table():\n"
            "    return build_metrics_table([metrics])\n"
            "\n"
            "\n"
            "table = metrics_table()\n"
            "table\n"
        ),
        code_cell(
            "def plot_validation_curves():\n"
            "    return plot_training_curves(metrics)\n"
            "\n"
            "\n"
            "plot_validation_curves()\n"
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (data-cu124)",
                "language": "python",
                "name": "data-cu124",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
            "experiment_matrix": {
                "dataset": dataset,
                "architecture": architecture,
                "backend_model": spec.backend_model,
                "seed": 42,
                "epochs": 40,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebooks() -> list[Path]:
    written: list[Path] = []
    for dataset in DATASETS:
        dataset_dir = NOTEBOOK_ROOT / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for architecture in ARCHITECTURES:
            path = dataset_dir / f"{dataset}_{architecture}.ipynb"
            path.write_text(
                json.dumps(notebook(dataset, architecture), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            written.append(path)
    return written


def main() -> None:
    written = write_notebooks()
    print(json.dumps({"written": [path.as_posix() for path in written], "count": len(written)}, indent=2))


if __name__ == "__main__":
    main()
