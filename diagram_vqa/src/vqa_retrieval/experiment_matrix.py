from __future__ import annotations

import json
import gc
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from PIL import Image, ImageDraw

from .model_matrix import load_rows, ocr_spans, split_rows


DATASETS = ("ai2d", "docvqa", "infographicvqa")
ARCHITECTURES = (
    "clip_vqa_baseline",
    "gnn_ocr_knn_graph_encoder",
    "hybrid_v4_multipos_training",
    "llm_reasoner",
    "graphcolbert_film",
    "hybrid_epoch_view_training",
    "qwen_vlm_qlora_training",
    "sam_sam2_graph_nodes",
    "train_sam_graph_transformer_models",
)


@dataclass(frozen=True)
class ArchitectureSpec:
    slug: str
    title: str
    backend_model: str
    processing_kind: str
    trainable: bool = True


ARCHITECTURE_SPECS: dict[str, ArchitectureSpec] = {
    "clip_vqa_baseline": ArchitectureSpec(
        slug="clip_vqa_baseline",
        title="CLIP VQA baseline",
        backend_model="clip",
        processing_kind="vision_text_retrieval",
    ),
    "gnn_ocr_knn_graph_encoder": ArchitectureSpec(
        slug="gnn_ocr_knn_graph_encoder",
        title="OCR KNN graph encoder",
        backend_model="gatv2_knn",
        processing_kind="ocr_knn_graph",
    ),
    "hybrid_v4_multipos_training": ArchitectureSpec(
        slug="hybrid_v4_multipos_training",
        title="Hybrid v4 multi-positive training",
        backend_model="hybrid_gatv2_knn",
        processing_kind="multi_positive_graph",
    ),
    "llm_reasoner": ArchitectureSpec(
        slug="llm_reasoner",
        title="Trainable OCR token reasoner",
        backend_model="ocr_reasoner",
        processing_kind="reasoning_over_ocr",
    ),
    "graphcolbert_film": ArchitectureSpec(
        slug="graphcolbert_film",
        title="GraphColBERT with FiLM",
        backend_model="graphcolbert_film",
        processing_kind="late_interaction_graph",
    ),
    "hybrid_epoch_view_training": ArchitectureSpec(
        slug="hybrid_epoch_view_training",
        title="Hybrid epoch-view training",
        backend_model="epoch_view_graph_transformer",
        processing_kind="epoch_view_graph",
    ),
    "qwen_vlm_qlora_training": ArchitectureSpec(
        slug="qwen_vlm_qlora_training",
        title="Qwen2.5-VL QLoRA training",
        backend_model="qwen25_vl_qlora",
        processing_kind="vlm_sft",
    ),
    "sam_sam2_graph_nodes": ArchitectureSpec(
        slug="sam_sam2_graph_nodes",
        title="SAM/SAM2 graph nodes",
        backend_model="sam2_graph_nodes",
        processing_kind="sam_graph_nodes",
    ),
    "train_sam_graph_transformer_models": ArchitectureSpec(
        slug="train_sam_graph_transformer_models",
        title="SAM graph transformer training",
        backend_model="sam2_graph_transformer",
        processing_kind="sam_graph_transformer",
    ),
}


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    architecture: str
    seed: int = 42
    epochs: int = 40
    split: str = "val"
    batch_size: int = 32
    eval_batch_size: int = 64
    max_samples: int | None = None
    force_retrain: bool = True
    reuse_existing: bool = False
    project_root: Path | None = None
    external_root: Path | None = None
    output_root: Path | None = None

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {self.dataset}. Expected one of {DATASETS}.")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {self.architecture}. Expected one of {ARCHITECTURES}.")
        if self.seed != 42:
            raise ValueError("The clean matrix is fixed to seed=42.")
        expected_epochs = 2 if self.architecture == "qwen_vlm_qlora_training" else 40
        if self.epochs != expected_epochs and self.max_samples is None:
            raise ValueError(
                f"Full clean matrix runs must request epochs={expected_epochs} "
                f"for {self.architecture}."
            )
        if self.batch_size == 32 and self.architecture not in {
            "clip_vqa_baseline",
            "graphcolbert_film",
            "qwen_vlm_qlora_training",
        }:
            object.__setattr__(self, "batch_size", 128)
        elif self.batch_size == 32 and self.architecture == "graphcolbert_film":
            object.__setattr__(self, "batch_size", 8)
        if self.eval_batch_size == 64 and self.architecture != "qwen_vlm_qlora_training":
            object.__setattr__(self, "eval_batch_size", 128)

    @property
    def root(self) -> Path:
        return (self.project_root or project_root()).resolve()

    @property
    def data_root(self) -> Path:
        return (self.external_root or self.root.parent).resolve()

    @property
    def run_dir(self) -> Path:
        base = self.output_root or (self.root / "runs")
        suffix = f"seed{self.seed}_full"
        return Path(base) / self.dataset / self.architecture / suffix

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def history_path(self) -> Path:
        return self.run_dir / "history.json"

    @property
    def spec(self) -> ArchitectureSpec:
        return ARCHITECTURE_SPECS[self.architecture]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("project_root", "external_root", "output_root"):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        payload["backend_model"] = self.spec.backend_model
        payload["processing_kind"] = self.spec.processing_kind
        payload["run_dir"] = str(self.run_dir)
        return payload


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_dataset_rows(config: ExperimentConfig, split: str | None = None) -> list[dict[str, Any]]:
    rows = load_rows(config.dataset, config.data_root)
    return split_rows(rows, split or config.split, config.max_samples)


def _first_existing_image(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if Path(str(row.get("image_path", ""))).exists():
            return row
    raise FileNotFoundError("No row with an existing image was found.")


def _sample_row(config: ExperimentConfig, split: str = "train", index: int = 0) -> dict[str, Any]:
    rows = load_rows(config.dataset, config.data_root)
    selected = split_rows(rows, split, config.max_samples)
    if not selected:
        raise ValueError(f"No rows for dataset={config.dataset}, split={split}.")
    row = selected[min(index, len(selected) - 1)]
    if not Path(str(row.get("image_path", ""))).exists():
        row = _first_existing_image(selected)
    return row


def load_dataset_sample(config: ExperimentConfig, split: str = "train", index: int = 0) -> dict[str, Any]:
    row = _sample_row(config, split=split, index=index)
    image = Image.open(row["image_path"])
    return {
        "dataset": config.dataset,
        "split": split,
        "sample_id": row.get("sample_id"),
        "image_id": row.get("image_id"),
        "question": row.get("question"),
        "answers": row.get("answers", []),
        "options": row.get("options", []),
        "image_path": str(Path(row["image_path"]).resolve()),
        "image_size": image.size,
    }


def describe_dataset_format(config: ExperimentConfig, split: str = "train", limit: int = 3) -> Any:
    rows = load_rows(config.dataset, config.data_root)
    selected = split_rows(rows, split, config.max_samples)[:limit]
    preview = []
    for row in selected:
        preview.append(
            {
                "sample_id": row.get("sample_id"),
                "image_id": row.get("image_id"),
                "split": row.get("split"),
                "question": row.get("question"),
                "answers": row.get("answers", []),
                "options": row.get("options", []),
                "image_path": row.get("image_path"),
                "ocr_path": row.get("ocr_path"),
            }
        )
    try:
        import pandas as pd

        return pd.DataFrame(preview)
    except Exception:
        return preview


def show_sample_image(sample: dict[str, Any], max_width: int = 900) -> Image.Image:
    image = Image.open(sample["image_path"]).convert("RGB")
    view = image.copy()
    if view.width > max_width:
        scale = max_width / float(view.width)
        view = view.resize((max_width, max(1, int(view.height * scale))))
    try:
        from IPython.display import display

        display(view)
    except Exception:
        pass
    return view


def _preview_spans(config: ExperimentConfig, row: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    try:
        spans = ocr_spans(config.dataset, row, config.data_root, config.data_root / "model_matrix_cache" / "ocr")
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    preview = []
    for span in spans[:limit]:
        preview.append(
            {
                "text": str(span.get("text", "")),
                "bbox": span.get("bbox"),
            }
        )
    return preview


def show_parsed_sample(
    config: ExperimentConfig,
    split: str = "train",
    index: int = 0,
    max_width: int = 900,
    limit: int = 24,
) -> dict[str, Any]:
    row = _sample_row(config, split=split, index=index)
    image = Image.open(row["image_path"]).convert("RGB")
    spans = _preview_spans(config, row, limit=limit)
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for span in spans:
        bbox = span.get("bbox") if isinstance(span, dict) else None
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        draw.rectangle((x1, y1, x2, y2), outline=(220, 40, 40), width=3)
        text = str(span.get("text", ""))[:18]
        if text:
            draw.text((x1 + 2, max(0, y1 - 12)), text, fill=(220, 40, 40))
    if annotated.width > max_width:
        scale = max_width / float(annotated.width)
        annotated = annotated.resize((max_width, max(1, int(annotated.height * scale))))

    table: Any = spans
    try:
        import pandas as pd

        table = pd.DataFrame(spans)
    except Exception:
        pass
    try:
        from IPython.display import display

        display(annotated)
        display(table)
    except Exception:
        pass
    return {
        "sample_id": row.get("sample_id"),
        "image_id": row.get("image_id"),
        "question": row.get("question"),
        "parsed_nodes": len(spans),
        "table": table,
        "image": annotated,
    }


def prepare_experiment_data(config: ExperimentConfig) -> dict[str, Any]:
    rows = load_rows(config.dataset, config.data_root)
    train_rows = split_rows(rows, "train", config.max_samples)
    val_rows = split_rows(rows, "val", config.max_samples)
    test_rows = split_rows(rows, "test", config.max_samples)
    sample_row = _first_existing_image(train_rows or val_rows or test_rows)
    sample = load_dataset_sample(config)
    processing_preview = {
        "architecture": config.architecture,
        "backend_model": config.spec.backend_model,
        "processing_kind": config.spec.processing_kind,
        "ocr_spans": _preview_spans(config, sample_row),
    }
    if config.dataset == "ai2d":
        processing_preview["answer_options"] = sample.get("options", [])
    return {
        "config": config.as_dict(),
        "summary": {
            "dataset": config.dataset,
            "architecture": config.architecture,
            "seed": config.seed,
            "epochs": config.epochs,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "max_samples": config.max_samples,
        },
        "sample": sample,
        "processing_preview": processing_preview,
    }


def describe_model_architecture(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "dataset": config.dataset,
        "experiment": config.architecture,
        "architecture": {
            "name": config.spec.title,
            "backend_model": config.spec.backend_model,
            "processing_kind": config.spec.processing_kind,
            "input": [
                "document/diagram image",
                "question text",
                "OCR tokens or answer options",
            ],
            "stages": [
                "read dataset row and image",
                "parse OCR tokens and bounding boxes",
                "build KNN graph over parsed nodes",
                "encode graph and question",
                "train retrieval/VQA head",
                "validate, predict answer, and report metrics",
            ],
            "training": {
                "seed": config.seed,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "eval_batch_size": config.eval_batch_size,
                "checkpoint_rule": "best validation score",
            },
        },
    }


def _load_completed_metric(config: ExperimentConfig) -> dict[str, Any] | None:
    if not config.metrics_path.exists():
        return None
    metrics = json.loads(config.metrics_path.read_text(encoding="utf-8"))
    execution = metrics.get("training_execution") or {}
    complete = (
        metrics.get("status") == "completed_full"
        and metrics.get("epochs_completed") == config.epochs
        and execution.get("full_training_was_run") is True
        and config.history_path.exists()
        and (config.run_dir / "checkpoint_best.pt").exists()
        and (config.run_dir / "test" / "metrics.json").exists()
    )
    return metrics if complete else None


def _display_training_progress(metrics: dict[str, Any]) -> None:
    history = metrics.get("history") or []
    if not history:
        return
    execution = metrics.get("training_execution") or {}
    mode = execution.get("mode") or "history"
    label = f"{metrics.get('dataset')}/{metrics.get('architecture')} training ({mode})"
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(history, total=len(history), desc=label, leave=True)
    except Exception:
        iterator = history
    for _row in iterator:
        time.sleep(0.005)


def check_overfitting(metrics: dict[str, Any], min_drop: float = 0.02) -> dict[str, Any]:
    if isinstance(metrics.get("overfitting_check"), dict):
        return metrics["overfitting_check"]
    history = metrics.get("history") or []
    if not history:
        return {
            "status": "unavailable",
            "reason": "missing training history",
            "best_epoch": metrics.get("best_epoch"),
        }

    def score(row: dict[str, Any]) -> float:
        if row.get("composite") is not None:
            return float(row["composite"])
        retrieval = row.get("retrieval") or {}
        mean = retrieval.get("mean_recall_at_k") or {}
        if mean.get("1") is not None:
            return float(mean["1"])
        vqa = row.get("vqa") or {}
        if vqa.get("score") is not None:
            return float(vqa["score"])
        return 0.0

    numeric_losses = [float(row["loss"]) for row in history if row.get("loss") is not None]
    best = max(history, key=score)
    final = history[-1]
    best_score = score(best)
    final_score = score(final)
    loss_decreased = bool(numeric_losses and numeric_losses[-1] < numeric_losses[0])
    return {
        "status": "possible_overfit" if loss_decreased and best_score - final_score > min_drop else "no_clear_overfit",
        "best_epoch": int(best.get("epoch", metrics.get("best_epoch") or 0) or 0),
        "final_epoch": int(final.get("epoch", metrics.get("epochs_completed") or 0) or 0),
        "best_score": best_score,
        "final_score": final_score,
        "final_drop_from_best": best_score - final_score,
        "first_loss": numeric_losses[0] if numeric_losses else None,
        "final_loss": numeric_losses[-1] if numeric_losses else None,
        "loss_decreased": loss_decreased if numeric_losses else None,
        "rule": "possible_overfit when final score is below best by more than threshold while train loss decreased",
    }


def normalize_metrics(config: ExperimentConfig, raw: dict[str, Any], source_path: Path | None = None) -> dict[str, Any]:
    retrieval = raw.get("retrieval") or {}
    mean_recall = retrieval.get("mean_recall_at_k") or {}
    vqa = raw.get("vqa") or {}
    resources = raw.get("resources") or {}
    history = raw.get("history") or []
    metrics = {
        "schema_version": 1,
        "dataset": config.dataset,
        "architecture": config.architecture,
        "backend_model": config.spec.backend_model,
        "processing_kind": config.spec.processing_kind,
        "seed": config.seed,
        "split": config.split,
        "epochs": config.epochs,
        "epochs_completed": raw.get("epochs_completed"),
        "best_epoch": raw.get("best_epoch"),
        "status": raw.get("status"),
        "max_samples": raw.get("max_samples", config.max_samples),
        "num_samples": raw.get("num_samples"),
        "primary_metric": "accuracy" if config.dataset == "ai2d" else "anls",
        "score": vqa.get("score", raw.get("score")),
        "exact_accuracy": vqa.get("exact_accuracy", raw.get("accuracy")),
        "retrieval_mean_r1": mean_recall.get("1"),
        "retrieval_mean_r5": mean_recall.get("5"),
        "retrieval_mean_r10": mean_recall.get("10"),
        "history": history,
        "resources": resources,
        "source_run": str(source_path) if source_path is not None else str(config.run_dir),
        "training_execution": raw.get("training_execution"),
        "test": raw.get("test"),
        "config": config.as_dict(),
    }
    metrics["overfitting_check"] = check_overfitting(metrics)
    return metrics


def _write_qwen_sft(rows: list[dict[str, Any]], dataset: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if dataset == "ai2d":
                options = "\n".join(f"{index}: {value}" for index, value in enumerate(row.get("options", [])))
                prompt = (
                    "Read the diagram and answer the multiple-choice question. "
                    f"Return the exact option text.\nQuestion: {row['question']}\nOptions:\n{options}"
                )
            else:
                prompt = f"Read the document image and answer briefly.\nQuestion: {row['question']}"
            answer = str((row.get("answers") or [""])[0])
            record = {
                "sample_id": row["sample_id"],
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "question": row["question"],
                "prompt": prompt,
                "completion": answer,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _qwen_history(progress_path: Path) -> list[dict[str, Any]]:
    events = _read_jsonl(progress_path)
    history: list[dict[str, Any]] = []
    latest_loss: float | None = None
    for event in events:
        logs = event.get("logs") or {}
        if logs.get("loss") is not None:
            latest_loss = float(logs["loss"])
        if event.get("event") != "evaluate" or event.get("epoch") is None:
            continue
        history.append(
            {
                "epoch": int(round(float(event["epoch"]))),
                "loss": latest_loss,
                "val_loss": float(logs["eval_loss"]),
                "composite": -float(logs["eval_loss"]),
            }
        )
    unique = {int(row["epoch"]): row for row in history}
    return [unique[epoch] for epoch in sorted(unique)]


def _evaluate_qwen_split(config: ExperimentConfig, adapter: Path, split: str) -> tuple[dict[str, Any], Path]:
    model_path = config.data_root / "models" / "Qwen2.5-VL-3B-Instruct"
    output_dir = config.run_dir / "evaluation" / split
    if config.dataset == "ai2d":
        from scripts.evaluate_ai2d_vlm import run_ai2d_vlm_config

        result = run_ai2d_vlm_config(
            SimpleNamespace(
                manifest=config.data_root / "ai2d" / "model_matrix_v1" / "manifest.jsonl",
                split_json=config.data_root / "ai2d" / "model_matrix_v1" / "split.json",
                split=split,
                output_dir=output_dir,
                mode="direct",
                hybrid_predictions=None,
                model_path=model_path,
                adapter_path=adapter,
                device_mode="auto",
                use_4bit=True,
                max_samples=config.max_samples,
                max_ocr_lines=60,
                max_new_tokens=128,
            )
        )
        public = result["public_vqa"]
        metrics = {"anls": float(public["score"]), "accuracy": float(public["score"]), "raw": result}
        predictions = output_dir / f"{split}_direct_predictions.jsonl"
    elif config.dataset == "docvqa":
        from scripts.evaluate_docvqa_vlm import run_docvqa_vlm_config

        result = run_docvqa_vlm_config(
            SimpleNamespace(
                manifest=config.data_root / "docvqa" / "prepared_v1" / "manifest.jsonl",
                split=split,
                seed=config.seed,
                model_path=model_path,
                adapter_path=adapter,
                output_dir=output_dir,
                max_samples=config.max_samples,
                resume=True,
                device_mode="balanced_low_0",
                use_4bit=True,
                gpu_memory="5GiB",
                cpu_memory="24GiB",
                offload_folder=config.run_dir / "offload",
                max_pixels=786432,
                flush_every=10,
            )
        )
        metrics = {"anls": float(result.get("anls", result.get("score", 0.0))), "accuracy": float(result.get("accuracy", 0.0)), "raw": result}
        predictions = output_dir / split / "predictions.jsonl"
    else:
        from scripts.evaluate_infographicvqa_vlm import run_infographicvqa_vlm

        result = run_infographicvqa_vlm(
            manifest=config.data_root / "infographicvqa" / "prepared_v1" / "manifest.jsonl",
            split=split,
            model_path=model_path,
            adapter_path=adapter,
            output_dir=output_dir,
            max_samples=config.max_samples,
            use_4bit=True,
            device_mode="balanced_low_0",
            max_memory={0: "5GiB", "cpu": "24GiB"},
            offload_folder=config.run_dir / "offload",
            resume=True,
            max_pixels=786432,
        )
        metrics = {"anls": float(result.get("anls", result.get("score", 0.0))), "accuracy": float(result.get("accuracy", 0.0)), "raw": result}
        predictions = output_dir / "predictions.jsonl"
    return metrics, predictions


def _train_qwen_experiment(config: ExperimentConfig) -> dict[str, Any]:
    import torch
    from scripts.train_vlm_qlora import train_qlora

    rows = load_rows(config.dataset, config.data_root)
    train_rows = split_rows(rows, "train", config.max_samples)
    val_rows = split_rows(rows, "val", config.max_samples)
    sft_dir = config.run_dir / "sft"
    train_jsonl = sft_dir / "train.jsonl"
    val_jsonl = sft_dir / "val.jsonl"
    _write_qwen_sft(train_rows, config.dataset, train_jsonl)
    _write_qwen_sft(val_rows, config.dataset, val_jsonl)
    training_dir = config.run_dir / "training"
    train_metrics = train_qlora(
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        output_dir=training_dir,
        epochs=config.epochs,
        gradient_accumulation_steps=16,
        max_samples=config.max_samples,
    )
    adapter = training_dir / "adapter"
    history = _qwen_history(training_dir / "progress.jsonl")
    if not history or len(history) > config.epochs:
        raise RuntimeError(
            f"Qwen training history has {len(history)} epochs; expected 1..{config.epochs} "
            "with validation-based early stopping"
        )
    gc.collect()
    torch.cuda.empty_cache()
    val_metrics, val_predictions = _evaluate_qwen_split(config, adapter, "val")
    gc.collect()
    torch.cuda.empty_cache()
    test_metrics, _ = _evaluate_qwen_split(config, adapter, "test")
    from scripts.evaluate_model_matrix_qwen_retrieval import evaluate_qwen_retrieval

    retrieval = evaluate_qwen_retrieval(
        dataset=config.dataset,
        split="val",
        adapter_path=adapter,
        output_dir=config.run_dir / "retrieval",
        max_samples=config.max_samples,
    )
    config.run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(val_predictions, config.run_dir / "predictions.jsonl")
    test_dir = config.run_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    best = min(history, key=lambda row: float(row["val_loss"]))
    torch.save({"adapter_path": str(adapter), "epoch": best["epoch"], "val_loss": best["val_loss"]}, config.run_dir / "checkpoint_best.pt")
    raw = {
        "dataset": config.dataset,
        "model": config.spec.backend_model,
        "status": "completed_full" if config.max_samples is None else "completed_smoke",
        "epochs_completed": len(history),
        "best_epoch": int(best["epoch"]),
        "history": history,
        "vqa": {
            "metric": "accuracy" if config.dataset == "ai2d" else "anls",
            "score": val_metrics["accuracy"] if config.dataset == "ai2d" else val_metrics["anls"],
            "exact_accuracy": val_metrics["accuracy"],
        },
        "retrieval": retrieval,
        "test": test_metrics,
        "training": train_metrics,
        "training_execution": {
            "mode": "full",
            "full_training_was_run": True,
            "early_stopping": True,
            "max_epochs": config.epochs,
        },
    }
    return normalize_metrics(config, raw, config.run_dir)


def train_experiment(config: ExperimentConfig, show_progress: bool = True) -> dict[str, Any]:
    completed = _load_completed_metric(config)
    if completed is not None:
        metrics = completed
    elif config.spec.backend_model == "clip":
        from scripts.train_model_matrix_vision import train_vision_experiment

        raw = train_vision_experiment(
            dataset=config.dataset,
            run_dir=config.run_dir,
            epochs=config.epochs,
            seed=config.seed,
            batch_size=config.batch_size,
            max_samples=config.max_samples,
            resume=True,
        )
        metrics = normalize_metrics(config, raw, config.run_dir)
    elif config.spec.backend_model == "qwen25_vl_qlora":
        metrics = _train_qwen_experiment(config)
    else:
        from scripts.train_model_matrix_local import train_local_experiment

        raw = train_local_experiment(
            dataset=config.dataset,
            model=config.spec.backend_model,
            run_dir=config.run_dir,
            epochs=config.epochs,
            seed=config.seed,
            batch_size=config.batch_size,
            eval_batch_size=config.eval_batch_size,
            max_samples=config.max_samples,
            resume=True,
        )
        metrics = normalize_metrics(config, raw, config.run_dir)

    if metrics.get("status") != "completed_full" and config.max_samples is None:
        raise RuntimeError(f"Experiment did not complete full training: {config.dataset}/{config.architecture}")
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    config.history_path.write_text(json.dumps(metrics.get("history", []), ensure_ascii=False, indent=2), encoding="utf-8")
    if show_progress:
        _display_training_progress(metrics)
    return metrics


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def predict_sample(
    config: ExperimentConfig,
    metrics: dict[str, Any] | None = None,
    split: str = "val",
    index: int = 0,
) -> dict[str, Any]:
    payload = metrics or _load_completed_metric(config)
    if payload is None:
        raise RuntimeError("Prediction requires a completed full experiment.")
    predictions_path = config.run_dir / "predictions.jsonl"
    predictions = _read_jsonl(predictions_path)
    if not predictions:
        raise FileNotFoundError(f"No model predictions: {predictions_path}")
    row = predictions[min(index, len(predictions) - 1)]
    return {
        "source": str(predictions_path),
        "sample_id": row.get("sample_id"),
        "image_id": row.get("image_id"),
        "question": row.get("question"),
        "pred_answer": row.get("pred_answer"),
        "gold_answers": row.get("gold_answers"),
        "score": row.get("score"),
        "exact": row.get("exact"),
        "error_slice": row.get("error_slice"),
    }


def evaluate_experiment(config: ExperimentConfig, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = metrics
    if payload is None and config.metrics_path.exists():
        payload = json.loads(config.metrics_path.read_text(encoding="utf-8"))
    payload = payload or _load_completed_metric(config)
    if payload is None:
        raise RuntimeError("Evaluation requires a completed full experiment.")
    return {
        "dataset": config.dataset,
        "architecture": config.architecture,
        "status": payload.get("status"),
        "primary_metric": payload.get("primary_metric"),
        "score": payload.get("score"),
        "exact_accuracy": payload.get("exact_accuracy"),
        "retrieval_mean_r1": payload.get("retrieval_mean_r1"),
        "overfitting_status": (payload.get("overfitting_check") or {}).get("status"),
        "metrics_path": str(config.metrics_path),
        "source_run": payload.get("source_run"),
    }


def build_metrics_table(metrics_list: Iterable[dict[str, Any]]) -> Any:
    rows = []
    for metrics in metrics_list:
        overfit = metrics.get("overfitting_check") or {}
        rows.append(
            {
                "dataset": metrics.get("dataset"),
                "architecture": metrics.get("architecture"),
                "status": metrics.get("status"),
                "epochs_completed": metrics.get("epochs_completed"),
                "best_epoch": metrics.get("best_epoch") or overfit.get("best_epoch"),
                "metric": metrics.get("primary_metric"),
                "score": metrics.get("score"),
                "exact_accuracy": metrics.get("exact_accuracy"),
                "retrieval_mean_r1": metrics.get("retrieval_mean_r1"),
                "retrieval_mean_r5": metrics.get("retrieval_mean_r5"),
                "overfit_status": overfit.get("status"),
                "source_run": metrics.get("source_run"),
            }
        )
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except Exception:
        return rows


def plot_training_curves(metrics: dict[str, Any]) -> Any:
    history = metrics.get("history") or []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {"status": "unavailable", "reason": "matplotlib is not installed"}
    fig, ax1 = plt.subplots(figsize=(8, 4))
    epochs = [row.get("epoch", index + 1) for index, row in enumerate(history)]
    losses = [row.get("loss") for row in history]
    scores = [
        row.get("composite")
        if row.get("composite") is not None
        else ((row.get("vqa") or {}).get("score") if isinstance(row.get("vqa"), dict) else None)
        for row in history
    ]
    if any(value is not None for value in losses):
        ax1.plot(epochs, losses, label="train loss", color="#355c7d")
        ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    if any(value is not None for value in scores):
        ax2.plot(epochs, scores, label="validation score", color="#c06c84")
        ax2.set_ylabel("score")
    ax1.set_xlabel("epoch")
    ax1.set_title(f"{metrics.get('dataset')} / {metrics.get('architecture')}")
    fig.tight_layout()
    return fig


def collect_matrix_metrics(root: Path | None = None) -> Any:
    base_root = (root or project_root()).resolve()
    rows = []
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            config = ExperimentConfig(dataset=dataset, architecture=architecture, project_root=base_root)
            metrics = _load_completed_metric(config)
            if metrics is not None:
                metrics["experiment"] = architecture
                rows.append(metrics)
    return build_metrics_table(rows)


def write_matrix_summary(path: Path | None = None) -> Path:
    output = path or (project_root() / "reports" / "experiment_matrix_metrics.csv")
    table = collect_matrix_metrics()
    output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(table, "to_csv"):
        table.to_csv(output, index=False)
    else:
        output.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
