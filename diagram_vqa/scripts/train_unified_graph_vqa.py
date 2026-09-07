"""Jointly train one OCR Graph Transformer on AI2D, InfographicVQA and DocVQA.

The checkpoint is deliberately shared: there is no dataset-specific encoder or
head.  Dataset boundaries are retained only for balanced sampling and for
reporting the native VQA metric of each benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch_geometric.data import Batch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vqa_retrieval.model_matrix import DATASETS, ensure_free_space, load_rows, split_rows  # noqa: E402
from train_model_matrix_local import (  # noqa: E402
    GraphMatrixModel,
    MatrixFeatureStore,
    evaluate,
    load_torch,
    multipositive_loss,
    seed_everything,
    write_jsonl,
)


class MixedRows(Dataset):
    def __init__(self, rows_by_dataset: dict[str, list[dict[str, Any]]]) -> None:
        self.items = [
            {**row, "_dataset": dataset}
            for dataset, rows in rows_by_dataset.items()
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def mixed_collate(rows: list[dict[str, Any]], stores: dict[str, MatrixFeatureStore]) -> dict[str, Any]:
    return {
        "rows": rows,
        "graphs": [stores[row["_dataset"]].graph(row) for row in rows],
        "questions": torch.stack([stores[row["_dataset"]].question(row) for row in rows]),
        "question_tokens": [stores[row["_dataset"]].question_tokens(row) for row in rows],
        # Prefix prevents an identically named image in two datasets becoming a
        # false retrieval positive.
        "image_ids": [f"{row['_dataset']}:{row['image_id']}" for row in rows],
    }


def native_metrics(per_dataset: dict[str, dict[str, Any]]) -> tuple[float, float]:
    mean_vqa = float(np.mean([item["vqa"]["score"] for item in per_dataset.values()]))
    mean_r1 = float(np.mean([item["retrieval"]["mean_recall_at_k"]["1"] for item in per_dataset.values()]))
    return mean_vqa, mean_r1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one joint Graph Transformer over all three VQA datasets.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test cap per dataset and split.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/unified_graph_vqa")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Optional cap; omitted means one balanced pass.")
    args = parser.parse_args()

    run_dir = args.output_dir / f"seed{args.seed}"
    ensure_free_space(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    if args.resume and metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("status") in {"available_full", "available_smoke"}:
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_rows = {dataset: split_rows(load_rows(dataset, EXTERNAL_ROOT), "train", args.max_samples) for dataset in args.datasets}
    val_rows = {dataset: split_rows(load_rows(dataset, EXTERNAL_ROOT), "val", args.max_samples) for dataset in args.datasets}
    stores = {
        dataset: MatrixFeatureStore(dataset, "graph_transformer", EXTERNAL_ROOT / "model_matrix_cache", device)
        for dataset in args.datasets
    }
    mixed = MixedRows(train_rows)
    if not mixed.items:
        raise RuntimeError("No joint training rows were found")
    # Give each dataset equal total probability even when its manifest is larger.
    sizes = {dataset: max(1, len(rows)) for dataset, rows in train_rows.items()}
    weights = [1.0 / sizes[row["_dataset"]] for row in mixed.items]
    sampler = WeightedRandomSampler(weights, num_samples=len(mixed), replacement=True)
    loader = DataLoader(
        mixed, batch_size=args.batch_size, sampler=sampler,
        collate_fn=lambda rows: mixed_collate(rows, stores),
    )
    model = GraphMatrixModel("graph_transformer").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    config = vars(args) | {"output_dir": str(args.output_dir), "architecture": "shared_graph_transformer"}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_composite = -math.inf
    stale = 0
    epochs = min(args.epochs, 2) if args.max_samples is not None else args.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        steps = 0
        iterator = tqdm(loader, desc=f"unified seed{args.seed} epoch {epoch}/{epochs}", disable=os.environ.get("DISABLE_TQDM") == "1")
        for payload in iterator:
            if args.steps_per_epoch is not None and steps >= args.steps_per_epoch:
                break
            graphs = Batch.from_data_list(payload["graphs"]).to(device)
            questions = payload["questions"].to(device)
            retrieval = model.retrieval_logits(graphs, questions, payload["question_tokens"])
            retrieval_loss = multipositive_loss(retrieval, payload["image_ids"])
            answer_losses = []
            for row, graph, question in zip(payload["rows"], payload["graphs"], questions):
                store = stores[row["_dataset"]]
                _, candidates, target = store.candidates(row, graph)
                logits = model.vqa_logits(graph, question, candidates)
                answer_losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=device)))
            loss = 0.5 * retrieval_loss + 0.5 * torch.stack(answer_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            steps += 1

        per_dataset = {dataset: evaluate(model, rows, stores[dataset], device, args.eval_batch_size)[0] for dataset, rows in val_rows.items()}
        mean_vqa, mean_r1 = native_metrics(per_dataset)
        composite = 0.5 * mean_vqa + 0.5 * mean_r1
        record = {"epoch": epoch, "loss": loss_sum / max(1, steps), "composite": composite, "macro_vqa": mean_vqa, "macro_mean_r1": mean_r1, "datasets": per_dataset}
        history.append(record)
        (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        if composite > best_composite:
            best_composite, stale = composite, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "composite": composite}, run_dir / "checkpoint_best.pt")
        else:
            stale += 1
        if stale >= args.early_stopping_patience:
            break

    best = load_torch(run_dir / "checkpoint_best.pt")
    model.load_state_dict(best["model"])
    final: dict[str, Any] = {}
    for dataset, rows in val_rows.items():
        evaluation, predictions = evaluate(model, rows, stores[dataset], device, args.eval_batch_size)
        final[dataset] = evaluation
        write_jsonl(predictions, run_dir / dataset / "val_predictions.jsonl")
        test_rows = split_rows(load_rows(dataset, EXTERNAL_ROOT), "test", args.max_samples)
        test_evaluation, test_predictions = evaluate(model, test_rows, stores[dataset], device, args.eval_batch_size)
        write_jsonl(test_predictions, run_dir / dataset / "test_predictions.jsonl")
        (run_dir / dataset / "test_metrics.json").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / dataset / "test_metrics.json").write_text(json.dumps(test_evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        if dataset != "ai2d":
            submission = [{"questionId": item["question_id"], "answer": item["pred_answer"]} for item in test_predictions]
            (run_dir / dataset / "test_submission.json").write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")

    macro_vqa, macro_r1 = native_metrics(final)
    elapsed = time.perf_counter() - started
    metrics = {
        "model": "unified_graph_transformer",
        "datasets": list(args.datasets),
        "seed": args.seed,
        "status": "available_full" if args.max_samples is None else "available_smoke",
        "epochs_completed": len(history), "best_epoch": int(best["epoch"]), "best_composite": float(best["composite"]),
        "macro": {"vqa_score": macro_vqa, "mean_recall_at_1": macro_r1},
        "per_dataset": final,
        "resources": {
            "wall_time_seconds": elapsed,
            "peak_ram_mb": psutil.Process().memory_info().rss / (1024 * 1024),
            "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "config": config,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
