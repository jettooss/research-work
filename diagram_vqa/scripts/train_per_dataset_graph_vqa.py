"""Sequentially train one independent Graph Transformer per VQA dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ai2d", "infographicvqa", "docvqa")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train three separate per-dataset Graph Transformer models.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/per_dataset_graph_vqa")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    for dataset in args.datasets:
        command = [
            sys.executable, "scripts/train_model_matrix_local.py",
            "--dataset", dataset, "--model", "graph_transformer", "--split", "val",
            "--seed", str(args.seed), "--output-dir", str(args.output_dir),
            "--epochs", str(args.epochs), "--early-stopping-patience", str(args.early_stopping_patience),
            "--batch-size", str(args.batch_size), "--eval-batch-size", str(args.eval_batch_size),
            "--resume" if args.resume else "--no-resume",
        ]
        if args.max_samples is not None:
            command.extend(["--max-samples", str(args.max_samples)])
        print(f"[PER-DATASET {dataset}]", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
