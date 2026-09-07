from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
VARIANTS = ("text_only", "coords_no_edges", "knn_k3", "knn_k5", "typed_spatial")
SEEDS = (42, 43, 44)


def is_full(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "available_full"
    except (OSError, json.JSONDecodeError):
        return False


def run(*arguments: str) -> None:
    command = [str(PYTHON), *arguments]
    print("[RUN]", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and resume the complete controlled DocVQA experiment.")
    parser.add_argument("--phase", choices=["all", "baselines", "gnn", "qwen", "finalize"], default="all")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    args = parser.parse_args()

    if args.phase in {"all", "baselines"}:
        run("scripts/prepare_docvqa_manifest.py")
        for mode in ("most_frequent", "random_train_answer", "question_retrieval"):
            metrics = ROOT / "runs" / "docvqa_baselines" / "open_answer" / mode / "val" / "metrics.json"
            if not is_full(metrics):
                run("scripts/evaluate_docvqa_baselines.py", "--family", "open_answer", "--mode", mode, "--split", "val")
        for split in ("val", "test"):
            for mode in ("random", "ocr_text"):
                metrics = ROOT / "runs" / "docvqa_baselines" / "retrieval" / mode / split / "metrics.json"
                if not is_full(metrics):
                    run("scripts/evaluate_docvqa_baselines.py", "--family", "retrieval", "--mode", mode, "--split", split)
            clip_metrics = ROOT / "runs" / "docvqa_clip_retrieval" / split / "metrics.json"
            if not is_full(clip_metrics):
                run("scripts/evaluate_docvqa_clip_retrieval.py", "--split", split)

    if args.phase in {"all", "gnn"}:
        for variant in VARIANTS:
            for seed in SEEDS:
                metrics = ROOT / "runs" / "docvqa_gnn" / variant / f"seed{seed}" / "metrics.json"
                if is_full(metrics):
                    continue
                run(
                    "scripts/train_docvqa_gnn.py",
                    "--variant", variant,
                    "--seed", str(seed),
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--eval-batch-size", str(args.eval_batch_size),
                    "--early-stopping-patience", str(args.early_stopping_patience),
                    "--minimum-free-gb", "6",
                )
        run("scripts/aggregate_docvqa_results.py")

    if args.phase in {"all", "qwen"}:
        for split in ("val", "test"):
            metrics = ROOT / "runs" / "docvqa_qwen25_vl" / split / "metrics.json"
            if not is_full(metrics):
                run(
                    "scripts/run_docvqa_vlm_resilient.py",
                    "--split", split,
                    "--max-pixels", "786432",
                    "--flush-every", "1",
                )

    if args.phase in {"all", "finalize"}:
        run("scripts/aggregate_docvqa_results.py")
        run("scripts/audit_docvqa_defense.py")


if __name__ == "__main__":
    main()
