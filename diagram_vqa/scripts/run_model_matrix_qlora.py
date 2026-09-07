from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vqa_retrieval.model_matrix import DATASETS, ensure_free_space, load_rows, split_rows  # noqa: E402


def prompt(dataset: str, row: dict) -> str:
    if dataset == "ai2d":
        options = "\n".join(f"{index}: {value}" for index, value in enumerate(row.get("options", [])))
        return f'Read the diagram and answer the multiple-choice question. Return JSON only: {{"answer": "exact option text"}}.\nQuestion: {row["question"]}\nOptions:\n{options}'
    return f'Read the document image and answer the question. Return JSON only: {{"answer": "short answer"}}.\nQuestion: {row["question"]}'


def completion(row: dict) -> str:
    answers = row.get("answers", [])
    return json.dumps({"answer": str(answers[0]) if answers else ""}, ensure_ascii=False)


def write_sft(rows: list[dict], dataset: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {
                "sample_id": row["sample_id"], "image_id": row["image_id"], "image_path": row["image_path"],
                "question": row["question"], "prompt": prompt(dataset, row), "completion": completion(row),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(command: list[str]) -> None:
    print("[RUN]", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one dataset-specific Qwen2.5-VL QLoRA adapter.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=["qwen25_vl_qlora"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--seed", type=int, choices=[42], default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/model_matrix")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    seed_dir = args.output_dir / args.dataset / args.model / "seed42"
    run_dir = seed_dir / args.split
    ensure_free_space(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "dataset": args.dataset, "model": args.model, "split": args.split, "seed": 42,
        "max_samples": args.max_samples, "base_model": str((EXTERNAL_ROOT / "models/Qwen2.5-VL-3B-Instruct").resolve()),
        "quantization": "4-bit NF4 double quant", "lora_rank": 16, "lora_alpha": 32,
        "epochs": 3, "microbatch": 1, "gradient_accumulation_steps": 8, "resume": args.resume,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path = run_dir / "metrics.json"
    if args.resume and metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        test_ready = args.split == "test" or (run_dir.parent / "test/metrics.json").exists()
        if existing.get("status") in {"available_full", "available_smoke"} and test_ready:
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return
    rows = load_rows(args.dataset, EXTERNAL_ROOT)
    train_rows = split_rows(rows, "train", args.max_samples)
    val_rows = split_rows(rows, "val", args.max_samples)
    eval_rows = split_rows(rows, args.split, args.max_samples)
    sft_dir = seed_dir / "sft"
    train_jsonl, val_jsonl = sft_dir / "train.jsonl", sft_dir / "val.jsonl"
    write_sft(train_rows, args.dataset, train_jsonl)
    write_sft(val_rows, args.dataset, val_jsonl)
    training_dir = seed_dir / "training"
    adapter = training_dir / "adapter"
    started = time.perf_counter()
    if not adapter.exists():
        command = [
            sys.executable, "scripts/train_vlm_qlora.py", "--train-jsonl", str(train_jsonl), "--val-jsonl", str(val_jsonl),
            "--output-dir", str(training_dir), "--num-train-epochs", "3", "--learning-rate", "0.0002",
            "--per-device-train-batch-size", "1", "--per-device-eval-batch-size", "1",
            "--gradient-accumulation-steps", "8", "--lora-r", "16", "--lora-alpha", "32",
            "--gradient-checkpointing", "--image-max-pixels", "786432", "--save-total-limit", "1",
        ]
        if args.max_samples is not None:
            command.extend(["--max-train-samples", str(args.max_samples), "--max-val-samples", str(args.max_samples)])
        run(command)

    eval_dir = run_dir / "evaluation"
    if args.dataset == "ai2d":
        command = [
            sys.executable, "scripts/evaluate_ai2d_vlm.py", "--manifest", str(EXTERNAL_ROOT / "ai2d/model_matrix_v1/manifest.jsonl"),
            "--split-json", str(EXTERNAL_ROOT / "ai2d/model_matrix_v1/split.json"), "--split", args.split,
            "--output-dir", str(eval_dir), "--adapter-path", str(adapter), "--use-4bit",
        ]
    elif args.dataset == "infographicvqa":
        command = [sys.executable, "scripts/evaluate_infographicvqa_vlm.py", "--split", args.split, "--output-dir", str(eval_dir), "--adapter-path", str(adapter), "--max-pixels", "786432"]
    else:
        command = [sys.executable, "scripts/evaluate_docvqa_vlm.py", "--split", args.split, "--output-dir", str(eval_dir.parent), "--adapter-path", str(adapter), "--max-pixels", "786432", "--flush-every", "1"]
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    run(command)

    if args.dataset == "ai2d":
        raw = json.loads((eval_dir / f"{args.split}_direct_metrics.json").read_text(encoding="utf-8"))
        score = float(raw["public_vqa"]["score"])
        vqa = {"metric": "accuracy", "score": score, "exact_accuracy": score}
    else:
        raw_path = (eval_dir / "metrics.json") if args.dataset == "infographicvqa" else (run_dir / args.split / "metrics.json")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        vqa = {"metric": "anls", "score": float(raw.get("anls", raw.get("score", 0.0))), "exact_accuracy": float(raw.get("accuracy", 0.0))}
        if args.split == "test":
            source_submission = (eval_dir / "submission.json") if args.dataset == "infographicvqa" else (run_dir / args.split / "submission.json")
            if source_submission.exists():
                shutil.copy2(source_submission, run_dir / "submission.json")
    retrieval_dir = run_dir / "retrieval"
    retrieval_command = [
        sys.executable, "scripts/evaluate_model_matrix_qwen_retrieval.py",
        "--dataset", args.dataset, "--split", args.split,
        "--adapter-path", str(adapter), "--output-dir", str(retrieval_dir),
    ]
    if args.max_samples is not None:
        retrieval_command.extend(["--max-samples", str(args.max_samples)])
    run(retrieval_command)
    retrieval = json.loads((retrieval_dir / "metrics.json").read_text(encoding="utf-8"))
    train_metrics = json.loads((training_dir / "train_metrics.json").read_text(encoding="utf-8"))
    elapsed = time.perf_counter() - started
    metrics = {
        "dataset": args.dataset, "model": args.model, "split": args.split, "seed": 42,
        "status": "available_full" if args.max_samples is None else "available_smoke",
        "num_samples": len(eval_rows), "retrieval": retrieval, "vqa": vqa,
        "adapter_path": str(adapter.resolve()), "training": train_metrics,
        "resources": {"wall_time_seconds": elapsed, "peak_ram_mb": psutil.Process().memory_info().rss / (1024 * 1024)},
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    # Intermediate Trainer checkpoints include optimizer state and are not part
    # of the compact experiment artifact once the best adapter exists.
    for checkpoint in training_dir.glob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.split == "val":
        command = [
            sys.executable, str(Path(__file__).resolve()), "--dataset", args.dataset,
            "--model", args.model, "--split", "test", "--seed", "42",
            "--output-dir", str(args.output_dir), "--resume",
        ]
        if args.max_samples is not None:
            command.extend(["--max-samples", str(args.max_samples)])
        run(command)


if __name__ == "__main__":
    main()
