from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
if __package__:
    from .model_matrix_integrity import (
        digest, directory_digest, file_digest, make_config, read_json, validate_directory, write_config,
    )
else:
    from model_matrix_integrity import (
        digest, directory_digest, file_digest, make_config, read_json, validate_directory, write_config,
    )


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
    global EXTERNAL_ROOT
    parser = argparse.ArgumentParser(description="Train one dataset-specific Qwen2.5-VL QLoRA adapter.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=["qwen25_vl_qlora"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="val: train/select on validation, then evaluate test; test: evaluate a saved adapter only")
    parser.add_argument("--seed", type=int, choices=[42], default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/model_matrix")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent,
                        help="Root containing dataset manifests, models and model_matrix_cache")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    EXTERNAL_ROOT = args.data_root.resolve()
    run_qlora_config(args)


def run_qlora_config(args):
    seed_dir = args.output_dir / args.dataset / args.model / "seed42"
    run_dir = seed_dir / args.split
    rows = load_rows(args.dataset, EXTERNAL_ROOT)
    train_config = make_config(args, rows, input_root=EXTERNAL_ROOT, kind="qlora_training",
                               base_model="models/Qwen2.5-VL-3B-Instruct",
                               quantization="4-bit NF4 double quant", lora_rank=16, lora_alpha=32,
                               epochs=3, microbatch=1, gradient_accumulation_steps=8,
                               checkpoint_selection_split="val", checkpoint_selection="eval_loss")
    train_config["split"] = "val"
    config = make_config(args, rows, input_root=EXTERNAL_ROOT, kind="qlora_evaluation", training_config_sha256=digest(train_config))
    training_dir = seed_dir / "training"
    adapter = training_dir / "adapter"
    receipt_path = training_dir / "completion.json"
    if args.split == "test" and not receipt_path.is_file():
        raise ValueError("--split test is evaluation-only and requires a completed, validated adapter; train with --split val first")
    validate_directory(training_dir, train_config, resume=args.resume or args.split == "test")
    validate_directory(run_dir, config, resume=args.resume)
    if args.split == "val":
        validate_directory(seed_dir / "test", dict(config, split="test"), resume=args.resume)
    adapter_hash = None
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        adapter_hash = directory_digest(adapter)
        if (receipt.get("config_sha256") != digest(train_config)
                or receipt.get("adapter_sha256") != adapter_hash
                or receipt.get("train_metrics_sha256") != file_digest(training_dir / "train_metrics.json")):
            raise ValueError(f"Adapter completion receipt does not match the training artifacts: {training_dir}")
    elif adapter.exists():
        raise ValueError(f"Adapter has no verified training completion receipt: {adapter}. Use a fresh --output-dir.")
    train_rows = split_rows(rows, "train", args.max_samples)
    val_rows = split_rows(rows, "val", args.max_samples)
    eval_rows = split_rows(rows, args.split, args.max_samples)
    if not train_rows or not val_rows or not eval_rows:
        raise ValueError("Training/validation/evaluation splits must be nonempty")
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        existing = read_json(metrics_path)
        expected = {"dataset": args.dataset, "model": args.model, "seed": args.seed, "split": args.split,
                    "config_sha256": digest(config), "adapter_sha256": adapter_hash,
                    "status": "available_full" if args.max_samples is None else "available_smoke",
                    "num_samples": len(eval_rows)}
        if adapter_hash is None or any(existing.get(key) != value for key, value in expected.items()):
            raise ValueError(f"QLoRA metrics do not match the requested run/adapter: {run_dir}")
        artifacts = existing.get("evaluation_artifacts_sha256", {})
        if not artifacts:
            raise ValueError(f"QLoRA evaluation has no verifiable source-artifact receipt: {run_dir}")
        for name, checksum in artifacts.items():
            path = (run_dir / name).resolve()
            if not path.is_relative_to(run_dir.resolve()) or not path.is_file() or file_digest(path) != checksum:
                raise ValueError(f"QLoRA evaluation artifact is missing or has changed: {name}")
        if args.split == "val":
            run_qlora_config(argparse.Namespace(**(vars(args) | {"split": "test"})))
        print(json.dumps(existing, ensure_ascii=False, indent=2))
        return existing
    ensure_free_space(run_dir)
    write_config(run_dir, config)
    sft_dir = seed_dir / "sft"
    train_jsonl, val_jsonl = sft_dir / "train.jsonl", sft_dir / "val.jsonl"
    started = time.perf_counter()
    if adapter_hash is None:
        write_config(training_dir, train_config)
        write_sft(train_rows, args.dataset, train_jsonl)
        write_sft(val_rows, args.dataset, val_jsonl)
        command = [
            sys.executable, "scripts/train_vlm_qlora.py", "--train-jsonl", str(train_jsonl), "--val-jsonl", str(val_jsonl),
            "--model-path", str(EXTERNAL_ROOT / "models/Qwen2.5-VL-3B-Instruct"),
            "--output-dir", str(training_dir), "--num-train-epochs", "3", "--learning-rate", "0.0002",
            "--per-device-train-batch-size", "1", "--per-device-eval-batch-size", "1",
            "--gradient-accumulation-steps", "8", "--lora-r", "16", "--lora-alpha", "32",
            "--gradient-checkpointing", "--image-max-pixels", "786432", "--save-total-limit", "1",
        ]
        if args.max_samples is not None:
            command.extend(["--max-train-samples", str(args.max_samples), "--max-val-samples", str(args.max_samples)])
        run(command)
        adapter_hash = directory_digest(adapter)
        receipt_path.write_text(json.dumps({"config_sha256": digest(train_config),
                                           "adapter_sha256": adapter_hash,
                                           "train_metrics_sha256": file_digest(training_dir / "train_metrics.json")}, indent=2), encoding="utf-8")

    eval_dir = run_dir / "evaluation"
    # Child evaluators receive normalized absolute image paths from the selected
    # data root, independent of their own working-directory defaults.
    eval_manifest = run_dir / "evaluation_manifest.jsonl"
    eval_manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    model_path = EXTERNAL_ROOT / "models/Qwen2.5-VL-3B-Instruct"
    if args.dataset == "ai2d":
        command = [
            sys.executable, "scripts/evaluate_ai2d_vlm.py", "--manifest", str(eval_manifest),
            "--split-json", str(EXTERNAL_ROOT / "ai2d/model_matrix_v1/split.json"), "--split", args.split,
            "--output-dir", str(eval_dir), "--adapter-path", str(adapter), "--use-4bit", "--model-path", str(model_path),
        ]
    elif args.dataset == "infographicvqa":
        command = [sys.executable, "scripts/evaluate_infographicvqa_vlm.py", "--split", args.split, "--output-dir", str(eval_dir), "--adapter-path", str(adapter), "--max-pixels", "786432",
                   "--manifest", str(eval_manifest), "--model-path", str(model_path)]
    else:
        command = [sys.executable, "scripts/evaluate_docvqa_vlm.py", "--split", args.split, "--output-dir", str(eval_dir.parent), "--adapter-path", str(adapter), "--max-pixels", "786432", "--flush-every", "1",
                   "--manifest", str(eval_manifest), "--model-path", str(model_path)]
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
        "--data-root", str(EXTERNAL_ROOT),
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
        "config_sha256": digest(config), "adapter_sha256": adapter_hash,
        "adapter_path": str(adapter.resolve()), "training": train_metrics,
        "resources": {"wall_time_seconds": elapsed, "peak_ram_mb": psutil.Process().memory_info().rss / (1024 * 1024)},
    }
    metrics["evaluation_artifacts_sha256"] = {
        path.relative_to(run_dir).as_posix(): file_digest(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".jsonl"}
        and path not in {run_dir / "metrics.json", run_dir / "config.json"}
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.split == "val":
        run_qlora_config(argparse.Namespace(**(vars(args) | {"split": "test", "resume": True})))
    return metrics


if __name__ == "__main__":
    main()
