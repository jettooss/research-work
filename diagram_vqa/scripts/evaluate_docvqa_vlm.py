from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vqa_retrieval.public_vqa_metrics import accuracy_score, evaluate_public_vqa_rows, write_public_vqa_report  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_answer(raw: str) -> str:
    text = str(raw or "").strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return str(payload.get("answer", "")).strip()
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start >= 0:
        try:
            payload, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(payload, dict):
                return str(payload.get("answer", "")).strip()
        except json.JSONDecodeError:
            pass
    return text.splitlines()[0].strip() if text else ""


def prompt(row: dict[str, Any]) -> str:
    return f"""You are answering a DocVQA question from a document image. Return STRICT JSON only.
Schema: {{"answer": "short answer in English", "confidence": 0.0}}
Rules:
- answer only the question and do not explain.
- copy names, dates, numbers, units, and punctuation exactly when visible.
- use an empty answer only when the document does not contain the requested information.

Question: {row['question']}"""


class OutputLock:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / ".evaluate_docvqa_vlm.lock"
        self.fd: int | None = None

    def __enter__(self) -> "OutputLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"Another DocVQA evaluator owns {self.path}") from exc
        os.write(self.fd, f"pid={os.getpid()}\nstarted={time.time()}\n".encode())
        atexit.register(self.cleanup)
        return self

    def cleanup(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.path.unlink(missing_ok=True)

    def __exit__(self, *_: Any) -> None:
        self.cleanup()


def parameter_count_from_safetensors(model_path: Path) -> int | None:
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    total = 0
    for path in model_path.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as handle:
            total += sum(math.prod(handle.get_slice(key).get_shape()) for key in handle.keys())
    return total or None


def ordered_predictions(predictions: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(item["sample_id"]): item for item in predictions}
    return [by_id[str(row["sample_id"])] for row in rows if str(row["sample_id"]) in by_id]


def save_outputs(
    predictions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    output_dir: Path,
    split: str,
    started: float,
    previous_wall: float,
    model_path: Path,
    model_parameter_count: int | None,
    device_mode: str,
    use_4bit: bool,
    max_pixels: int,
    complete_target: int,
) -> dict[str, Any]:
    ordered = ordered_predictions(predictions, rows)
    elapsed = previous_wall + time.perf_counter() - started
    common = {
        "dataset": "docvqa",
        "mode": "qwen25_vl_open_answer",
        "metric_family": "open_answer",
        "split": split,
        "status": "available_full" if len(ordered) == complete_target else "available_partial",
        "num_samples": len(ordered),
        "total_eval_rows": complete_target,
        "wall_time_seconds": elapsed,
        "latency_ms_per_question": elapsed * 1000 / max(1, len(ordered)),
        "throughput_questions_per_second": len(ordered) / elapsed if elapsed else None,
        "peak_ram_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
        "parameter_count": parameter_count_from_safetensors(model_path) or model_parameter_count,
        "model_path": str(model_path.resolve()),
        "device_mode": device_mode,
        "use_4bit": use_4bit,
        "max_pixels": max_pixels,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(ordered, output_dir / "predictions.jsonl")
    if split == "val":
        metrics, sample_results = evaluate_public_vqa_rows(ordered, dataset_name="docvqa")
        exact = [accuracy_score(row["pred_answer"], row["gold_answers"]) for row in ordered]
        metrics.update(common)
        metrics["accuracy"] = sum(exact) / max(1, len(exact))
        write_public_vqa_report(metrics, sample_results, output_dir, prefix="public_vqa")
    else:
        metrics = common
        submission = [
            {
                "questionId": int(row["question_id"]) if str(row["question_id"]).isdigit() else row["question_id"],
                "answer": row["pred_answer"],
            }
            for row in ordered
        ]
        (output_dir / "submission.json").write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on official DocVQA val or export test predictions.")
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "docvqa" / "prepared_v1" / "manifest.jsonl")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--seed", type=int, default=42, help="Reserved for reproducible generation settings.")
    parser.add_argument("--model-path", type=Path, default=ROOT.parent / "models" / "Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "docvqa_qwen25_vl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-mode", choices=["auto", "balanced", "balanced_low_0", "sequential", "cpu"], default="balanced_low_0")
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-memory", default="5GiB")
    parser.add_argument("--cpu-memory", default="24GiB")
    parser.add_argument("--offload-folder", type=Path, default=ROOT / "runs" / "docvqa_qwen25_vl_offload")
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--flush-every", type=int, default=10)
    args = parser.parse_args()
    run_docvqa_vlm_config(args)


def run_docvqa_vlm_config(args: argparse.Namespace) -> dict[str, Any]:

    all_rows = [row for row in load_jsonl(args.manifest) if row.get("split") == args.split]
    target_rows = all_rows[: args.max_samples] if args.max_samples is not None else all_rows
    output_dir = args.output_dir / args.split
    state_path = output_dir / "all_predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    predictions = load_jsonl(state_path) if args.resume else []
    predictions = ordered_predictions(predictions, all_rows)
    previous_wall = 0.0
    if args.resume and metrics_path.exists():
        try:
            previous_wall = float(json.loads(metrics_path.read_text(encoding="utf-8")).get("wall_time_seconds", 0.0))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    target_ids = {str(row["sample_id"]) for row in target_rows}
    predictions = [row for row in predictions if str(row["sample_id"]) in target_ids]
    if len(predictions) >= len(target_rows):
        metrics = save_outputs(predictions, target_rows, output_dir, args.split, time.perf_counter(), previous_wall, args.model_path, None, args.device_mode, args.use_4bit, args.max_pixels, len(all_rows))
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return metrics

    from vqa_retrieval.vlm_service import QwenVlmRunner

    max_memory = {0: args.gpu_memory, "cpu": args.cpu_memory} if args.device_mode != "cpu" else {"cpu": args.cpu_memory}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with OutputLock(output_dir):
        runner = QwenVlmRunner(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            use_4bit=args.use_4bit,
            device_mode=args.device_mode,
            max_memory=max_memory,
            offload_folder=args.offload_folder,
            low_cpu_mem_usage=True,
            max_pixels=args.max_pixels,
        )
        model_parameter_count = sum(parameter.numel() for parameter in runner.model.parameters())
        done = {str(row["sample_id"]) for row in predictions}
        since_flush = 0
        for row in target_rows:
            if str(row["sample_id"]) in done:
                continue
            raw = runner.generate(row["image_path"], prompt(row), max_new_tokens=96)
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "question_id": row["question_id"],
                    "image_id": row["image_id"],
                    "question": row["question"],
                    "pred_answer": parse_answer(raw),
                    "raw_response": raw,
                    "gold_answers": row.get("answers", []),
                }
            )
            done.add(str(row["sample_id"]))
            since_flush += 1
            if since_flush >= args.flush_every:
                write_jsonl(ordered_predictions(predictions, target_rows), state_path)
                save_outputs(predictions, target_rows, output_dir, args.split, started, previous_wall, args.model_path, model_parameter_count, args.device_mode, args.use_4bit, args.max_pixels, len(all_rows))
                since_flush = 0
        write_jsonl(ordered_predictions(predictions, target_rows), state_path)
        metrics = save_outputs(predictions, target_rows, output_dir, args.split, started, previous_wall, args.model_path, model_parameter_count, args.device_mode, args.use_4bit, args.max_pixels, len(all_rows))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    main()
