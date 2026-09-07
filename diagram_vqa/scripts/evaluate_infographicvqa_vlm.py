from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - allows --help outside the ML environment.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_METRICS_SPEC = importlib.util.spec_from_file_location(
    "public_vqa_metrics",
    ROOT / "src" / "vqa_retrieval" / "public_vqa_metrics.py",
)
if _METRICS_SPEC is None or _METRICS_SPEC.loader is None:
    raise ImportError("Cannot load public_vqa_metrics.py")
_METRICS_MODULE = importlib.util.module_from_spec(_METRICS_SPEC)
sys.modules[_METRICS_SPEC.name] = _METRICS_MODULE
_METRICS_SPEC.loader.exec_module(_METRICS_MODULE)
accuracy_score = _METRICS_MODULE.accuracy_score
evaluate_public_vqa_rows = _METRICS_MODULE.evaluate_public_vqa_rows
write_public_vqa_report = _METRICS_MODULE.write_public_vqa_report


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class OutputDirLock:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / ".evaluate_infographicvqa_vlm.lock"
        self.fd: int | None = None

    def __enter__(self) -> "OutputDirLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            self.fd = os.open(self.path, flags)
        except FileExistsError as exc:
            try:
                payload = self.path.read_text(encoding="utf-8")
            except Exception:
                payload = ""
            raise RuntimeError(
                f"Another InfographicVQA VLM evaluator is already using {self.path.parent}. "
                f"Lock file: {self.path}. {payload}"
            ) from exc
        os.write(self.fd, f"pid={os.getpid()}\nstarted={time.time()}\n".encode("utf-8"))
        atexit.register(self.cleanup)
        return self

    def cleanup(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.cleanup()


def current_ram_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / 1024 / 1024)
    except Exception:
        return None


def current_peak_vram_mb() -> float:
    if torch is None or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / 1024 / 1024)


def safetensors_parameter_count(model_path: Path) -> int | None:
    try:
        from safetensors import safe_open
    except Exception:
        return None

    total = 0
    for tensor_path in Path(model_path).glob("*.safetensors"):
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                total += math.prod(handle.get_slice(key).get_shape())
    return int(total) if total else None


def build_infographicvqa_prompt(row: dict[str, Any]) -> str:
    return f"""You are answering an InfographicVQA question. Return STRICT JSON only.
Schema: {{"answer": "short answer in English", "confidence": 0.0}}
Rules:
- answer must be concise.
- copy numbers, units, names, and labels exactly when visible.
- do not explain outside JSON.

Question: {row['question']}"""


def parse_vlm_answer(raw: str) -> str:
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
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            if isinstance(payload, dict):
                return str(payload.get("answer", "")).strip()
        except json.JSONDecodeError:
            pass
    return text.splitlines()[0].strip() if text else ""


def refresh_parsed_answers(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refreshed = []
    for row in predictions:
        item = dict(row)
        if item.get("raw_response") is not None:
            parsed = parse_vlm_answer(str(item.get("raw_response", "")))
            if parsed:
                item["pred_answer"] = parsed
        refreshed.append(item)
    return refreshed


def ordered_unique_predictions(predictions: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row.get("sample_id")): row for row in predictions}
    ordered = []
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if sample_id in by_id:
            ordered.append(by_id[sample_id])
    return ordered


def load_previous_predictions(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    predictions = []
    decoder = json.JSONDecoder(strict=False)
    cursor = 0
    while cursor < len(text):
        while cursor < len(text):
            if text[cursor].isspace():
                cursor += 1
                continue
            if text.startswith("\\n", cursor) or text.startswith("\\r", cursor):
                cursor += 2
                continue
            break
        if cursor >= len(text):
            break
        try:
            payload, end = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError:
            line_end = text.find("\n", cursor)
            if line_end < 0:
                line_end = len(text)
            payload = json.loads(text[cursor:line_end], strict=False)
            end = line_end
        if isinstance(payload, dict):
            predictions.append(payload)
        cursor = end
    return ordered_unique_predictions(predictions, rows)


def compute_and_write_metrics(
    predictions: list[dict[str, Any]],
    output_dir: Path,
    *,
    split: str,
    model_path: Path,
    adapter_path: Path | None,
    max_samples: int | None,
    total_eval_rows: int,
    wall_time_seconds: float,
    parameter_count: int | None,
    device_mode: str,
    use_4bit: bool,
    max_memory: dict[Any, str] | None,
    offload_folder: Path | None,
    status: str,
    max_pixels: int | None,
    previous_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous_metrics = previous_metrics or {}
    metrics, public_results = evaluate_public_vqa_rows(predictions, dataset_name="infographicvqa")
    exact_scores = [accuracy_score(row["pred_answer"], row["gold_answers"]) for row in predictions]
    current_peak_ram = current_ram_mb()
    previous_peak_ram = previous_metrics.get("peak_ram_mb")
    try:
        peak_ram_mb = max(float(current_peak_ram or 0.0), float(previous_peak_ram or 0.0))
    except (TypeError, ValueError):
        peak_ram_mb = current_peak_ram or previous_peak_ram
    metrics.update(
        {
            "mode": "qwen25_vl_open_answer",
            "split": split,
            "status": status,
            "accuracy": sum(exact_scores) / max(1, len(exact_scores)),
            "wall_time_seconds": wall_time_seconds,
            "latency_ms_per_question": wall_time_seconds * 1000 / max(1, len(predictions)),
            "throughput_questions_per_second": len(predictions) / wall_time_seconds if wall_time_seconds > 0 else None,
            "parameter_count": safetensors_parameter_count(model_path)
            or parameter_count
            or previous_metrics.get("parameter_count"),
            "loaded_parameter_count": parameter_count or previous_metrics.get("loaded_parameter_count"),
            "peak_ram_mb": peak_ram_mb,
            "peak_vram_mb": max(current_peak_vram_mb(), float(previous_metrics.get("peak_vram_mb") or 0.0)),
            "model_path": str(model_path),
            "adapter_path": str(adapter_path) if adapter_path is not None else None,
            "device_mode": device_mode,
            "use_4bit": use_4bit,
            "max_memory": max_memory,
            "offload_folder": str(offload_folder) if offload_folder is not None else None,
            "max_pixels": max_pixels,
            "max_samples": max_samples,
            "total_eval_rows": total_eval_rows,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions, output_dir / "predictions.jsonl")
    if split == "test":
        submission = [
            {"questionId": row.get("question_id", row.get("sample_id")), "answer": row.get("pred_answer", "")}
            for row in predictions
        ]
        (output_dir / "submission.json").write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_public_vqa_report(metrics, public_results, output_dir, prefix="public_vqa")
    return metrics


def run_infographicvqa_vlm(
    *,
    manifest: Path,
    split: str,
    model_path: Path,
    adapter_path: Path | None,
    output_dir: Path,
    max_samples: int | None,
    use_4bit: bool,
    device_mode: str,
    max_memory: dict[Any, str] | None,
    offload_folder: Path | None,
    resume: bool,
    max_pixels: int | None,
) -> dict[str, Any]:
    all_rows = [row for row in load_jsonl(manifest) if row.get("split") == split and (split == "test" or row.get("answers"))]
    rows = all_rows[:max_samples] if max_samples is not None else all_rows
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    state_predictions_path = output_dir / "all_predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    previous_metrics = {}
    if metrics_path.exists():
        try:
            previous_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            previous_metrics = {}

    resume_path = state_predictions_path if state_predictions_path.exists() else predictions_path
    predictions = load_previous_predictions(resume_path, all_rows) if resume else []
    predictions = refresh_parsed_answers(predictions)
    previous_wall = 0.0
    if resume and previous_metrics and predictions:
        try:
            previous_wall = float(previous_metrics.get("wall_time_seconds", 0.0))
        except Exception:
            previous_wall = 0.0

    target_ids = {str(row.get("sample_id")) for row in rows}
    target_predictions = [row for row in predictions if str(row.get("sample_id")) in target_ids]

    if len(target_predictions) >= len(rows):
        status = "available_full" if max_samples is None and len(target_predictions) == len(all_rows) else "available_partial"
        return compute_and_write_metrics(
            target_predictions,
            output_dir,
            split=split,
            model_path=model_path,
            adapter_path=adapter_path,
            max_samples=max_samples,
            total_eval_rows=len(all_rows),
            wall_time_seconds=previous_wall,
            parameter_count=None,
            device_mode=device_mode,
            use_4bit=use_4bit,
            max_memory=max_memory,
            offload_folder=offload_folder,
            status=status,
            max_pixels=max_pixels,
            previous_metrics=previous_metrics,
        )

    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    from vqa_retrieval.vlm_service import QwenVlmRunner

    started = time.perf_counter()
    runner = QwenVlmRunner(
        model_path=model_path,
        adapter_path=adapter_path,
        use_4bit=use_4bit,
        device_mode=device_mode,
        max_memory=max_memory,
        offload_folder=offload_folder,
        low_cpu_mem_usage=True,
        max_pixels=max_pixels,
    )
    parameter_count = int(sum(parameter.numel() for parameter in runner.model.parameters()))

    done_ids = {str(row["sample_id"]) for row in predictions}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in done_ids:
            continue
        raw = runner.generate(row["image_path"], build_infographicvqa_prompt(row), max_new_tokens=96)
        prediction = {
            "sample_id": row["sample_id"],
            "image_id": row["image_id"],
            "question": row["question"],
            "pred_answer": parse_vlm_answer(raw),
            "raw_response": raw,
            "gold_answers": row["answers"],
            "answer_type": row.get("answer_type", []),
            "evidence": row.get("evidence", []),
            "operation_reasoning": row.get("operation_reasoning", []),
        }
        predictions.append(prediction)
        write_jsonl(ordered_unique_predictions(predictions, all_rows), state_predictions_path)
        done_ids.add(sample_id)
        target_predictions = [item for item in predictions if str(item.get("sample_id")) in target_ids]
        status = "available_full" if max_samples is None and len(target_predictions) == len(all_rows) else "available_partial"
        compute_and_write_metrics(
            target_predictions,
            output_dir,
            split=split,
            model_path=model_path,
            adapter_path=adapter_path,
            max_samples=max_samples,
            total_eval_rows=len(all_rows),
            wall_time_seconds=previous_wall + time.perf_counter() - started,
            parameter_count=parameter_count,
            device_mode=device_mode,
            use_4bit=use_4bit,
            max_memory=max_memory,
            offload_folder=offload_folder,
            status=status,
            max_pixels=max_pixels,
            previous_metrics=previous_metrics,
        )

    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL open-answer predictions on InfographicVQA.")
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "infographicvqa" / "prepared_v1" / "manifest.jsonl")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--model-path", type=Path, default=ROOT.parent / "models" / "Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "infographicvqa_qwen25_vl_open_answer_full")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device-mode", choices=["auto", "balanced", "balanced_low_0", "sequential", "cpu"], default="balanced_low_0")
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-memory", default="5GiB")
    parser.add_argument("--cpu-memory", default="24GiB")
    parser.add_argument("--offload-folder", type=Path, default=ROOT / "runs" / "infographicvqa_qwen25_vl_offload")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    args = parser.parse_args()

    max_memory = {0: args.gpu_memory, "cpu": args.cpu_memory} if args.device_mode != "cpu" else None
    with OutputDirLock(args.output_dir):
        metrics = run_infographicvqa_vlm(
            manifest=args.manifest,
            split=args.split,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            use_4bit=args.use_4bit,
            device_mode=args.device_mode,
            max_memory=max_memory,
            offload_folder=args.offload_folder,
            resume=args.resume,
            max_pixels=args.max_pixels,
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
