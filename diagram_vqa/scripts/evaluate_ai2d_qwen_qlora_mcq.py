from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vqa_retrieval.vlm_service import QwenVlmRunner  # noqa: E402


def _normalize(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def _parse_answer(raw: str) -> str:
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        payload = json.loads(raw[start:end])
        return str(payload.get("answer", "")).strip()
    except (ValueError, json.JSONDecodeError):
        match = re.search(r'["\']answer["\']\s*:\s*["\']([^"\']+)', raw, re.I)
        return match.group(1).strip() if match else raw.strip()


def _resolve_option_index(answer: str, options: list[str]) -> int:
    normalized_answer = _normalize(answer)
    exact = next(
        (i for i, option in enumerate(options) if _normalize(option) == normalized_answer),
        None,
    )
    if exact is not None:
        return exact

    numeric = re.match(r"^\s*(\d+)\s*(?:[:.)-]|$)", answer)
    if numeric and 0 <= int(numeric.group(1)) < len(options):
        return int(numeric.group(1))

    labeled = re.match(r"^\s*([A-Z])\s*(?:[:.)-]|$)", answer, re.I)
    if labeled:
        index = ord(labeled.group(1).upper()) - ord("A")
        if 0 <= index < len(options):
            return index

    if ":" in answer:
        answer_text = _normalize(answer.split(":", maxsplit=1)[1])
        return next(
            (i for i, option in enumerate(options) if _normalize(option) == answer_text),
            -1,
        )
    return -1


def _compatible_adapter_copy(source: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("adapter_config.json", "README.md"):
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, target / filename)

    tensors = load_file(source / "adapter_model.safetensors", device="cpu")
    remapped = {}
    for key, value in tensors.items():
        new_key = key.replace(
            "base_model.model.model.language_model.",
            "base_model.model.model.",
        ).replace(
            "base_model.model.model.visual.",
            "base_model.model.visual.",
        )
        remapped[new_key] = value
    save_file(remapped, target / "adapter_model.safetensors")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EXTERNAL_ROOT / "ai2d/model_matrix_v1/manifest.jsonl",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=EXTERNAL_ROOT / "models/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--adapter-path", type=Path, default=Path("runs/vlm_qlora_mcq/adapter")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/ai2d_top_models_100_seed_analysis/qwen25_vl_qlora"),
    )
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("split") == "test"][: args.max_samples]
    compatible_adapter = _compatible_adapter_copy(
        args.adapter_path,
        args.output_dir / "adapter_compat",
    )
    runner = QwenVlmRunner(
        model_path=args.model_path,
        adapter_path=compatible_adapter,
        use_4bit=True,
        device_mode="auto",
    )

    predictions: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        options = [str(option) for option in row["options"]]
        option_block = "\n".join(f"{i}: {option}" for i, option in enumerate(options))
        prompt = (
            "Read the diagram and answer the multiple-choice question. "
            'Return JSON only: {"answer": "exact option text"}.\n'
            f"Question: {row['question']}\nOptions:\n{option_block}"
        )
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = EXTERNAL_ROOT / image_path
        raw = runner.generate(image_path, prompt, max_new_tokens=args.max_new_tokens)
        answer = _parse_answer(raw)
        predicted_index = _resolve_option_index(answer, options)
        gold_index = int(row["correct_option_idx"])
        correct = predicted_index == gold_index
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "image_id": str(row["image_id"]),
                "question": row["question"],
                "options": options,
                "gold_option_idx": gold_index,
                "gold_option_text": options[gold_index],
                "pred_option_idx": predicted_index,
                "pred_option_text": answer,
                "is_correct": correct,
                "raw_response": raw,
                "prompt_style": "training_aligned_exact_option_text",
            }
        )
        print(
            f"[{index}/{len(rows)}] {row['sample_id']} pred={predicted_index} "
            f"gold={gold_index} correct={correct}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    accuracy = sum(bool(row["is_correct"]) for row in predictions) / max(
        1, len(predictions)
    )
    metrics = {
        "model": "Qwen2.5-VL-3B QLoRA MCQ",
        "samples": len(predictions),
        "accuracy": accuracy,
        "prompt_style": "training_aligned_exact_option_text",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
