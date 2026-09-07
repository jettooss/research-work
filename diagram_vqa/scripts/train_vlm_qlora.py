from __future__ import annotations

import argparse
import json
import inspect
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
from qwen_vl_utils import process_vision_info  # noqa: E402


def _require_qlora_deps():
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "QLoRA training requires `peft`. Install with: "
            ".\\.venv\\Scripts\\pip install peft"
        ) from exc
    return LoraConfig, get_peft_model, prepare_model_for_kbit_training


class VlmSftJsonlDataset(Dataset):
    def __init__(self, path: str | Path, max_samples: int | None = None) -> None:
        path = Path(path)
        self.rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if max_samples is not None and max_samples > 0:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError(f"Dataset is empty: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class QwenVlmSftCollator:
    def __init__(
        self,
        processor,
        max_length: int = 2048,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        user_messages = []
        full_messages = []
        for item in batch:
            image_path = str(Path(item["image_path"]).resolve())
            prompt = str(item["prompt"])
            completion = str(item["completion"])
            image_payload: dict[str, Any] = {"type": "image", "image": image_path}
            if self.image_min_pixels is not None and self.image_min_pixels > 0:
                image_payload["min_pixels"] = int(self.image_min_pixels)
            if self.image_max_pixels is not None and self.image_max_pixels > 0:
                image_payload["max_pixels"] = int(self.image_max_pixels)
            user_msg = {
                "role": "user",
                "content": [
                    image_payload,
                    {"type": "text", "text": prompt},
                ],
            }
            assistant_msg = {
                "role": "assistant",
                "content": [{"type": "text", "text": completion}],
            }
            user_messages.append([user_msg])
            full_messages.append([user_msg, assistant_msg])

        prompt_texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in user_messages
        ]
        full_texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in full_messages
        ]

        image_inputs = []
        video_inputs = []
        for messages in user_messages:
            images, videos = process_vision_info(messages)
            if images:
                image_inputs.extend(images)
            if videos:
                video_inputs.extend(videos)

        common_kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }
        full_inputs = self.processor(
            text=full_texts,
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            **common_kwargs,
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            **common_kwargs,
        )

        labels = full_inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100
        prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)
        for row_idx, prompt_len in enumerate(prompt_lens.tolist()):
            labels[row_idx, : int(prompt_len)] = -100
        full_inputs["labels"] = labels
        return full_inputs


class ProgressJsonlCallback(TrainerCallback):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def _write(self, event: str, args, state, logs: dict[str, Any] | None = None) -> None:
        payload = {
            "event": event,
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "max_steps": int(state.max_steps),
            "progress": float(state.global_step / state.max_steps) if state.max_steps else 0.0,
            "logs": logs or {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self._write("train_begin", args, state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._write("log", args, state, logs=logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._write("evaluate", args, state, logs=metrics)

    def on_save(self, args, state, control, **kwargs):
        self._write("save", args, state)

    def on_train_end(self, args, state, control, **kwargs):
        self._write("train_end", args, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL-3B with QLoRA on SFT JSONL data.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=EXTERNAL_ROOT / "models" / "Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "vlm_qlora")

    parser.add_argument("--num-train-epochs", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--image-min-pixels", type=int, default=None)
    parser.add_argument("--image-max-pixels", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--progress-jsonl", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_qlora_config(args)


def train_qlora_config(args: argparse.Namespace) -> dict[str, Any]:
    LoraConfig, get_peft_model, prepare_model_for_kbit_training = _require_qlora_deps()

    has_cuda = torch.cuda.is_available()
    supports_bf16 = bool(
        has_cuda
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    use_bf16 = supports_bf16
    use_fp16 = has_cuda and not use_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if has_cuda else torch.float32)

    use_4bit = not args.no_4bit
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True, use_fast=False)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["torch_dtype"] = compute_dtype
    else:
        model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForImageTextToText.from_pretrained(str(args.model_path), **model_kwargs)
    if args.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    train_ds = VlmSftJsonlDataset(args.train_jsonl, max_samples=args.max_train_samples)
    val_ds = VlmSftJsonlDataset(args.val_jsonl, max_samples=args.max_val_samples)
    collator = QwenVlmSftCollator(
        processor=processor,
        max_length=args.max_length,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
    )

    train_args_kwargs = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "disable_tqdm": bool(args.disable_tqdm),
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "remove_unused_columns": False,
        "report_to": [],
        "gradient_checkpointing": bool(args.gradient_checkpointing),
    }
    init_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in init_params:
        train_args_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in init_params:
        train_args_kwargs["eval_strategy"] = "epoch"
    else:
        raise RuntimeError("TrainingArguments does not support evaluation strategy parameters")

    training_args = TrainingArguments(**train_args_kwargs)
    progress_path = args.progress_jsonl or (args.output_dir / "progress.jsonl")
    callbacks = [
        ProgressJsonlCallback(progress_path),
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
    ]
    print(f"[INFO] Trainer progress will be saved to {progress_path}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    checkpoints = sorted(
        args.output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
    trainer.save_model(str(args.output_dir / "adapter"))
    processor.save_pretrained(str(args.output_dir / "adapter"))
    metrics = trainer.evaluate(eval_dataset=val_ds)
    metrics["progress_jsonl"] = str(progress_path)
    (args.output_dir / "train_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Adapter saved to {args.output_dir / 'adapter'}")
    print(f"[INFO] Progress log saved to {progress_path}")
    return metrics


def train_qlora(
    *,
    train_jsonl: Path,
    val_jsonl: Path,
    output_dir: Path,
    epochs: int = 2,
    early_stopping_patience: int = 1,
    gradient_accumulation_steps: int = 16,
    max_samples: int | None = None,
) -> dict[str, Any]:
    args = argparse.Namespace(
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        model_path=EXTERNAL_ROOT / "models" / "Qwen2.5-VL-3B-Instruct",
        output_dir=output_dir,
        num_train_epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        learning_rate=2e-4,
        weight_decay=0.01,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.03,
        logging_steps=10,
        save_total_limit=2,
        max_length=2048,
        image_min_pixels=None,
        image_max_pixels=786432,
        max_train_samples=max_samples,
        max_val_samples=max_samples,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        gradient_checkpointing=True,
        no_4bit=False,
        disable_tqdm=False,
        progress_jsonl=output_dir / "progress.jsonl",
    )
    return train_qlora_config(args)


if __name__ == "__main__":
    main()
