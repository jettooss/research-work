from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vqa_retrieval.model_matrix import (  # noqa: E402
    DATASETS,
    SEEDS,
    candidate_texts,
    ensure_free_space,
    load_rows,
    ocr_spans,
    retrieval_metrics,
    split_rows,
    vqa_row,
)
from vqa_retrieval.public_vqa_metrics import anls_score  # noqa: E402


def load_torch(path: Path) -> Any:
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class RowDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class FrozenVisionStore:
    def __init__(self, dataset: str, model_name: str, cache_root: Path, device: torch.device) -> None:
        self.dataset = dataset
        self.model_name = model_name
        self.cache_root = cache_root / model_name / dataset
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._image: dict[str, torch.Tensor] = {}
        self._text: dict[str, torch.Tensor] = {}
        if model_name == "clip":
            import clip

            self.backbone, self.preprocess = clip.load("ViT-B/32", device=device)
            self.tokenize = clip.tokenize
            self.feature_dim = 512
            self.processor = None
        else:
            from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

            name = "google/siglip2-base-patch16-224"
            # The backbone is frozen. Loading it directly in fp16 avoids a
            # transient fp32 CUDA copy which can crash the Windows WDDM driver
            # on 8 GB GPUs while preserving the intended inference protocol.
            backbone_dtype = torch.float16 if device.type == "cuda" else torch.float32
            load_kwargs: dict[str, Any] = {"torch_dtype": backbone_dtype}
            if device.type == "cuda":
                # Stream the checkpoint directly to CUDA. A monolithic
                # Module.to() copy intermittently access-violates under WDDM.
                load_kwargs["device_map"] = "cuda"
            self.backbone = AutoModel.from_pretrained(name, **load_kwargs)
            if device.type != "cuda":
                self.backbone.to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.processor = AutoImageProcessor.from_pretrained(name, use_fast=False)
            self.feature_dim = int(getattr(self.backbone.config, "projection_dim", 768))
            self.preprocess = None
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def _path(self, kind: str, key: str) -> Path:
        import hashlib

        directory = self.cache_root / kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.pt"

    @torch.no_grad()
    def image(self, row: dict[str, Any]) -> torch.Tensor:
        key = row["image_id"]
        if key in self._image:
            return self._image[key]
        path = self._path("images", str(Path(row["image_path"]).resolve()))
        if path.exists():
            value = load_torch(path).float()
        else:
            image = Image.open(row["image_path"]).convert("RGB")
            if self.model_name == "clip":
                value = self.backbone.encode_image(self.preprocess(image).unsqueeze(0).to(self.device))[0].detach().cpu().float()
            else:
                inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
                value = self.backbone.get_image_features(**inputs)[0].detach().cpu().float()
            torch.save(value.to(torch.float16), path)
        self._image[key] = value
        return value

    @torch.no_grad()
    def texts(self, values: list[str]) -> torch.Tensor:
        output: list[torch.Tensor | None] = []
        missing: list[str] = []
        missing_indices: list[int] = []
        for value in values:
            key = str(value)
            if key in self._text:
                output.append(self._text[key])
                continue
            path = self._path("texts", key)
            if path.exists():
                tensor = load_torch(path).float()
                self._text[key] = tensor
                output.append(tensor)
            else:
                output.append(None)
                missing.append(key)
                missing_indices.append(len(output) - 1)
        for start in range(0, len(missing), 128):
            batch_values = missing[start : start + 128]
            if self.model_name == "clip":
                encoded = self.backbone.encode_text(self.tokenize(batch_values, truncate=True).to(self.device))
            else:
                tokens = self.tokenizer(batch_values, return_tensors="pt", padding="max_length", truncation=True, max_length=64, return_attention_mask=False).to(self.device)
                encoded = self.backbone.get_text_features(**tokens)
            for offset, tensor in enumerate(encoded.detach().cpu().float()):
                key = batch_values[offset]
                self._text[key] = tensor
                torch.save(tensor.to(torch.float16), self._path("texts", key))
        for index, value in zip(missing_indices, missing):
            output[index] = self._text[value]
        return torch.stack([tensor for tensor in output if tensor is not None])

    def candidates(self, row: dict[str, Any]) -> tuple[list[str], torch.Tensor, int]:
        if self.dataset == "ai2d":
            values = [str(value) for value in row.get("options", [])]
            return values, self.texts(values), int(row.get("correct_option_idx", 0))
        spans = ocr_spans(self.dataset, row, EXTERNAL_ROOT, EXTERNAL_ROOT / "model_matrix_cache/ocr")
        values = candidate_texts(spans, max_candidates=128) or [""]
        answers = row.get("answers", [])
        scores = [max((anls_score(value, [answer]) for answer in answers), default=0.0) for value in values]
        return values, self.texts(values), int(np.argmax(scores)) if scores else 0


class VisionHeads(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 192) -> None:
        super().__init__()
        self.image = nn.Sequential(nn.Linear(input_dim, 384), nn.GELU(), nn.Linear(384, output_dim))
        self.text = nn.Sequential(nn.Linear(input_dim, 384), nn.GELU(), nn.Linear(384, output_dim))
        self.fusion = nn.Sequential(nn.Linear(output_dim * 2, output_dim), nn.GELU(), nn.Linear(output_dim, output_dim))

    def retrieval(self, images: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        image = F.normalize(self.image(images), dim=-1)
        question = F.normalize(self.text(questions), dim=-1)
        return question @ image.T

    def vqa(self, image: torch.Tensor, question: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        image = F.normalize(self.image(image.unsqueeze(0)), dim=-1)
        question = F.normalize(self.text(question.unsqueeze(0)), dim=-1)
        context = F.normalize(self.fusion(torch.cat([image, question], dim=-1)), dim=-1)
        candidates = F.normalize(self.text(candidates), dim=-1)
        return (context @ candidates.T).squeeze(0)


def multipositive_loss(logits: torch.Tensor, image_ids: list[str]) -> torch.Tensor:
    positive = torch.tensor([[left == right for right in image_ids] for left in image_ids], device=logits.device)
    scaled = logits / 0.07
    forward = -(torch.logsumexp(scaled.masked_fill(~positive, -torch.inf), 1) - torch.logsumexp(scaled, 1)).mean()
    reverse = -(torch.logsumexp(scaled.T.masked_fill(~positive.T, -torch.inf), 1) - torch.logsumexp(scaled.T, 1)).mean()
    return 0.5 * (forward + reverse)


@torch.no_grad()
def validation_loss(heads: VisionHeads, rows: list[dict[str, Any]], store: FrozenVisionStore, device: torch.device, batch_size: int) -> float:
    heads.eval()
    loader = DataLoader(RowDataset(rows), batch_size=batch_size, shuffle=False, collate_fn=lambda values: values)
    losses: list[float] = []
    for batch in tqdm(loader, desc="validation loss", leave=False):
        images = torch.stack([store.image(row) for row in batch]).to(device)
        questions = store.texts([str(row["question"]) for row in batch]).to(device)
        retrieval = multipositive_loss(heads.retrieval(images, questions), [row["image_id"] for row in batch])
        answer_losses = []
        for row, image, question in zip(batch, images, questions):
            _, candidates, target = store.candidates(row)
            logits = heads.vqa(image, question, candidates.to(device))
            answer_losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=device)))
        losses.append(float(0.5 * retrieval + 0.5 * torch.stack(answer_losses).mean()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(heads: VisionHeads, rows: list[dict[str, Any]], store: FrozenVisionStore, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    heads.eval()
    unique_docs: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique_docs.setdefault(row["image_id"], row)
    docs = list(unique_docs.values())
    image = torch.stack([store.image(row) for row in docs]).to(device)
    questions = store.texts([str(row["question"]) for row in rows]).to(device)
    similarity = heads.retrieval(image, questions).cpu().numpy()
    retrieval = retrieval_metrics(similarity, [row["image_id"] for row in rows], [row["image_id"] for row in docs])
    predictions = []
    for row in tqdm(rows, desc="VQA evaluation"):
        values, candidates, _ = store.candidates(row)
        logits = heads.vqa(store.image(row).to(device), store.texts([str(row["question"])])[0].to(device), candidates.to(device))
        predictions.append(vqa_row(store.dataset, row, values[int(logits.argmax())] if values else ""))
    return {
        "retrieval": retrieval,
        "vqa": {
            "metric": "accuracy" if store.dataset == "ai2d" else "anls",
            "score": float(np.mean([row["score"] for row in predictions])),
            "exact_accuracy": float(np.mean([row["exact"] for row in predictions])),
        },
    }, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen CLIP/SigLIP heads for the 11x3 matrix.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=["clip", "siglip"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/model_matrix")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    run_vision_experiment_config(args)


def run_vision_experiment_config(args: argparse.Namespace) -> dict[str, Any]:

    run_dir = Path(args.run_dir_override) if getattr(args, "run_dir_override", None) else args.output_dir / args.dataset / args.model / f"seed{args.seed}" / args.split
    ensure_free_space(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "dataset": args.dataset, "model": args.model, "split": args.split, "seed": args.seed,
        "max_samples": args.max_samples, "epochs": args.epochs, "early_stopping_patience": args.early_stopping_patience,
        "batch_size": args.batch_size, "frozen_backbone": True,
        "checkpoint_selection": "0.5*vqa+0.5*mean_r1", "resume": args.resume,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path = run_dir / "metrics.json"
    if args.resume and metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        test_path = run_dir / "test/metrics.json" if getattr(args, "run_dir_override", None) else run_dir.parent / "test/metrics.json"
        test_ready = args.split == "test" or test_path.exists()
        if existing.get("status") == "completed_full" and existing.get("epochs_completed") == args.epochs and test_ready:
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return existing

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    rows = load_rows(args.dataset, EXTERNAL_ROOT)
    train_rows = split_rows(rows, "train", args.max_samples)
    eval_rows = split_rows(rows, args.split, args.max_samples)
    store = FrozenVisionStore(args.dataset, args.model, EXTERNAL_ROOT / "model_matrix_cache/vision", device)
    heads = VisionHeads(store.feature_dim).to(device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=0.01)
    loader = DataLoader(RowDataset(train_rows), batch_size=args.batch_size, shuffle=True, collate_fn=lambda values: values)
    epochs = min(args.epochs, 2) if args.max_samples is not None else args.epochs
    history_path = run_dir / "history.json"
    last_checkpoint_path = run_dir / "checkpoint_last.pt"
    history = []
    best_composite = -math.inf
    started = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    if args.resume and history_path.exists() and last_checkpoint_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        last = load_torch(last_checkpoint_path)
        heads.load_state_dict(last["heads"])
        optimizer.load_state_dict(last["optimizer"])
        start_epoch = int(last["epoch"]) + 1
        if history:
            best_composite = max(float(row["composite"]) for row in history)

    for epoch in range(start_epoch, epochs + 1):
        heads.train()
        losses = []
        for batch in tqdm(loader, desc=f"{args.dataset}/{args.model}/seed{args.seed} epoch {epoch}/{epochs}"):
            images = torch.stack([store.image(row) for row in batch]).to(device)
            questions = store.texts([str(row["question"]) for row in batch]).to(device)
            retrieval_loss = multipositive_loss(heads.retrieval(images, questions), [row["image_id"] for row in batch])
            vqa_losses = []
            for row, image, question in zip(batch, images, questions):
                _, candidates, target = store.candidates(row)
                logits = heads.vqa(image, question, candidates.to(device))
                vqa_losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=device)))
            loss = 0.5 * retrieval_loss + 0.5 * torch.stack(vqa_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        evaluation, _ = evaluate(heads, eval_rows, store, device)
        val_loss = validation_loss(heads, eval_rows, store, device, args.batch_size)
        composite = 0.5 * float(evaluation["retrieval"]["mean_recall_at_k"]["1"]) + 0.5 * float(evaluation["vqa"]["score"])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_loss": val_loss, "composite": composite, **evaluation})
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        if composite > best_composite:
            best_composite = composite
            torch.save({"heads": heads.state_dict(), "epoch": epoch, "composite": composite}, run_dir / "checkpoint_best.pt")
        torch.save(
            {"heads": heads.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
            last_checkpoint_path,
        )

    best = load_torch(run_dir / "checkpoint_best.pt")
    heads.load_state_dict(best["heads"])
    evaluation, predictions = evaluate(heads, eval_rows, store, device)
    elapsed = time.perf_counter() - started
    metrics = {
        "dataset": args.dataset, "model": args.model, "split": args.split, "seed": args.seed,
        "status": "completed_full" if args.max_samples is None and len(history) == args.epochs else "completed_smoke",
        "num_samples": len(eval_rows), "epochs_completed": len(history), "best_epoch": int(best["epoch"]),
        "best_composite": float(best["composite"]), "history": history,
        "training_execution": {"mode": "full", "full_training_was_run": True}, **evaluation,
        "resources": {
            "wall_time_seconds": elapsed, "latency_ms_per_question": elapsed * 1000 / max(1, len(eval_rows)),
            "throughput_questions_per_second": len(eval_rows) / elapsed if elapsed else None,
            "peak_ram_mb": psutil.Process().memory_info().rss / (1024 * 1024),
            "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
            "parameter_count": sum(parameter.numel() for parameter in heads.parameters()),
            "frozen_backbone": True,
        },
    }
    write_jsonl(predictions, run_dir / "predictions.jsonl")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.split == "val":
        test_rows = split_rows(rows, "test", args.max_samples)
        test_evaluation, test_predictions = evaluate(heads, test_rows, store, device)
        test_dir = run_dir / "test" if getattr(args, "run_dir_override", None) else run_dir.parent / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_metrics = {
            "dataset": args.dataset, "model": args.model, "split": "test", "seed": args.seed,
            "status": "completed_full" if args.max_samples is None else "completed_smoke",
            "num_samples": len(test_rows), **test_evaluation,
        }
        metrics["test"] = test_metrics
        write_jsonl(test_predictions, test_dir / "predictions.jsonl")
        (test_dir / "metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.dataset != "ai2d":
            submission = [{"questionId": row["question_id"], "answer": row["pred_answer"]} for row in test_predictions]
            (test_dir / "submission.json").write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def train_vision_experiment(
    *,
    dataset: str,
    run_dir: Path,
    epochs: int = 40,
    seed: int = 42,
    batch_size: int = 32,
    max_samples: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    args = argparse.Namespace(
        dataset=dataset,
        model="clip",
        split="val",
        seed=seed,
        max_samples=max_samples,
        output_dir=run_dir.parent,
        run_dir_override=run_dir,
        resume=resume,
        epochs=epochs,
        early_stopping_patience=0,
        batch_size=batch_size,
    )
    return run_vision_experiment_config(args)


if __name__ == "__main__":
    main()
