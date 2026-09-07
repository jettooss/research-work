from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
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
    ensure_free_space,
    load_rows,
    retrieval_metrics,
    split_rows,
)


class ClipRows(Dataset):
    def __init__(self, rows: list[dict[str, Any]], preprocess: Any) -> None:
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    @lru_cache(maxsize=256)
    def _image(self, path: str) -> torch.Tensor:
        with Image.open(path) as image:
            return self.preprocess(image.convert("RGB"))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "image": self._image(str(row["image_path"])),
            "question": str(row["question"]),
            "image_id": str(row["image_id"]),
        }


def collate(values: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([value["image"] for value in values]),
        "questions": [value["question"] for value in values],
        "image_ids": [value["image_id"] for value in values],
    }


def multipositive_loss(logits: torch.Tensor, image_ids: list[str]) -> torch.Tensor:
    positive = torch.tensor(
        [[left == right for right in image_ids] for left in image_ids],
        dtype=torch.bool,
        device=logits.device,
    )
    forward = -(torch.logsumexp(logits.masked_fill(~positive, -torch.inf), 1) - torch.logsumexp(logits, 1)).mean()
    reverse = -(torch.logsumexp(logits.T.masked_fill(~positive.T, -torch.inf), 1) - torch.logsumexp(logits.T, 1)).mean()
    return 0.5 * (forward + reverse)


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if name in names}


def set_trainable(model: torch.nn.Module, mode: str, blocks: int) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(mode == "full")
    if mode == "last-blocks":
        modules = [
            *list(model.visual.transformer.resblocks[-blocks:]),
            model.visual.ln_post,
            *list(model.transformer.resblocks[-blocks:]),
            model.ln_final,
        ]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        model.visual.proj.requires_grad_(True)
        model.text_projection.requires_grad_(True)
        model.logit_scale.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No CLIP parameters were selected for fine-tuning")
    return parameters


def encode_rows(
    model: torch.nn.Module,
    preprocess: Any,
    tokenize: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    model.eval()
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(str(row["image_id"]), row)
    documents = list(unique.values())
    image_features: list[torch.Tensor] = []
    text_features: list[torch.Tensor] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(documents), batch_size), desc="encode images", leave=False):
            images = torch.stack([
                preprocess(Image.open(row["image_path"]).convert("RGB"))
                for row in documents[start : start + batch_size]
            ]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                encoded = F.normalize(model.encode_image(images), dim=-1)
            image_features.append(encoded.float().cpu())
        for start in tqdm(range(0, len(rows), batch_size * 2), desc="encode questions", leave=False):
            tokens = tokenize(
                [str(row["question"]) for row in rows[start : start + batch_size * 2]],
                truncate=True,
            ).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                encoded = F.normalize(model.encode_text(tokens), dim=-1)
            text_features.append(encoded.float().cpu())
    return (
        torch.cat(image_features).numpy(),
        [str(row["image_id"]) for row in documents],
        torch.cat(text_features).numpy(),
        [str(row["image_id"]) for row in rows],
    )


def evaluate(
    model: torch.nn.Module,
    preprocess: Any,
    tokenize: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    images, document_ids, questions, question_ids = encode_rows(
        model, preprocess, tokenize, rows, device, batch_size
    )
    similarity = questions @ images.T
    return retrieval_metrics(similarity, question_ids, document_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP itself for diagram retrieval")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--mode", choices=["last-blocks", "full"], default="last-blocks")
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/clip_finetune")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    import clip

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.output_dir / args.dataset / args.mode / f"seed{args.seed}"
    ensure_free_space(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"output_dir": str(args.output_dir), "device": str(device), "backbone": "CLIP ViT-B/32"}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = load_rows(args.dataset, EXTERNAL_ROOT)
    train_rows = split_rows(rows, "train", args.max_samples)
    val_rows = split_rows(rows, "val", args.max_samples)
    test_rows = split_rows(rows, "test", args.max_samples)
    model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
    model.float().to(device)
    parameters = set_trainable(model, args.mode, args.unfreeze_blocks)
    trainable = sum(parameter.numel() for parameter in parameters)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device} mode={args.mode} trainable={trainable:,}/{total:,}")

    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    dataset = ClipRows(train_rows, preprocess)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    last_path = run_dir / "checkpoint_last.pt"
    best_path = run_dir / "checkpoint_best.pt"
    history_path = run_dir / "history.json"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_r1 = -math.inf
    stale = 0
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"]) + 1
        if history:
            best_r1 = max(float(row["mean_r1"]) for row in history)

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for step, batch in enumerate(tqdm(loader, desc=f"{args.dataset} CLIP epoch {epoch}/{args.epochs}"), start=1):
            images = batch["images"].to(device, non_blocking=True)
            tokens = clip.tokenize(batch["questions"], truncate=True).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                image_features = F.normalize(model.encode_image(images), dim=-1)
                text_features = F.normalize(model.encode_text(tokens), dim=-1)
                logits = model.logit_scale.exp().clamp(max=100) * (text_features @ image_features.T)
                loss = multipositive_loss(logits, batch["image_ids"]) / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()) * args.grad_accum)

        val = evaluate(model, preprocess, clip.tokenize, val_rows, device, args.eval_batch_size)
        mean_r1 = float(val["mean_recall_at_k"]["1"])
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "mean_r1": mean_r1, "retrieval": val}
        history.append(entry)
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        state = trainable_state(model)
        torch.save({"model": state, "optimizer": optimizer.state_dict(), "epoch": epoch, "history": history}, last_path)
        if mean_r1 > best_r1:
            best_r1 = mean_r1
            stale = 0
            torch.save({"model": state, "epoch": epoch, "mean_r1": mean_r1}, best_path)
        else:
            stale += 1
        print(json.dumps(entry, ensure_ascii=False))
        if args.early_stopping_patience and stale >= args.early_stopping_patience:
            print(f"early stopping after epoch {epoch}")
            break

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"], strict=False)
    test = evaluate(model, preprocess, clip.tokenize, test_rows, device, args.eval_batch_size)
    elapsed = time.perf_counter() - started
    metrics = {
        "schema_version": 1,
        "dataset": args.dataset,
        "model": "clip_finetuned",
        "backbone": "CLIP ViT-B/32",
        "fine_tuning_mode": args.mode,
        "unfreeze_blocks": args.unfreeze_blocks if args.mode == "last-blocks" else None,
        "seed": args.seed,
        "status": "completed_full" if args.max_samples is None else "completed_smoke",
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": int(best["epoch"]),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "test": {"split": "test", "num_samples": len(test_rows), "retrieval": test},
        "history": history,
        "resources": {
            "wall_time_seconds": elapsed,
            "peak_ram_mb": psutil.Process().memory_info().rss / (1024 * 1024),
            "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
