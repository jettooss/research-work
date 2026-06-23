from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from vqa_retrieval.ai2d_hybrid import (  # noqa: E402
    Ai2dHybridDataset,
    compute_option_logits,
    contrastive_loss,
    load_manifest_hybrid,
    load_split_payload,
    make_hybrid_collate_fn,
    retrieval_metrics_from_embeddings,
    resolve_sample_file_paths,
    select_samples_for_split,
    vqa_accuracy_from_logits,
)
from vqa_retrieval.graph_builder_v2 import FeatureCacheV2, GraphEncoderV2, NodeFeaturizerV2  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_limit_samples(
    samples: list,
    limit: Optional[int],
    seed: int,
) -> list:
    if limit is None or limit <= 0 or len(samples) <= limit:
        return samples
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    idx = sorted(idx[:limit])
    return [samples[i] for i in idx]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_cache_signature(args, featurizer: NodeFeaturizerV2) -> str:
    return FeatureCacheV2.make_signature(
        vision_model_name=featurizer.vision_model_name,
        text_model_name=featurizer.text_model_name,
        ocr_level=args.ocr_level,
        min_area=args.extract_min_area,
        max_shape_nodes=args.extract_max_nodes,
        min_text_conf=args.extract_min_text_conf,
        knn_k=args.extract_knn_k,
        include_shapes=not args.disable_shape_nodes,
    )


def encode_question_embeddings(batch, featurizer: NodeFeaturizerV2, cache: FeatureCacheV2, device: torch.device):
    q_emb = cache.get_text_batch(batch["question_texts"], featurizer.text_enc, normalize=False)
    return q_emb.to(device)


def build_graph_batch(
    batch: dict[str, Any],
    cache: FeatureCacheV2,
    featurizer: NodeFeaturizerV2,
    args,
    device: torch.device,
) -> Batch:
    graphs = [
        cache.get_graph(
            image_path=image_path,
            featurizer=featurizer,
            ocr_path=ocr_path,
            ocr_level=args.ocr_level,
            min_area=args.extract_min_area,
            max_shape_nodes=args.extract_max_nodes,
            min_text_conf=args.extract_min_text_conf,
            knn_k=args.extract_knn_k,
            include_shapes=not args.disable_shape_nodes,
        )
        for image_path, ocr_path in zip(batch["image_paths"], batch["ocr_paths"])
    ]
    return Batch.from_data_list(graphs).to(device)


def evaluate_split(
    loader: DataLoader,
    gnn: GraphEncoderV2,
    text_proj: nn.Module,
    featurizer: NodeFeaturizerV2,
    cache: FeatureCacheV2,
    args,
    device: torch.device,
    split_name: str,
) -> dict[str, Any]:
    gnn.eval()
    text_proj.eval()
    all_img_emb: list[torch.Tensor] = []
    all_q_emb: list[torch.Tensor] = []
    ordered_sample_ids: list[str] = []
    ordered_image_ids: list[str] = []
    vqa_predictions: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[eval:{split_name}]", leave=False):
            batch_graph = build_graph_batch(batch, cache, featurizer, args, device)
            q_emb = encode_question_embeddings(batch, featurizer, cache, device)

            z_img = F.normalize(gnn(batch_graph), dim=1)
            z_q = F.normalize(text_proj(q_emb), dim=1)

            logits = compute_option_logits(
                z_img=z_img,
                option_texts=batch["option_texts"],
                option_mask=batch["option_mask"],
                text_encoder=featurizer.text_enc,
                text_proj=text_proj,
                cache=cache,
                temperature=args.temperature,
            )
            targets = batch["correct_indices"].to(device)
            preds = logits.argmax(dim=1)

            all_img_emb.append(z_img.detach().cpu())
            all_q_emb.append(z_q.detach().cpu())
            ordered_sample_ids.extend(batch["sample_ids"])
            ordered_image_ids.extend(batch["image_ids"])

            for i, sample_id in enumerate(batch["sample_ids"]):
                options = batch["options"][i]
                pred_idx = int(preds[i].item())
                gold_idx = int(targets[i].item())
                pred_text = options[pred_idx] if 0 <= pred_idx < len(options) else ""
                gold_text = options[gold_idx] if 0 <= gold_idx < len(options) else ""
                vqa_predictions.append(
                    {
                        "sample_id": sample_id,
                        "image_id": batch["image_ids"][i],
                        "question": batch["questions"][i],
                        "pred_option_idx": pred_idx,
                        "pred_option_text": pred_text,
                        "gold_option_idx": gold_idx,
                        "gold_option_text": gold_text,
                        "is_correct": bool(pred_idx == gold_idx),
                    }
                )

    if not all_img_emb:
        raise RuntimeError(f"No batches for split={split_name}.")

    z_img = torch.cat(all_img_emb, dim=0)
    z_q = torch.cat(all_q_emb, dim=0)
    retrieval = retrieval_metrics_from_embeddings(
        z_img=z_img,
        z_txt=z_q,
        ks=(1, 5, 10),
        image_ids=ordered_image_ids,
    )
    sim = retrieval["sim"]

    logits_tensor = torch.tensor([1.0 if x["is_correct"] else 0.0 for x in vqa_predictions], dtype=torch.float32)
    vqa_acc = float(logits_tensor.mean().item()) if len(logits_tensor) else 0.0
    composite = 0.5 * float(retrieval["mean"][10]) + 0.5 * vqa_acc

    top_k = min(10, sim.size(1))
    top_idx = sim.topk(k=top_k, dim=1).indices
    retrieval_predictions: list[dict[str, Any]] = []
    for row_idx in range(sim.size(0)):
        ranks = [ordered_image_ids[int(col_idx)] for col_idx in top_idx[row_idx].tolist()]
        own_image_id = ordered_image_ids[row_idx]
        retrieval_predictions.append(
            {
                "sample_id": ordered_sample_ids[row_idx],
                "image_id": own_image_id,
                "top10_image_ids": ranks,
                "hit@10": bool(own_image_id in ranks),
            }
        )

    metrics = {
        "split": split_name,
        "i2t": {str(k): float(v) for k, v in retrieval["i2t"].items()},
        "t2i": {str(k): float(v) for k, v in retrieval["t2i"].items()},
        "mean": {str(k): float(v) for k, v in retrieval["mean"].items()},
        "vqa_acc": float(vqa_acc),
        "composite": float(composite),
    }
    return {
        "metrics": metrics,
        "vqa_predictions": vqa_predictions,
        "retrieval_predictions": retrieval_predictions,
    }


def train_epoch(
    loader: DataLoader,
    gnn: GraphEncoderV2,
    text_proj: nn.Module,
    featurizer: NodeFeaturizerV2,
    cache: FeatureCacheV2,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    args,
    device: torch.device,
    stage: str,
) -> dict[str, float]:
    gnn.train()
    text_proj.train()

    total_loss = 0.0
    total_ret = 0.0
    total_vqa = 0.0
    n_steps = 0

    use_amp = scaler.is_enabled()
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"[train:{stage}]", leave=False)
    for step, batch in enumerate(pbar, start=1):
        batch_graph = build_graph_batch(batch, cache, featurizer, args, device)
        q_emb = encode_question_embeddings(batch, featurizer, cache, device)

        with autocast_ctx():
            z_img = F.normalize(gnn(batch_graph), dim=1)
            z_q = F.normalize(text_proj(q_emb), dim=1)
            loss_ret = contrastive_loss(
                z_img,
                z_q,
                temperature=args.temperature,
                group_ids=batch["image_ids"],
            )

            if stage == "stage2":
                logits = compute_option_logits(
                    z_img=z_img,
                    option_texts=batch["option_texts"],
                    option_mask=batch["option_mask"],
                    text_encoder=featurizer.text_enc,
                    text_proj=text_proj,
                    cache=cache,
                    temperature=args.temperature,
                )
                targets = batch["correct_indices"].to(device)
                loss_vqa = F.cross_entropy(logits, targets)
                loss = args.lambda_ret * loss_ret + args.lambda_vqa * loss_vqa
            else:
                loss_vqa = torch.zeros_like(loss_ret)
                loss = loss_ret

        loss_for_backward = loss / max(1, args.grad_accum_steps)
        if scaler.is_enabled():
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        if step % max(1, args.grad_accum_steps) == 0 or step == len(loader):
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item())
        total_ret += float(loss_ret.detach().item())
        total_vqa += float(loss_vqa.detach().item())
        n_steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", ret=f"{loss_ret.item():.4f}", vqa=f"{loss_vqa.item():.4f}")

    denom = max(1, n_steps)
    return {
        "loss": total_loss / denom,
        "loss_ret": total_ret / denom,
        "loss_vqa": total_vqa / denom,
    }


def save_checkpoint(
    path: Path,
    gnn: GraphEncoderV2,
    text_proj: nn.Module,
    optimizer: torch.optim.Optimizer,
    featurizer: NodeFeaturizerV2,
    cache_signature: str,
    args,
    epoch_idx: int,
    stage: str,
    val_metrics: dict[str, Any],
) -> None:
    payload = {
        "epoch": epoch_idx,
        "stage": stage,
        "gnn_state_dict": gnn.state_dict(),
        "text_proj_state_dict": text_proj.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cache_signature": cache_signature,
        "model_config": {
            "in_dim": int(args.in_dim),
            "hidden_dim": int(args.hidden_dim),
            "out_dim": int(args.out_dim),
            "use_attn_pool": bool(not args.disable_attn_pool),
            "vision_model_name": featurizer.vision_model_name,
            "text_model_name": featurizer.text_model_name,
            "ocr_level": args.ocr_level,
            "extract_min_area": int(args.extract_min_area),
            "extract_max_nodes": int(args.extract_max_nodes),
            "extract_min_text_conf": float(args.extract_min_text_conf),
            "extract_knn_k": int(args.extract_knn_k),
            "include_shapes": bool(not args.disable_shape_nodes),
            "temperature": float(args.temperature),
            "use_caption_context": bool(args.use_caption_context),
        },
        "val_metrics": val_metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid AI2D model (retrieval + multiple-choice VQA).")
    parser.add_argument("--manifest", type=Path, default=EXTERNAL_ROOT / "ai2d" / "prepared_v2" / "manifest_hybrid.jsonl")
    parser.add_argument("--split-json", type=Path, default=EXTERNAL_ROOT / "ai2d" / "prepared_v2" / "split_hybrid.json")
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "ai2d_hybrid")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--epochs-stage1", type=int, default=25)
    parser.add_argument("--epochs-stage2", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=6)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--disable-attn-pool", action="store_true")

    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-ret", type=float, default=0.6)
    parser.add_argument("--lambda-vqa", type=float, default=0.4)
    parser.add_argument("--use-caption-context", action="store_true")

    parser.add_argument("--extract-min-area", type=int, default=300)
    parser.add_argument("--extract-max-nodes", type=int, default=80)
    parser.add_argument("--extract-min-text-conf", type=float, default=35.0)
    parser.add_argument("--extract-knn-k", type=int, default=4)
    parser.add_argument("--ocr-level", choices=["line", "word"], default="line")
    parser.add_argument("--disable-shape-nodes", action="store_true")

    parser.add_argument("--cache-dir", type=Path, default=EXTERNAL_ROOT / "ai2d" / "_cache_graph_hybrid")
    parser.add_argument("--disable-cache", action="store_true")

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    samples = resolve_sample_file_paths(
        load_manifest_hybrid(args.manifest),
        roots=[Path.cwd(), ROOT, EXTERNAL_ROOT],
    )
    split_payload = load_split_payload(args.split_json)

    train_samples = select_samples_for_split(samples, "train", split_payload)
    val_samples = select_samples_for_split(samples, "val", split_payload)
    test_samples = select_samples_for_split(samples, "test", split_payload)

    train_samples = maybe_limit_samples(train_samples, args.max_train_samples, seed=args.seed)
    val_samples = maybe_limit_samples(val_samples, args.max_val_samples, seed=args.seed + 1)
    test_samples = maybe_limit_samples(test_samples, args.max_test_samples, seed=args.seed + 2)

    if not train_samples or not val_samples or not test_samples:
        raise SystemExit(
            "Empty split detected. Check manifest/split json or disable restrictive --max-*-samples values."
        )

    print(f"[INFO] train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    collate_fn = make_hybrid_collate_fn(use_caption_context=args.use_caption_context)

    train_loader = DataLoader(
        Ai2dHybridDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    eval_loader_kwargs = {
        "batch_size": args.eval_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
    }
    val_loader = DataLoader(Ai2dHybridDataset(val_samples), **eval_loader_kwargs)
    test_loader = DataLoader(Ai2dHybridDataset(test_samples), **eval_loader_kwargs)

    featurizer = NodeFeaturizerV2(device=str(device))
    cache_signature = build_cache_signature(args, featurizer)
    cache = FeatureCacheV2(
        cache_dir=args.cache_dir,
        signature=cache_signature,
        enabled=not args.disable_cache,
    )

    args.in_dim = featurizer.vision_dim + featurizer.text_dim + 12 + 1 + 1
    gnn = GraphEncoderV2(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        use_attn_pool=not args.disable_attn_pool,
    ).to(device)
    text_proj = nn.Sequential(
        nn.Linear(featurizer.text_dim, args.hidden_dim),
        nn.GELU(),
        nn.Linear(args.hidden_dim, args.out_dim),
    ).to(device)

    optimizer = torch.optim.AdamW(list(gnn.parameters()) + list(text_proj.parameters()), lr=args.lr)
    use_amp = (device.type == "cuda") and (not args.disable_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_composite = float("-inf")
    best_epoch = -1
    no_improve = 0

    checkpoint_path = args.output_dir / "checkpoint_best.pt"
    metrics_path = args.output_dir / "metrics.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage_plan = [("stage1", args.epochs_stage1), ("stage2", args.epochs_stage2)]
    global_epoch = 0
    stop_training = False

    for stage_name, stage_epochs in stage_plan:
        if stage_epochs <= 0:
            continue
        for local_epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            train_stats = train_epoch(
                loader=train_loader,
                gnn=gnn,
                text_proj=text_proj,
                featurizer=featurizer,
                cache=cache,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                device=device,
                stage=stage_name,
            )

            val_result = evaluate_split(
                loader=val_loader,
                gnn=gnn,
                text_proj=text_proj,
                featurizer=featurizer,
                cache=cache,
                args=args,
                device=device,
                split_name="val",
            )
            val_metrics = val_result["metrics"]
            entry = {
                "epoch": global_epoch,
                "stage": stage_name,
                "train": train_stats,
                "val": val_metrics,
            }
            history.append(entry)
            print(
                f"[E{global_epoch} {stage_name}] loss={train_stats['loss']:.4f} "
                f"ret={train_stats['loss_ret']:.4f} vqa={train_stats['loss_vqa']:.4f} | "
                f"val MeanR@10={val_metrics['mean']['10']:.4f} "
                f"val VQA={val_metrics['vqa_acc']:.4f} "
                f"val Composite={val_metrics['composite']:.4f}"
            )

            if val_metrics["composite"] > best_composite:
                best_composite = float(val_metrics["composite"])
                best_epoch = global_epoch
                no_improve = 0
                save_checkpoint(
                    path=checkpoint_path,
                    gnn=gnn,
                    text_proj=text_proj,
                    optimizer=optimizer,
                    featurizer=featurizer,
                    cache_signature=cache_signature,
                    args=args,
                    epoch_idx=global_epoch,
                    stage=stage_name,
                    val_metrics=val_metrics,
                )
                print(f"[INFO] New best checkpoint at epoch {global_epoch} (composite={best_composite:.4f})")
            else:
                no_improve += 1
                if no_improve >= args.early_stopping_patience:
                    print(
                        f"[INFO] Early stopping triggered: "
                        f"no improvement for {no_improve} eval steps."
                    )
                    stop_training = True
                    break
        if stop_training:
            break

    if not checkpoint_path.exists():
        raise RuntimeError("Best checkpoint was not saved.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    gnn.load_state_dict(checkpoint["gnn_state_dict"])
    text_proj.load_state_dict(checkpoint["text_proj_state_dict"])

    test_result = evaluate_split(
        loader=test_loader,
        gnn=gnn,
        text_proj=text_proj,
        featurizer=featurizer,
        cache=cache,
        args=args,
        device=device,
        split_name="test",
    )

    write_jsonl(test_result["vqa_predictions"], args.output_dir / "test_vqa_predictions.jsonl")
    write_jsonl(test_result["retrieval_predictions"], args.output_dir / "test_retrieval_predictions.jsonl")

    metrics_payload = {
        "best_epoch": best_epoch,
        "best_composite": best_composite,
        "train_history": history,
        "test": test_result["metrics"],
        "config": {
            "manifest": str(args.manifest),
            "split_json": str(args.split_json),
            "use_caption_context": bool(args.use_caption_context),
            "lambda_ret": float(args.lambda_ret),
            "lambda_vqa": float(args.lambda_vqa),
            "epochs_stage1": int(args.epochs_stage1),
            "epochs_stage2": int(args.epochs_stage2),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "temperature": float(args.temperature),
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] Best epoch: {best_epoch}")
    print(f"[INFO] Best composite: {best_composite:.4f}")
    print(f"[INFO] Test composite: {test_result['metrics']['composite']:.4f}")
    print(f"[INFO] Artifacts saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
