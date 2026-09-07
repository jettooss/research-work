"""
Run training on AI2D, DocVQA, InfographicVQA sequentially.
Usage: .venv/Scripts/python scripts/train_all.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import json
import os
import random
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Batch
from tqdm.auto import tqdm

try:
    import psutil
except ImportError:  # pragma: no cover - optional reporting dependency
    psutil = None

from vqa_retrieval.datasets import (
    Ai2dRetrievalDataset,
    DocVQARetrievalDataset,
    InfographicVQARetrievalDataset,
)
from vqa_retrieval.graph_builder import (
    FeatureCache,
    GraphEncoder,
    NodeFeaturizer,
    contrastive_loss,
    recall_at_k,
)


def current_ram_mb() -> float | None:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def build_overfitting_check(history):
    if not history:
        return {
            "status": "unavailable",
            "reason": "empty history",
        }

    def mean_r1(row):
        return float(row.get("mean", {}).get("1", 0.0))

    first = history[0]
    final = history[-1]
    best = max(history, key=mean_r1)
    best_mean_r1 = mean_r1(best)
    final_mean_r1 = mean_r1(final)
    final_drop = best_mean_r1 - final_mean_r1
    losses = [float(row.get("loss", 0.0)) for row in history]
    loss_delta = losses[-1] - losses[0] if losses else None
    loss_decreased = losses[-1] < losses[0] if losses else None
    best_epoch = int(best.get("epoch", 0))
    final_epoch = int(final.get("epoch", 0))
    return {
        "status": "possible_overfit" if final_drop > 0.02 and loss_decreased else "no_clear_overfit",
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "first_mean_r1": mean_r1(first),
        "best_mean_r1": best_mean_r1,
        "final_mean_r1": final_mean_r1,
        "final_minus_best_mean_r1": final_mean_r1 - best_mean_r1,
        "final_drop_from_best_mean_r1": final_drop,
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "loss_delta": loss_delta,
        "loss_decreased": loss_decreased,
        "rule": "possible_overfit when final Mean R@1 is >0.02 below best while train loss decreased",
    }


class RetrievalDatasetAdapter(Dataset):
    def __init__(self, base_dataset, filter_missing: bool = True):
        self.samples = []
        for s in base_dataset:
            if filter_missing and not Path(s.image_path).exists():
                continue
            self.samples.append((s.image_path, s.text, s.ocr_path))
        print(f"[{base_dataset.dataset_name}] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def retrieval_collate(batch):
    img_paths, texts, ocr_paths = zip(*batch)
    return list(img_paths), list(texts), list(ocr_paths)


def train_on_dataset(
    dataset_name: str,
    data_root: Path,
    split: str = "train",
    device: str = "cuda",
    batch_size: int = 4,
    epochs: int = 10,
    lr: float = 2e-4,
    hidden_dim: int = 256,
    out_dim: int = 256,
    temperature: float = 0.07,
    ocr_source: str = "auto",
    ocr_lang: str = "eng",
    extract_min_area: int = 300,
    extract_max_nodes: int = 80,
    extract_ocr_conf: int = 60,
    extract_knn_k: int = 4,
    use_attn_pool: bool = True,
    use_cache: bool = True,
    eval_max_items: Optional[int] = 500,
    max_samples: Optional[int] = None,
    val_split: float = 0.1,
    return_metrics: bool = False,
    seed: int = 42,
    checkpoint_dir: Optional[Path] = None,
):
    device = torch.device(device)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()

    # --- dataset ---
    if dataset_name == "ai2d":
        base = Ai2dRetrievalDataset(data_root)
    elif dataset_name == "docvqa":
        base = DocVQARetrievalDataset(data_root, split=split)
    elif dataset_name == "infographicvqa":
        base = InfographicVQARetrievalDataset(data_root, split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    full_ds = RetrievalDatasetAdapter(base)
    if max_samples and max_samples > 0 and len(full_ds.samples) > max_samples:
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(full_ds.samples)), max_samples))
        full_ds.samples = [full_ds.samples[i] for i in idx]
        print(f"[{dataset_name}] limited to {len(full_ds.samples)} samples")
    n = len(full_ds)
    n_val = max(1, int(val_split * n))
    train_ds, val_ds = random_split(full_ds, [n - n_val, n_val],
                                    generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=retrieval_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=retrieval_collate)

    # --- featurizer & cache ---
    featurizer = NodeFeaturizer(device=str(device))
    cache_dir = data_root / f"_cache_graph_{dataset_name}_{ocr_source}"
    sig = FeatureCache.make_signature(
        vision_model_name=featurizer.vision_model_name,
        text_model_name=featurizer.text_model_name,
        ocr_source=ocr_source,
        ocr_lang=ocr_lang,
        min_area=extract_min_area,
        max_nodes=extract_max_nodes,
        ocr_conf=extract_ocr_conf,
        knn_k=extract_knn_k,
    )
    cache = FeatureCache(cache_dir=cache_dir, signature=sig, enabled=use_cache)

    in_dim = featurizer.vision_dim + featurizer.text_dim + 9 + 1
    gnn = GraphEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                       use_attn_pool=use_attn_pool).to(device)
    text_proj = nn.Sequential(
        nn.Linear(featurizer.text_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(gnn.parameters()) + list(text_proj.parameters()), lr=lr)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_graphs(paths, ocr_paths):
        return [cache.get_graph(p, featurizer, ocr_path=op, ocr_source=ocr_source,
                                ocr_lang=ocr_lang, min_area=extract_min_area,
                                max_nodes=extract_max_nodes, ocr_conf=extract_ocr_conf,
                                knn_k=extract_knn_k)
                for p, op in zip(paths, ocr_paths)]

    best_r1 = 0.0
    history = []
    start_epoch = 1
    if checkpoint_dir is not None:
        partial_path = checkpoint_dir / "metrics_partial.json"
        if partial_path.exists():
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            completed = int(partial.get("epochs_completed") or 0)
            if 0 < completed < epochs:
                gnn_path = checkpoint_dir / f"gnn_epoch_{completed:03d}.pt"
                text_proj_path = checkpoint_dir / f"text_proj_epoch_{completed:03d}.pt"
                optimizer_path = checkpoint_dir / f"optimizer_epoch_{completed:03d}.pt"
                if gnn_path.exists() and text_proj_path.exists():
                    gnn.load_state_dict(torch.load(gnn_path, map_location=device))
                    text_proj.load_state_dict(torch.load(text_proj_path, map_location=device))
                    if optimizer_path.exists():
                        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
                    history = list(partial.get("history") or [])
                    best_r1 = float(partial.get("best_mean_r1") or 0.0)
                    start_epoch = completed + 1
                    print(f"[{dataset_name}] resumed from epoch {completed}/{epochs}")

    for epoch in range(start_epoch, epochs + 1):
        gnn.train(); text_proj.train()
        epoch_loss, n_steps = 0.0, 0
        pbar = tqdm(
            train_loader,
            desc=f"[{dataset_name}] Epoch {epoch}/{epochs}",
            leave=False,
            disable=os.environ.get("DISABLE_TQDM") == "1",
        )

        for img_paths, texts, ocr_paths in pbar:
            graphs = get_graphs(img_paths, ocr_paths)
            batch_graph = Batch.from_data_list(graphs).to(device)
            text_emb = cache.get_text_batch(texts, featurizer.text_enc).to(device).clone().detach()

            z_img = F.normalize(gnn(batch_graph), dim=1)
            z_txt = F.normalize(text_proj(text_emb), dim=1)
            loss = contrastive_loss(z_img, z_txt, temperature=temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss)
            n_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- eval ---
        gnn.eval(); text_proj.eval()
        img_embs, txt_embs, seen = [], [], 0
        with torch.no_grad():
            for img_paths, texts, ocr_paths in val_loader:
                graphs = get_graphs(img_paths, ocr_paths)
                batch_graph = Batch.from_data_list(graphs).to(device)
                text_emb = cache.get_text_batch(texts, featurizer.text_enc).to(device)
                img_embs.append(F.normalize(gnn(batch_graph), dim=1).cpu())
                txt_embs.append(F.normalize(text_proj(text_emb), dim=1).cpu())
                seen += len(texts)
                if eval_max_items and seen >= eval_max_items:
                    break

        sim = torch.cat(img_embs) @ torch.cat(txt_embs).t()
        r_i2t = recall_at_k(sim)
        r_t2i = recall_at_k(sim.t())
        r_mean = {k: 0.5 * (r_i2t[k] + r_t2i[k]) for k in (1, 5, 10)}

        if r_mean[1] > best_r1:
            best_r1 = r_mean[1]
        epoch_metrics = {
            "epoch": epoch,
            "loss": epoch_loss / max(1, n_steps),
            "i2t": {str(k): float(v) for k, v in r_i2t.items()},
            "t2i": {str(k): float(v) for k, v in r_t2i.items()},
            "mean": {str(k): float(v) for k, v in r_mean.items()},
        }
        history.append(epoch_metrics)
        overfitting_check = build_overfitting_check(history)
        if checkpoint_dir is not None:
            torch.save(gnn.state_dict(), checkpoint_dir / f"gnn_epoch_{epoch:03d}.pt")
            torch.save(text_proj.state_dict(), checkpoint_dir / f"text_proj_epoch_{epoch:03d}.pt")
            torch.save(optimizer.state_dict(), checkpoint_dir / f"optimizer_epoch_{epoch:03d}.pt")
            partial_metrics = {
                "best_mean_r1": float(best_r1),
                "best_epoch": overfitting_check.get("best_epoch"),
                "overfitting_check": overfitting_check,
                "history": history,
                "final": history[-1] if history else {},
                "num_samples": n,
                "num_train_samples": len(train_ds),
                "num_val_samples": len(val_ds),
                "epochs_completed": epoch,
                "epochs_requested": epochs,
                "elapsed_seconds": time.perf_counter() - started_at,
                "peak_ram_mb": current_ram_mb(),
                "peak_vram_mb": (
                    torch.cuda.max_memory_allocated() / (1024 * 1024)
                    if torch.cuda.is_available()
                    else None
                ),
            }
            (checkpoint_dir / "metrics_partial.json").write_text(
                json.dumps(partial_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        print(
            f"[{dataset_name}|{ocr_source}] Epoch {epoch}/{epochs} "
            f"loss={epoch_metrics['loss']:.4f} | "
            f"Mean R@1={r_mean[1]:.4f} R@5={r_mean[5]:.4f} R@10={r_mean[10]:.4f}"
        )

    print(f"\n[{dataset_name}] Best Mean R@1 = {best_r1:.4f}")
    if return_metrics:
        overfitting_check = build_overfitting_check(history)
        return gnn, text_proj, featurizer, {
            "best_mean_r1": float(best_r1),
            "best_epoch": overfitting_check.get("best_epoch"),
            "overfitting_check": overfitting_check,
            "history": history,
            "final": history[-1] if history else {},
            "num_samples": n,
            "num_train_samples": len(train_ds),
            "num_val_samples": len(val_ds),
            "peak_ram_mb": current_ram_mb(),
            "peak_vram_mb": (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if torch.cuda.is_available()
                else None
            ),
        }
    return gnn, text_proj, featurizer


if __name__ == "__main__":
    configs = [
        dict(dataset_name="docvqa",         data_root=EXTERNAL_ROOT / "docvqa",        ocr_source="auto",      split="train"),
        dict(dataset_name="infographicvqa", data_root=EXTERNAL_ROOT / "infographicvqa", ocr_source="tesseract", split="train"),
    ]

    all_results = {}
    for cfg in configs:
        name = cfg["dataset_name"]
        print(f"\n{'='*60}")
        print(f"  DATASET: {name.upper()}")
        print(f"{'='*60}")
        gnn, tp, feat = train_on_dataset(**cfg, epochs=10, batch_size=4, eval_max_items=500)

        out_dir = ROOT / "runs" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(gnn.state_dict(), out_dir / "gnn.pt")
        torch.save(tp.state_dict(), out_dir / "text_proj.pt")
        print(f"Saved weights -> {out_dir}")

    print("\nAll done.")
