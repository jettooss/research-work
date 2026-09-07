from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GlobalAttention
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentence_transformers import SentenceTransformer  # noqa: E402
from vqa_retrieval.docvqa_compact_graph import CompactDocVqaGraphCache  # noqa: E402
from vqa_retrieval.graph_builder import FeatureCache  # noqa: E402


VARIANTS = ("text_only", "coords_no_edges", "knn_k3", "knn_k5", "typed_spatial")
EDGE_TYPES = {"knn": 0, "text_near_shape": 1, "contains": 2, "left_right": 3, "above_below": 4, "label_to_object": 5}
TEXT_DIM = 384
GEOM_START = TEXT_DIM
KIND_INDEX = GEOM_START + 9


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class DocRows(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {key: [row[key] for row in rows] for key in ("image_id", "image_path", "ocr_path", "question")}


def self_loops(num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(num_nodes, dtype=torch.long)
    return torch.stack([indices, indices]), torch.zeros(num_nodes, dtype=torch.long)


def knn_edges(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    centers = x[:, GEOM_START : GEOM_START + 2]
    if len(centers) <= 1:
        return self_loops(len(centers))
    distances = torch.cdist(centers, centers)
    distances.fill_diagonal_(float("inf"))
    k_eff = min(k, len(centers) - 1)
    neighbors = distances.topk(k_eff, largest=False).indices
    src = torch.arange(len(centers)).repeat_interleave(k_eff)
    dst = neighbors.reshape(-1)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    return edge_index, torch.zeros(edge_index.shape[1], dtype=torch.long)


def typed_edges(x: torch.Tensor, k: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index, _ = knn_edges(x, k)
    boxes = x[:, GEOM_START + 5 : GEOM_START + 9]
    kinds = x[:, KIND_INDEX]
    edge_types: list[int] = []
    for src, dst in edge_index.t().tolist():
        if src == dst:
            edge_types.append(EDGE_TYPES["knn"])
            continue
        left, right = boxes[src], boxes[dst]
        if left[2] < right[0] or right[2] < left[0]:
            relation = "left_right"
        elif left[3] < right[1] or right[3] < left[1]:
            relation = "above_below"
        elif left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]:
            relation = "contains"
        elif bool(kinds[src] > 0.5) != bool(kinds[dst] > 0.5):
            relation = "text_near_shape"
        else:
            relation = "knn"
        edge_types.append(EDGE_TYPES[relation])
    return edge_index, torch.tensor(edge_types, dtype=torch.long)


def apply_variant(graph: Data, variant: str) -> Data:
    x = graph.x.clone()
    if variant == "text_only":
        x[:, GEOM_START:] = 0
        edge_index, edge_type = self_loops(x.shape[0])
    elif variant == "coords_no_edges":
        edge_index, edge_type = self_loops(x.shape[0])
    elif variant == "knn_k3":
        edge_index, edge_type = knn_edges(x, 3)
    elif variant == "knn_k5":
        edge_index, edge_type = knn_edges(x, 5)
    elif variant == "typed_spatial":
        edge_index, edge_type = typed_edges(x, 4)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return Data(x=x, edge_index=edge_index, edge_type=edge_type)


class AblationGraphEncoder(nn.Module):
    """One parameter-identical encoder for every edge/feature ablation."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 256, heads: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.edge_embedding = nn.Embedding(len(EDGE_TYPES), hidden_dim)
        self.gnn1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=hidden_dim, dropout=0.1)
        self.gnn2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=hidden_dim, dropout=0.1)
        self.out = nn.Linear(hidden_dim, out_dim)
        gate = nn.Sequential(nn.Linear(out_dim, out_dim // 2), nn.GELU(), nn.Linear(out_dim // 2, 1))
        self.pool = GlobalAttention(gate_nn=gate)

    def forward(self, batch: Batch) -> torch.Tensor:
        edge_attr = self.edge_embedding(batch.edge_type)
        x = F.gelu(self.proj(batch.x))
        x = F.gelu(self.gnn1(x, batch.edge_index, edge_attr=edge_attr))
        x = F.gelu(self.gnn2(x, batch.edge_index, edge_attr=edge_attr))
        return self.pool(F.gelu(self.out(x)), batch.batch)


def multi_positive_loss(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, image_ids: list[str], temperature: float) -> torch.Tensor:
    logits = image_embeddings @ text_embeddings.T / temperature
    positive = torch.tensor(
        [[left == right for right in image_ids] for left in image_ids],
        dtype=torch.bool,
        device=logits.device,
    )

    def direction(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numerator = torch.logsumexp(values.masked_fill(~mask, -torch.inf), dim=1)
        denominator = torch.logsumexp(values, dim=1)
        return -(numerator - denominator).mean()

    return 0.5 * (direction(logits, positive) + direction(logits.T, positive.T))


def load_state(path: Path) -> Any:
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def graph_batch(batch: dict[str, list[Any]], cache: CompactDocVqaGraphCache, variant: str, device: torch.device) -> Batch:
    graphs = [
        apply_variant(cache.get(image, ocr), variant)
        for image, ocr in zip(batch["image_path"], batch["ocr_path"])
    ]
    return Batch.from_data_list(graphs).to(device)


@torch.no_grad()
def evaluate(
    model: AblationGraphEncoder,
    text_projection: nn.Module,
    rows: list[dict[str, Any]],
    graph_cache: CompactDocVqaGraphCache,
    text_cache: FeatureCache,
    text_encoder,
    variant: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    text_projection.eval()
    unique_docs: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique_docs.setdefault(str(row["image_id"]), row)
    docs = list(unique_docs.values())
    doc_loader = DataLoader(DocRows(docs), batch_size=batch_size, shuffle=False, collate_fn=collate)
    question_loader = DataLoader(DocRows(rows), batch_size=batch_size * 2, shuffle=False, collate_fn=collate)
    doc_embeddings = []
    for batch in doc_loader:
        doc_embeddings.append(F.normalize(model(graph_batch(batch, graph_cache, variant, device)), dim=1).cpu())
    question_embeddings = []
    for batch in question_loader:
        text = text_cache.get_text_batch(batch["question"], text_encoder).to(device)
        question_embeddings.append(F.normalize(text_projection(text), dim=1).cpu())
    doc_tensor = torch.cat(doc_embeddings)
    question_tensor = torch.cat(question_embeddings)
    sim = question_tensor @ doc_tensor.T
    doc_index = {str(row["image_id"]): idx for idx, row in enumerate(docs)}
    q2d_target = torch.tensor([doc_index[str(row["image_id"])] for row in rows], dtype=torch.long)
    q2d_scores = sim[torch.arange(len(rows)), q2d_target]
    q2d_ranks = 1 + (sim > q2d_scores[:, None]).sum(dim=1)
    questions_by_doc: dict[str, set[int]] = defaultdict(set)
    for idx, row in enumerate(rows):
        questions_by_doc[str(row["image_id"])].add(idx)
    d2q_sim = sim.T
    d2q_best_scores = torch.stack(
        [d2q_sim[idx, list(questions_by_doc[str(row["image_id"])])].max() for idx, row in enumerate(docs)]
    )
    d2q_ranks = 1 + (d2q_sim > d2q_best_scores[:, None]).sum(dim=1)

    def retrieval_from_ranks(ranks: torch.Tensor) -> dict[str, Any]:
        return {
            "recall_at_k": {str(k): float((ranks <= k).float().mean()) for k in (1, 5, 10)},
            "mrr": float((1.0 / ranks.float()).mean()),
        }

    q2d_metrics = retrieval_from_ranks(q2d_ranks)
    d2q_metrics = retrieval_from_ranks(d2q_ranks)
    return {
        "question_to_document": q2d_metrics,
        "document_to_question": d2q_metrics,
        "mean_recall_at_k": {str(k): (q2d_metrics["recall_at_k"][str(k)] + d2q_metrics["recall_at_k"][str(k)]) / 2 for k in (1, 5, 10)},
        "num_samples": len(rows),
        "num_documents": len(docs),
    }


def ensure_disk_space(path: Path, minimum_free_gb: float) -> None:
    free = shutil.disk_usage(path.resolve().anchor).free / (1024**3)
    if free < minimum_free_gb:
        raise SystemExit(f"Only {free:.2f} GiB free; {minimum_free_gb:.2f} GiB is required before training.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reproducible DocVQA OCR graph ablations.")
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "docvqa" / "prepared_v1" / "manifest.jsonl")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "docvqa_gnn")
    parser.add_argument("--cache-dir", type=Path, default=ROOT.parent / "docvqa" / "_cache_graph_docvqa_auto")
    parser.add_argument("--compact-cache-dir", type=Path, default=ROOT.parent / "docvqa" / "_cache_graph_docvqa_ocr_compact")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--minimum-free-gb", type=float, default=8.0)
    parser.add_argument("--test-after-training", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    ensure_disk_space(args.output_dir, args.minimum_free_gb)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rows = load_jsonl(args.manifest)
    train_rows = [row for row in rows if row.get("split") == "train"]
    eval_rows = [row for row in rows if row.get("split") == args.split]
    test_rows = [row for row in rows if row.get("split") == "test"]
    if args.max_samples:
        train_rows = train_rows[: args.max_samples]
        eval_rows = eval_rows[: max(1, min(args.max_samples, len(eval_rows)))]
        test_rows = test_rows[: max(1, min(args.max_samples, len(test_rows)))]

    run_dir = args.output_dir / args.variant / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
    signature = FeatureCache.make_signature(
        vision_model_name="vit_base_patch16_224",
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
        ocr_source="auto",
        ocr_lang="eng",
        min_area=300,
        max_nodes=80,
        ocr_conf=60,
        knn_k=4,
    )
    text_cache = FeatureCache(args.cache_dir, signature=signature, enabled=True, max_mem_graphs=0, max_mem_texts=2048)
    graph_cache = CompactDocVqaGraphCache(args.compact_cache_dir, args.cache_dir, signature, text_encoder)
    in_dim = TEXT_DIM + 10
    model = AblationGraphEncoder(in_dim).to(device)
    text_projection = nn.Sequential(nn.Linear(TEXT_DIM, 256), nn.GELU(), nn.Linear(256, 256)).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(text_projection.parameters()), lr=2e-4)
    parameter_count = sum(parameter.numel() for parameter in model.parameters()) + sum(parameter.numel() for parameter in text_projection.parameters())
    history: list[dict[str, Any]] = []
    start_epoch = 1
    state_path = run_dir / "resume_state.pt"
    partial_path = run_dir / "metrics_partial.json"
    if args.resume and state_path.exists() and partial_path.exists():
        state = load_state(state_path)
        model.load_state_dict(state["model"])
        text_projection.load_state_dict(state["text_projection"])
        optimizer.load_state_dict(state["optimizer"])
        history = json.loads(partial_path.read_text(encoding="utf-8")).get("history", [])
        start_epoch = int(state["epoch"]) + 1

    loader = DataLoader(DocRows(train_rows), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    best_r1 = max((float(item["eval"]["mean_recall_at_k"]["1"]) for item in history), default=-math.inf)
    if history:
        best_history_index = max(range(len(history)), key=lambda idx: float(history[idx]["eval"]["mean_recall_at_k"]["1"]))
        stale_epochs = len(history) - best_history_index - 1
    else:
        stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        text_projection.train()
        total_loss = 0.0
        steps = 0
        progress = tqdm(loader, desc=f"DocVQA {args.variant} seed={args.seed} epoch={epoch}/{args.epochs}", disable=os.environ.get("DISABLE_TQDM") == "1")
        for batch in progress:
            graph = graph_batch(batch, graph_cache, args.variant, device)
            text = text_cache.get_text_batch(batch["question"], text_encoder).to(device)
            image_embeddings = F.normalize(model(graph), dim=1)
            text_embeddings = F.normalize(text_projection(text), dim=1)
            loss = multi_positive_loss(image_embeddings, text_embeddings, [str(value) for value in batch["image_id"]], 0.07)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
            progress.set_postfix(loss=f"{float(loss):.4f}")
        eval_metrics = evaluate(model, text_projection, eval_rows, graph_cache, text_cache, text_encoder, args.variant, device, args.eval_batch_size)
        epoch_row = {"epoch": epoch, "loss": total_loss / max(1, steps), "eval": eval_metrics}
        history.append(epoch_row)
        current_r1 = float(eval_metrics["mean_recall_at_k"]["1"])
        if current_r1 > best_r1:
            best_r1 = current_r1
            stale_epochs = 0
            torch.save(model.state_dict(), run_dir / "gnn_best.pt")
            torch.save(text_projection.state_dict(), run_dir / "text_proj_best.pt")
        else:
            stale_epochs += 1
        torch.save({"epoch": epoch, "model": model.state_dict(), "text_projection": text_projection.state_dict(), "optimizer": optimizer.state_dict()}, state_path)
        partial = {"dataset": "docvqa", "variant": args.variant, "seed": args.seed, "epochs_completed": epoch, "history": history}
        partial_path.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.early_stopping_patience > 0 and stale_epochs >= args.early_stopping_patience:
            break

    best = max(history, key=lambda item: float(item["eval"]["mean_recall_at_k"]["1"]))
    test_metrics = None
    if args.test_after_training and args.split == "val":
        model.load_state_dict(load_state(run_dir / "gnn_best.pt"))
        text_projection.load_state_dict(load_state(run_dir / "text_proj_best.pt"))
        test_metrics = evaluate(model, text_projection, test_rows, graph_cache, text_cache, text_encoder, args.variant, device, args.eval_batch_size)
        (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started
    metrics = {
        "dataset": "docvqa",
        "metric_family": "retrieval",
        "variant": args.variant,
        "seed": args.seed,
        "split": args.split,
        "status": "available_full" if args.max_samples is None else "available_partial",
        "epochs_completed": len(history),
        "epochs_requested": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "best_epoch": best["epoch"],
        "best": best["eval"],
        "test": test_metrics,
        "history": history,
        "wall_time_seconds": elapsed,
        "latency_ms_per_question": elapsed * 1000 / max(1, len(eval_rows)),
        "throughput_questions_per_second": len(eval_rows) / elapsed if elapsed else None,
        "peak_ram_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
        "parameter_count": parameter_count,
        "num_train_samples": len(train_rows),
        "num_eval_samples": len(eval_rows),
        "checkpoint": str((run_dir / "gnn_best.pt").resolve()),
        "cache_signature": signature,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
