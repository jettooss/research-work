from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GlobalAttention, TransformerConv
from torch_geometric.utils import to_dense_batch
from tqdm.auto import tqdm
if __package__:
    from .model_matrix_integrity import (
        capture_rng_state, checkpoint_config, completed_metrics, digest, file_digest,
        restore_rng_state, test_config, training_config, training_paths,
        validate_directory, validate_partial_checkpoint, write_config,
    )
else:
    from model_matrix_integrity import (
        capture_rng_state, checkpoint_config, completed_metrics, digest, file_digest,
        restore_rng_state, test_config, training_config, training_paths,
        validate_directory, validate_partial_checkpoint, write_config,
    )


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentence_transformers import SentenceTransformer  # noqa: E402
from vqa_retrieval.model_matrix import (  # noqa: E402
    DATASETS,
    SEEDS,
    TRAINABLE_LOCAL_MODELS,
    candidate_texts,
    ensure_free_space,
    load_rows,
    ocr_spans,
    retrieval_metrics,
    split_rows,
    vqa_row,
)
from vqa_retrieval.public_vqa_metrics import anls_score, normalize_answer  # noqa: E402


TEXT_DIM = 384
NODE_DIM = TEXT_DIM + 10
GRAPH_MODELS = {
    "gatv2_knn",
    "graph_transformer",
    "sam2_graph_transformer",
    "hybrid_gatv2_knn",
    "graphcolbert",
    "graphcolbert_film",
    "ocr_reasoner",
    "epoch_view_graph_transformer",
    "sam2_graph_nodes",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_torch(path: Path) -> Any:
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def knn_edges(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    if len(x) <= 1:
        return torch.zeros((2, 1), dtype=torch.long)
    centers = x[:, TEXT_DIM : TEXT_DIM + 2]
    distances = torch.cdist(centers, centers)
    distances.fill_diagonal_(float("inf"))
    k_eff = min(k, len(x) - 1)
    dst = distances.topk(k_eff, largest=False).indices.reshape(-1)
    src = torch.arange(len(x)).repeat_interleave(k_eff)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


class RowDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class MatrixFeatureStore:
    def __init__(self, dataset: str, model_name: str, cache_root: Path, device: torch.device) -> None:
        self.dataset = dataset
        self.model_name = model_name
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
        self._question: dict[str, torch.Tensor] = {}
        self._question_tokens: dict[str, torch.Tensor] = {}
        self._text_values: dict[str, torch.Tensor] = {}
        self._graphs: dict[str, Data] = {}
        self._candidate_targets: dict[str, int] = {}
        self._sam_model = None

    def encode_texts(self, values: list[str]) -> torch.Tensor:
        if not values:
            return torch.zeros((0, TEXT_DIM), dtype=torch.float32)
        # Keep the tensors required by this call separately from the bounded
        # cache.  A large OCR candidate set can evict an earlier requested
        # value while the same call is still being assembled (for example the
        # valid AI2D option "-").  In that case the cache must not make a
        # batch fail with KeyError.
        unique_values = list(dict.fromkeys(str(value) for value in values))
        resolved = {value: self._text_values[value] for value in unique_values if value in self._text_values}
        missing = [value for value in unique_values if value not in resolved]
        if missing:
            encoded = self.text_encoder.encode(missing, batch_size=128, convert_to_tensor=True, normalize_embeddings=False)
            for value, tensor in zip(missing, encoded.detach().cpu().float()):
                resolved[value] = tensor
                self._text_values[value] = tensor
                if len(self._text_values) > 50_000:
                    self._text_values.pop(next(iter(self._text_values)))
        return torch.stack([resolved[str(value)] for value in values])

    def question(self, row: dict[str, Any]) -> torch.Tensor:
        key = row["sample_id"]
        if key not in self._question:
            self._question[key] = self.encode_texts([str(row["question"])])[0]
        return self._question[key]

    def question_tokens(self, row: dict[str, Any]) -> torch.Tensor:
        key = row["sample_id"]
        if key not in self._question_tokens:
            tokens = [token for token in str(row["question"]).split() if token][:32] or [""]
            self._question_tokens[key] = self.encode_texts(tokens)
        return self._question_tokens[key]

    def prepare_questions(self, rows: list[dict[str, Any]]) -> None:
        missing = [row for row in rows if row["sample_id"] not in self._question]
        if missing:
            values = [str(row["question"]) for row in missing]
            encoded = self.encode_texts(values)
            for row, tensor in zip(missing, encoded):
                self._question[row["sample_id"]] = tensor

        token_rows = [row for row in rows if row["sample_id"] not in self._question_tokens]
        if token_rows:
            tokens_by_row = [[token for token in str(row["question"]).split() if token][:32] or [""] for row in token_rows]
            flat_tokens = [token for tokens in tokens_by_row for token in tokens]
            encoded = self.encode_texts(flat_tokens)
            offset = 0
            for row, tokens in zip(token_rows, tokens_by_row):
                self._question_tokens[row["sample_id"]] = encoded[offset : offset + len(tokens)]
                offset += len(tokens)

    def preload_graphs(self, rows: list[dict[str, Any]], workers: int = 8) -> None:
        directory = self.cache_root / "graphs" / self.dataset / self.model_name
        directory.mkdir(parents=True, exist_ok=True)
        rows_by_image = {str(row["image_id"]): row for row in rows}
        image_ids = list(dict.fromkeys(str(row["image_id"]) for row in rows))
        missing = [image_id for image_id in image_ids if image_id not in self._graphs]

        cached = [image_id for image_id in missing if (directory / f"{image_id}.pt").exists()]
        cached_ids = set(cached)
        uncached = [image_id for image_id in missing if image_id not in cached_ids]

        def load_one(image_id: str) -> tuple[str, Data]:
            return image_id, load_torch(directory / f"{image_id}.pt")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for image_id, graph in tqdm(executor.map(load_one, cached), total=len(cached), desc=f"preload {self.dataset} graphs"):
                self._graphs[image_id] = graph

        # A partially built disk cache is valid: generate only the absent
        # graphs instead of assuming every image already has a .pt file.
        # Build sequentially because SentenceTransformer.encode is not
        # guaranteed to be thread-safe on the shared encoder instance.
        for image_id in tqdm(uncached, desc=f"build missing {self.dataset} graphs"):
            self.graph(rows_by_image[image_id])

    def _sam_boxes(self, row: dict[str, Any], width: int, height: int) -> list[list[float]]:
        cache_dir = self.cache_root / "sam2_boxes" / self.dataset
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{row['image_id']}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        try:
            from ultralytics import SAM

            if self._sam_model is None:
                self._sam_model = SAM(str(EXTERNAL_ROOT / "models/sam2/sam2.1_b.pt"))
            result = self._sam_model(str(row["image_path"]), imgsz=768, verbose=False)[0]
            boxes = result.boxes.xyxy.detach().cpu().tolist()[:80] if result.boxes is not None else []
        except Exception as exc:
            print(f"[SAM2 fallback] {type(exc).__name__}: {exc}", flush=True)
            boxes = [[0.0, 0.0, float(width), float(height)]]
        path.write_text(json.dumps(boxes), encoding="utf-8")
        return boxes

    def graph(self, row: dict[str, Any]) -> Data:
        key = row["image_id"]
        if key in self._graphs:
            return self._graphs[key]
        directory = self.cache_root / "graphs" / self.dataset / self.model_name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{key}.pt"
        if path.exists():
            graph = load_torch(path)
            self._graphs[key] = graph
            return graph

        spans = ocr_spans(self.dataset, row, EXTERNAL_ROOT, self.cache_root / "ocr")
        width, height = Image.open(row["image_path"]).size
        texts = [str(span["text"]) for span in spans[:80]]
        text_embeddings = self.encode_texts(texts)
        features = []
        for index, span in enumerate(spans[:80]):
            x1, y1, x2, y2 = [float(value) for value in span.get("bbox", [0, 0, 0, 0])]
            geometry = torch.tensor([
                (x1 + x2) / 2 / max(1, width), (y1 + y2) / 2 / max(1, height),
                max(1.0, x2 - x1) / max(1, width), max(1.0, y2 - y1) / max(1, height),
                max(1.0, (x2 - x1) * (y2 - y1)) / max(1, width * height),
                x1 / max(1, width), y1 / max(1, height), x2 / max(1, width), y2 / max(1, height), 1.0,
            ])
            features.append(torch.cat([text_embeddings[index], geometry]))
        if self.model_name in {"sam2_graph_nodes", "sam2_graph_transformer"}:
            for x1, y1, x2, y2 in self._sam_boxes(row, width, height):
                geometry = torch.tensor([
                    (x1 + x2) / 2 / max(1, width), (y1 + y2) / 2 / max(1, height),
                    max(1.0, x2 - x1) / max(1, width), max(1.0, y2 - y1) / max(1, height),
                    max(1.0, (x2 - x1) * (y2 - y1)) / max(1, width * height),
                    x1 / max(1, width), y1 / max(1, height), x2 / max(1, width), y2 / max(1, height), 0.0,
                ])
                features.append(torch.cat([torch.zeros(TEXT_DIM), geometry]))
        if not features:
            features = [torch.zeros(NODE_DIM)]
            texts = [""]
        x = torch.stack(features).float()
        candidates = candidate_texts([{"text": value} for value in texts], max_candidates=128) or [""]
        candidate_embeddings = self.encode_texts(candidates)
        graph = Data(x=x, edge_index=knn_edges(x), candidate_embeddings=candidate_embeddings)
        graph.candidate_texts = candidates
        torch.save(graph, path)
        self._graphs[key] = graph
        return graph

    def candidates(self, row: dict[str, Any], graph: Data) -> tuple[list[str], torch.Tensor, int]:
        if self.dataset == "ai2d":
            values = [str(value) for value in row.get("options", [])]
            embeddings = self.encode_texts(values)
            target = int(row.get("correct_option_idx", 0))
            return values, embeddings, target
        values = list(graph.candidate_texts)
        embeddings = graph.candidate_embeddings.float()
        key = row["sample_id"]
        if key not in self._candidate_targets:
            answers = row.get("answers", [])
            normalized_answers = {normalize_answer(str(answer)) for answer in answers}
            exact_target = next(
                (index for index, value in enumerate(values) if normalize_answer(str(value)) in normalized_answers),
                None,
            )
            if exact_target is not None:
                self._candidate_targets[key] = exact_target
            else:
                scores = [max((anls_score(value, [answer]) for answer in answers), default=0.0) for value in values]
                self._candidate_targets[key] = int(np.argmax(scores)) if scores else 0
        target = self._candidate_targets[key]
        return values, embeddings, target


class GraphMatrixModel(nn.Module):
    def __init__(self, model_name: str, hidden: int = 192, out_dim: int = 192) -> None:
        super().__init__()
        self.model_name = model_name
        self.input = nn.Linear(NODE_DIM, hidden)
        self.q_proj = nn.Sequential(nn.Linear(TEXT_DIM, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        self.candidate_proj = nn.Sequential(nn.Linear(TEXT_DIM, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        if model_name == "ocr_reasoner":
            self.layer1 = nn.Linear(hidden, hidden)
            self.layer2 = nn.Linear(hidden, hidden)
        elif model_name in {"graph_transformer", "epoch_view_graph_transformer", "sam2_graph_transformer"}:
            self.layer1 = TransformerConv(hidden, hidden // 4, heads=4, dropout=0.1)
            self.layer2 = TransformerConv(hidden, hidden // 4, heads=4, dropout=0.1)
        else:
            self.layer1 = GATv2Conv(hidden, hidden // 4, heads=4, dropout=0.1)
            self.layer2 = GATv2Conv(hidden, hidden // 4, heads=4, dropout=0.1)
        self.node_out = nn.Linear(hidden, out_dim)
        gate = nn.Sequential(nn.Linear(out_dim, out_dim // 2), nn.GELU(), nn.Linear(out_dim // 2, 1))
        self.pool = GlobalAttention(gate_nn=gate)
        self.hybrid = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.film = nn.Linear(out_dim, out_dim * 2)
        self.epoch_view = nn.Sequential(nn.Linear(out_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def encode_nodes(self, batch: Batch) -> torch.Tensor:
        x = F.gelu(self.input(batch.x))
        if self.model_name == "ocr_reasoner":
            x = F.gelu(self.layer1(x))
            x = F.gelu(self.layer2(x))
        else:
            x = F.gelu(self.layer1(x, batch.edge_index))
            x = F.gelu(self.layer2(x, batch.edge_index))
        nodes = F.gelu(self.node_out(x))
        if self.model_name == "epoch_view_graph_transformer":
            nodes = nodes + self.epoch_view(nodes)
        return nodes

    def encode(self, batch: Batch, questions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = self.encode_nodes(batch)
        if self.model_name == "graphcolbert_film" and questions is not None:
            q = self.q_proj(questions)
            gamma, beta = self.film(q).chunk(2, dim=-1)
            nodes = nodes * (1 + gamma[batch.batch]) + beta[batch.batch]
        pooled = self.pool(nodes, batch.batch)
        return F.normalize(pooled, dim=-1), nodes

    def retrieval_logits(self, batch: Batch, questions: torch.Tensor, question_tokens: list[torch.Tensor] | None = None) -> torch.Tensor:
        q = F.normalize(self.q_proj(questions), dim=-1)
        docs, nodes = self.encode(batch)
        if self.model_name not in {"graphcolbert", "graphcolbert_film"}:
            return q @ docs.T
        token_queries = [self.q_proj(tokens.to(q.device)) for tokens in (question_tokens or [])]
        max_tokens = max(tokens.shape[0] for tokens in token_queries)
        token_dense = torch.zeros((len(token_queries), max_tokens, token_queries[0].shape[-1]), device=q.device)
        token_mask = torch.zeros((len(token_queries), max_tokens), dtype=torch.bool, device=q.device)
        for index, tokens in enumerate(token_queries):
            token_dense[index, : tokens.shape[0]] = F.normalize(tokens, dim=-1)
            token_mask[index, : tokens.shape[0]] = True
        node_dense, node_mask = to_dense_batch(nodes, batch.batch)
        if self.model_name == "graphcolbert_film":
            gamma, beta = self.film(q).chunk(2, dim=-1)
            conditioned = node_dense.unsqueeze(0) * (1 + gamma[:, None, None, :]) + beta[:, None, None, :]
            conditioned = F.normalize(conditioned, dim=-1)
            scores = torch.einsum("itd,ijnd->ijtn", token_dense, conditioned)
        else:
            node_dense = F.normalize(node_dense, dim=-1)
            scores = torch.einsum("itd,jnd->ijtn", token_dense, node_dense)
        scores = scores.masked_fill(~node_mask[None, :, None, :], -torch.inf)
        token_scores = scores.max(dim=-1).values.masked_fill(~token_mask[:, None, :], 0.0)
        return token_scores.sum(dim=-1) / token_mask.sum(dim=-1).clamp_min(1)[:, None]

    def vqa_logits(self, graph: Data, question: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        batch = Batch.from_data_list([graph]).to(question.device)
        q = F.normalize(self.q_proj(question.unsqueeze(0)), dim=-1)
        doc, nodes = self.encode(batch, question.unsqueeze(0) if self.model_name == "graphcolbert_film" else None)
        if self.model_name in {"hybrid_gatv2_knn", "graphcolbert", "graphcolbert_film"}:
            if self.model_name.startswith("graphcolbert"):
                node_context = F.normalize(nodes, dim=-1)
                context = F.normalize((q @ node_context.T).softmax(dim=1) @ node_context + q, dim=-1)
            else:
                context = F.normalize(self.hybrid(torch.cat([doc, q], dim=-1)), dim=-1)
        else:
            context = F.normalize(doc + q, dim=-1)
        candidate = F.normalize(self.candidate_proj(candidates.to(question.device)), dim=-1)
        return (context @ candidate.T).squeeze(0)

    def vqa_logits_batch(
        self,
        batch: Batch,
        questions: torch.Tensor,
        candidates: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        q = F.normalize(self.q_proj(questions), dim=-1)
        docs, nodes = self.encode(batch, questions if self.model_name == "graphcolbert_film" else None)
        contexts: list[torch.Tensor] = []
        for index in range(int(batch.num_graphs)):
            if self.model_name.startswith("graphcolbert"):
                current_nodes = F.normalize(nodes[batch.batch == index], dim=-1)
                context = F.normalize((q[index : index + 1] @ current_nodes.T).softmax(dim=1) @ current_nodes + q[index : index + 1], dim=-1)
            elif self.model_name == "hybrid_gatv2_knn":
                context = F.normalize(self.hybrid(torch.cat([docs[index : index + 1], q[index : index + 1]], dim=-1)), dim=-1)
            else:
                context = F.normalize(docs[index : index + 1] + q[index : index + 1], dim=-1)
            contexts.append(context)
        return [
            (context @ F.normalize(self.candidate_proj(values.to(questions.device)), dim=-1).T).squeeze(0)
            for context, values in zip(contexts, candidates)
        ]


def multipositive_loss(logits: torch.Tensor, image_ids: list[str]) -> torch.Tensor:
    positive = torch.tensor([[left == right for right in image_ids] for left in image_ids], device=logits.device)
    numerator = torch.logsumexp(logits.masked_fill(~positive, -torch.inf) / 0.07, dim=1)
    denominator = torch.logsumexp(logits / 0.07, dim=1)
    reverse_num = torch.logsumexp(logits.T.masked_fill(~positive.T, -torch.inf) / 0.07, dim=1)
    reverse_den = torch.logsumexp(logits.T / 0.07, dim=1)
    return -0.5 * ((numerator - denominator).mean() + (reverse_num - reverse_den).mean())


def collate(rows: list[dict[str, Any]], store: MatrixFeatureStore) -> dict[str, Any]:
    store.prepare_questions(rows)
    return {
        "rows": rows,
        "graphs": [store.graph(row) for row in rows],
        "questions": torch.stack([store.question(row) for row in rows]),
        "question_tokens": [store.question_tokens(row) for row in rows],
        "image_ids": [row["image_id"] for row in rows],
    }


@torch.no_grad()
def validation_loss(
    model: GraphMatrixModel,
    rows: list[dict[str, Any]],
    store: MatrixFeatureStore,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    loader = DataLoader(RowDataset(rows), batch_size=batch_size, shuffle=False, collate_fn=lambda values: collate(values, store))
    losses: list[float] = []
    for payload in tqdm(loader, desc="validation loss", leave=False, disable=os.environ.get("DISABLE_TQDM") == "1"):
        graphs = Batch.from_data_list(payload["graphs"]).to(device)
        questions = payload["questions"].to(device)
        retrieval = multipositive_loss(model.retrieval_logits(graphs, questions, payload["question_tokens"]), payload["image_ids"])
        candidate_rows = [store.candidates(row, graph) for row, graph in zip(payload["rows"], payload["graphs"])]
        logits_rows = model.vqa_logits_batch(graphs, questions, [values for _, values, _ in candidate_rows])
        answer_losses = [
            F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=device))
            for logits, (_, _, target) in zip(logits_rows, candidate_rows)
        ]
        losses.append(float(0.5 * retrieval + 0.5 * torch.stack(answer_losses).mean()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model: GraphMatrixModel, rows: list[dict[str, Any]], store: MatrixFeatureStore, device: torch.device, batch_size: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    unique_docs: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique_docs.setdefault(row["image_id"], row)
    docs = list(unique_docs.values())
    doc_graphs = [store.graph(row) for row in docs]
    questions = torch.stack([store.question(row) for row in rows]).to(device)
    q = F.normalize(model.q_proj(questions), dim=-1)
    similarities = []
    for start in range(0, len(doc_graphs), batch_size):
        batch = Batch.from_data_list(doc_graphs[start : start + batch_size]).to(device)
        doc_emb, _nodes = model.encode(batch)
        # Full validation uses pooled document embeddings. The late-interaction
        # objective remains active during training, while this avoids an
        # impractical O(questions * documents * nodes) Python loop.
        similarities.append((q @ doc_emb.T).cpu())
    similarity = torch.cat(similarities, dim=1).numpy()
    retrieval = retrieval_metrics(similarity, [row["image_id"] for row in rows], [row["image_id"] for row in docs])

    predictions = []
    loader = DataLoader(RowDataset(rows), batch_size=batch_size, shuffle=False, collate_fn=lambda values: collate(values, store))
    for payload in tqdm(loader, desc="VQA evaluation", disable=os.environ.get("DISABLE_TQDM") == "1"):
        graphs = Batch.from_data_list(payload["graphs"]).to(device)
        questions = payload["questions"].to(device)
        candidate_rows = [store.candidates(row, graph) for row, graph in zip(payload["rows"], payload["graphs"])]
        logits_rows = model.vqa_logits_batch(graphs, questions, [values for _, values, _ in candidate_rows])
        for row, logits, (values, _, _) in zip(payload["rows"], logits_rows, candidate_rows):
            prediction = values[int(logits.argmax())] if values else ""
            predictions.append(vqa_row(store.dataset, row, prediction))
    score = float(np.mean([row["score"] for row in predictions])) if predictions else 0.0
    exact = float(np.mean([row["exact"] for row in predictions])) if predictions else 0.0
    return {"retrieval": retrieval, "vqa": {"metric": "accuracy" if store.dataset == "ai2d" else "anls", "score": score, "exact_accuracy": exact}}, predictions


def main() -> None:
    global EXTERNAL_ROOT
    parser = argparse.ArgumentParser(description="Train local models in the controlled 11x3 matrix.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=TRAINABLE_LOCAL_MODELS, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="val: train/select on validation, then evaluate test; test: evaluate a saved validation checkpoint only")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/model_matrix")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent,
                        help="Root containing dataset manifests, models and model_matrix_cache")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    EXTERNAL_ROOT = args.data_root.resolve()
    result = run_local_experiment_config(args)
    if args.split == "test":
        print(json.dumps(result, ensure_ascii=False, indent=2))


def run_local_experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.model in {"clip", "siglip"}:
        raise NotImplementedError("Frozen CLIP/SigLIP feature runner is provided by train_model_matrix_vision.py")

    rows = load_rows(args.dataset, EXTERNAL_ROOT)
    run_dir, test_dir = training_paths(args)
    if args.split == "test":
        return evaluate_local_test(args, rows)
    config = training_config(args, rows, input_root=EXTERNAL_ROOT, backend="graph", epochs=args.epochs,
                             early_stopping_patience=args.early_stopping_patience,
                             batch_size=args.batch_size, eval_batch_size=args.eval_batch_size)
    validate_directory(run_dir, config, resume=args.resume)
    train_rows = split_rows(rows, "train", args.max_samples)
    eval_rows = split_rows(rows, "val", args.max_samples)
    if not train_rows or not eval_rows or not split_rows(rows, "test", args.max_samples):
        raise ValueError("Training requires nonempty train, val and final test splits")
    last, history = validate_partial_checkpoint(run_dir, config, load_torch)
    metrics_path = run_dir / "metrics.json"
    existing = completed_metrics(run_dir, config, eval_rows, trained=True)
    if existing is not None:
        existing["test"] = evaluate_local_test(args, rows)
        print(json.dumps(existing, ensure_ascii=False, indent=2))
        return existing
    if test_dir.exists() and any(test_dir.iterdir()):
        raise ValueError(f"Test artifacts exist without a completed matching training run: {test_dir}")
    ensure_free_space(run_dir)
    write_config(run_dir, config)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    store = MatrixFeatureStore(args.dataset, args.model, EXTERNAL_ROOT / "model_matrix_cache", device)
    store.preload_graphs(train_rows + eval_rows)
    # Encode every question and its token sequence once in large internal
    # batches.  Without this warm-up, the first epoch repeatedly invokes the
    # sentence encoder from the DataLoader collate function and can be several
    # times slower, especially after resuming from a checkpoint.
    store.prepare_questions(train_rows + eval_rows)
    model = GraphMatrixModel(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    loader = DataLoader(RowDataset(train_rows), batch_size=args.batch_size, shuffle=True, collate_fn=lambda values: collate(values, store))
    history_path = run_dir / "history.json"
    last_checkpoint_path = run_dir / "checkpoint_last.pt"
    best_composite = -math.inf
    started = time.perf_counter()

    start_epoch = 1
    epochs_without_improvement = 0
    early_stopped = False
    if last is not None:
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        start_epoch = int(last["epoch"]) + 1
        restore_rng_state(last["rng_state"])
        if history:
            best_composite = max(float(row["composite"]) for row in history)
            best_index = max(range(len(history)), key=lambda index: float(history[index]["composite"]))
            epochs_without_improvement = len(history) - best_index - 1

    epochs = min(args.epochs, 2) if args.max_samples is not None else args.epochs
    early_stopped = args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience
    for epoch in range(start_epoch, start_epoch if early_stopped else epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for payload in tqdm(loader, desc=f"{args.dataset}/{args.model}/seed{args.seed} epoch {epoch}/{epochs}", disable=os.environ.get("DISABLE_TQDM") == "1"):
            graphs = Batch.from_data_list(payload["graphs"]).to(device)
            questions = payload["questions"].to(device)
            retrieval_logits = model.retrieval_logits(graphs, questions, payload["question_tokens"])
            retrieval_loss = multipositive_loss(retrieval_logits, payload["image_ids"])
            candidate_rows = [store.candidates(row, graph) for row, graph in zip(payload["rows"], payload["graphs"])]
            logits_rows = model.vqa_logits_batch(graphs, questions, [values for _, values, _ in candidate_rows])
            vqa_losses = [
                F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=device))
                for logits, (_, _, target) in zip(logits_rows, candidate_rows)
            ]
            vqa_loss = torch.stack(vqa_losses).mean()
            loss = 0.5 * retrieval_loss + 0.5 * vqa_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
        evaluation, _ = evaluate(model, eval_rows, store, device, args.eval_batch_size)
        val_loss = validation_loss(model, eval_rows, store, device, args.eval_batch_size)
        mean_r1 = float(evaluation["retrieval"]["mean_recall_at_k"]["1"])
        vqa_score = float(evaluation["vqa"]["score"])
        composite = 0.5 * mean_r1 + 0.5 * vqa_score
        history.append({"epoch": epoch, "loss": total_loss / max(1, steps), "val_loss": val_loss, "composite": composite, **evaluation})
        run_dir.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        if composite > best_composite:
            best_composite = composite
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "composite": composite,
                        "config_sha256": digest(config)}, run_dir / "checkpoint_best.pt")
        else:
            epochs_without_improvement += 1
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
             "config_sha256": digest(config), "history_sha256": digest(history), "rng_state": capture_rng_state()},
            last_checkpoint_path,
        )
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            early_stopped = True
            print(
                f"[EARLY STOP] {args.dataset}/{args.model}/seed{args.seed}: "
                f"no composite improvement for {epochs_without_improvement} epochs",
                flush=True,
            )
            break

    best = load_torch(run_dir / "checkpoint_best.pt")
    model.load_state_dict(best["model"])
    evaluation, predictions = evaluate(model, eval_rows, store, device, args.eval_batch_size)
    elapsed = time.perf_counter() - started
    metrics = {
        "dataset": args.dataset,
        "model": args.model,
        "split": args.split,
        "seed": args.seed,
        "status": "completed_full" if args.max_samples is None and (len(history) == args.epochs or early_stopped) else "completed_smoke",
        "num_samples": len(eval_rows),
        "epochs_completed": len(history),
        "best_epoch": int(best["epoch"]),
        "best_composite": float(best["composite"]),
        "config_sha256": digest(config),
        "checkpoint_sha256": file_digest(run_dir / "checkpoint_best.pt"),
        "history": history,
        "training_execution": {
            "mode": "full" if args.max_samples is None else "smoke",
            "full_training_was_run": args.max_samples is None,
            "early_stopped": early_stopped,
            "early_stopping_patience": args.early_stopping_patience,
        },
        **evaluation,
        "resources": {
            "wall_time_seconds": elapsed,
            "latency_ms_per_question": elapsed * 1000 / max(1, len(eval_rows)),
            "throughput_questions_per_second": len(eval_rows) / elapsed if elapsed else None,
            "peak_ram_mb": psutil.Process().memory_info().rss / (1024 * 1024),
            "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "config": vars(args) | {"output_dir": str(args.output_dir)},
    }
    write_jsonl(predictions, run_dir / "predictions.jsonl")
    metrics["predictions_sha256"] = file_digest(run_dir / "predictions.jsonl")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    metrics["test"] = evaluate_local_test(args, rows, model=model, store=store, device=device)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    return metrics


def evaluate_local_test(args, rows, *, model=None, store=None, device=None):
    run_dir, test_dir = training_paths(args)
    saved_config = checkpoint_config(args, rows, run_dir, input_root=EXTERNAL_ROOT)
    validation = completed_metrics(run_dir, saved_config, split_rows(rows, "val", args.max_samples), trained=True)
    if validation is None:
        raise ValueError("--split test needs a completed validation-selected checkpoint; train with --split val first")
    checkpoint = run_dir / "checkpoint_best.pt"
    config = test_config(args, rows, saved_config, checkpoint, input_root=EXTERNAL_ROOT)
    validate_directory(test_dir, config, resume=args.resume)
    test_rows = split_rows(rows, "test", args.max_samples)
    if not test_rows:
        raise ValueError("The test split is empty")
    existing = completed_metrics(test_dir, config, test_rows)
    if existing is not None:
        return existing
    best = load_torch(checkpoint)
    if best.get("config_sha256") != digest(saved_config) or best.get("epoch") != validation["best_epoch"]:
        raise ValueError("Checkpoint does not match the completed validation run")
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        store = MatrixFeatureStore(args.dataset, args.model, EXTERNAL_ROOT / "model_matrix_cache", device)
        model = GraphMatrixModel(args.model).to(device)
        model.load_state_dict(best["model"])
    evaluation, predictions = evaluate(model, test_rows, store, device, args.eval_batch_size)
    result = {"dataset": args.dataset, "model": args.model, "split": "test", "seed": args.seed,
              "status": "available_full" if args.max_samples is None else "available_smoke",
              "num_samples": len(test_rows), "config_sha256": digest(config),
              "checkpoint_sha256": config["checkpoint_sha256"], "checkpoint_selection_split": "val",
              "best_epoch": validation["best_epoch"], **evaluation}
    write_config(test_dir, config)
    write_jsonl(predictions, test_dir / "predictions.jsonl")
    result["predictions_sha256"] = file_digest(test_dir / "predictions.jsonl")
    if args.dataset != "ai2d":
        submission = [{"questionId": row["question_id"], "answer": row["pred_answer"]} for row in predictions]
        (test_dir / "submission.json").write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    (test_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def train_local_experiment(
    *,
    dataset: str,
    model: str,
    run_dir: Path,
    epochs: int = 40,
    seed: int = 42,
    batch_size: int = 8,
    eval_batch_size: int = 16,
    max_samples: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    args = argparse.Namespace(
        dataset=dataset,
        model=model,
        split="val",
        seed=seed,
        max_samples=max_samples,
        output_dir=run_dir.parent,
        run_dir_override=run_dir,
        resume=resume,
        epochs=epochs,
        early_stopping_patience=0,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
    )
    return run_local_experiment_config(args)


if __name__ == "__main__":
    main()
