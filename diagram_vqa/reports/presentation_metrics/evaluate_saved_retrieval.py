"""Recompute full-test retrieval from notebook architectures and saved weights."""

from __future__ import annotations

import ast
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent
ARCHITECTURES = (
    "sam2_dinov2_dual_branch_evidence_graph",
    "sam2_dinov2_heterogeneous_balanced_evidence_graph",
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCHEMA = "sam2_dinov2_sparse_regularized_v1"


def notebook_model(dataset, architecture):
    path = ROOT / "notebooks/experiments" / dataset / f"{dataset}_{architecture}.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"nn": nn, "F": F, "torch": torch, "math": math,
                 "TEXT_DIM": 384, "VISUAL_DIM": 768, "GEOMETRY_DIM": 10}
    # Import literal hyperparameters and model definitions without executing training cells.
    for node in ast.parse("".join(notebook["cells"][1]["source"])).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = value
    body = ast.parse("".join(notebook["cells"][12]["source"])).body
    definitions = [node for node in body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["DualBranchEvidenceGraph"]().to(DEVICE).eval()


def rank_metrics(similarity, positive_indices):
    ranking = np.argsort(-similarity, axis=1, kind="stable")
    ranks = np.asarray([
        next(rank for rank, index in enumerate(order, 1) if int(index) in positive)
        for order, positive in zip(ranking, positive_indices, strict=True)
    ])
    return {"recall_at_k": {str(k): float(np.mean(ranks <= k)) for k in (1, 5, 10)},
            "mrr": float(np.mean(1.0 / ranks))}


@torch.inference_mode()
def evaluate_dataset(dataset, text_encoder):
    manifest = DATA / dataset / ("model_matrix_v1" if dataset == "ai2d" else "prepared_v1") / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["split"] == "test"]
    ids = [str(row["image_id"]) for row in rows]
    doc_ids = list(dict.fromkeys(ids))
    doc_index = {value: index for index, value in enumerate(doc_ids)}
    q_positive = [{doc_index[value]} for value in ids]
    d_positive = [set() for _ in doc_ids]
    for index, value in enumerate(ids):
        d_positive[doc_index[value]].add(index)
    if dataset == "ai2d":
        cache = ROOT / "runs/ai2d/sam2_dinov2_regularized_sparse_learned_graph/seed42_full" / f"node_features_{SCHEMA}"
    else:
        cache = DATA / "model_matrix_cache/sam2_dinov2_features" / dataset / f"node_features_{SCHEMA}"
    features = []
    for image_id in tqdm(doc_ids, desc=f"{dataset} cached test documents"):
        payload = torch.load(cache / f"{image_id}.pt", map_location="cpu", weights_only=False)
        assert payload["schema_version"] == SCHEMA
        x = payload["x"].float()
        assert x.ndim == 2 and x.shape[1] == 778 and len(x) > 0 and torch.isfinite(x).all()
        features.append(x)
    questions = text_encoder.encode([row["question"] for row in rows], batch_size=128,
                                    normalize_embeddings=True, convert_to_tensor=True,
                                    show_progress_bar=True).float().to(DEVICE)
    for architecture in ARCHITECTURES:
        model = notebook_model(dataset, architecture)
        for seed in (42, 43, 44):
            run = ROOT / "runs" / dataset / architecture / f"seed{seed}_full"
            if not (run / "metrics.json").exists():
                continue
            saved = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            assert saved["status"] in ("completed_full", "completed_early_stopped")
            checkpoint_path = run / "checkpoint_best.pt"
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            assert checkpoint["epoch"] == saved["best_epoch"]
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            documents = []
            for start in tqdm(range(0, len(features), 32), desc=f"{dataset}/{architecture}/seed{seed}"):
                values = features[start:start + 32]
                max_nodes = max(len(x) for x in values)
                nodes = torch.zeros(len(values), max_nodes, 778, device=DEVICE)
                mask = torch.zeros(len(values), max_nodes, dtype=torch.bool, device=DEVICE)
                for index, value in enumerate(values):
                    nodes[index, :len(value)] = value.to(DEVICE)
                    mask[index, :len(value)] = True
                x, is_ocr = model.project_nodes(nodes)
                centers = nodes[..., 768:770]
                if architecture == ARCHITECTURES[1]:
                    image_nodes, _ = model.image_layer(x, centers, mask, is_ocr)
                else:
                    image_nodes, _ = model.image_layer(x, centers, mask)
                encoded = F.normalize(model.pool(image_nodes, mask, model.document_gate), dim=-1)
                documents.append(encoded.cpu())
                if start == 0:
                    # Verify the optimized retrieval path against the model's complete forward.
                    q = questions[:len(values)]
                    options = torch.zeros(len(values), 1, 384, device=DEVICE)
                    full = model(nodes, mask, q, options)
                    torch.testing.assert_close(full["documents"], encoded)
            docs = torch.cat(documents)
            projected_q = F.normalize(model.question(questions), dim=-1).cpu()
            similarities = (projected_q @ docs.T).numpy()
            q2d = rank_metrics(similarities, q_positive)
            d2q = rank_metrics(similarities.T, d_positive)
            retrieval = {"question_to_document": q2d, "document_to_question": d2q,
                         "mean_recall_at_k": {str(k): (q2d["recall_at_k"][str(k)] + d2q["recall_at_k"][str(k)]) / 2 for k in (1, 5, 10)}}
            report = {"dataset": dataset, "architecture": architecture, "seed": seed,
                      "status": "evaluated_full_test", "num_samples": len(rows),
                      "num_documents": len(doc_ids), "best_epoch": checkpoint["epoch"],
                      "protocol": "bidirectional_mean_unique_documents_v1",
                      "tie_break": "stable original candidate order", "retrieval": retrieval,
                      "checkpoint": str(checkpoint_path),
                      "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}
            output = run / "retrieval_unique_test.json"
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({"dataset": dataset, "architecture": architecture, "seed": seed,
                              "mean_recall_at_k": retrieval["mean_recall_at_k"]}), flush=True)
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


def main():
    torch.set_num_threads(4)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(DEVICE), local_files_only=True)
    for dataset in ("ai2d", "docvqa", "infographicvqa"):
        evaluate_dataset(dataset, encoder)
    print("PRESENTATION_RETRIEVAL_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
