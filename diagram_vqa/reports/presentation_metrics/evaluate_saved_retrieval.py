"""Recompute full-test retrieval from notebook architectures and saved weights."""

from __future__ import annotations

import ast
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent
ARCHITECTURES = (
    "sam2_dinov2_dual_branch_evidence_graph",
    "sam2_dinov2_heterogeneous_balanced_evidence_graph",
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCHEMA = "sam2_dinov2_sparse_regularized_v1"


def ensure_new_outputs(output_root, datasets, architectures, seeds):
    for dataset in datasets:
        for architecture in architectures:
            for seed in seeds:
                target = Path(output_root) / dataset / architecture / f"seed{seed}_full/retrieval_unique_test.json"
                if target.exists():
                    raise FileExistsError(f"Refusing to replace retrieval output: {target}; choose a new --output-dir")


def notebook_model(dataset, architecture, *, project_root=ROOT, device=DEVICE):
    path = Path(project_root) / "notebooks/experiments" / dataset / f"{dataset}_{architecture}.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"nn": nn, "F": F, "torch": torch, "math": math,
                 "TEXT_DIM": 384, "VISUAL_DIM": 768, "GEOMETRY_DIM": 10}
    # Scan code cells by AST; moving/inserting markdown or setup cells must not
    # change which architecture is loaded. Never execute dataset/training cells.
    available_definitions = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for node in ast.parse("".join(cell.get("source", []))).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                available_definitions[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        namespace[target.id] = value
    target = "DualBranchEvidenceGraph"
    if target not in available_definitions:
        raise ValueError(f"{path} does not define {target}")
    needed, pending = set(), [target]
    while pending:
        name = pending.pop()
        if name in needed:
            continue
        needed.add(name)
        pending.extend(node.id for node in ast.walk(available_definitions[name])
                       if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                       and node.id in available_definitions and node.id not in needed)
    definitions = [node for name, node in available_definitions.items() if name in needed]
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[target]().to(device).eval()


def rank_metrics(similarity, positive_indices):
    ranking = np.argsort(-similarity, axis=1, kind="stable")
    ranks = np.asarray([
        next(rank for rank, index in enumerate(order, 1) if int(index) in positive)
        for order, positive in zip(ranking, positive_indices, strict=True)
    ])
    return {"recall_at_k": {str(k): float(np.mean(ranks <= k)) for k in (1, 5, 10)},
            "mrr": float(np.mean(1.0 / ranks))}


@torch.inference_mode()
def evaluate_dataset(dataset, text_encoder, *, project_root=ROOT, data_root=DATA,
                     runs_root=None, output_root, architectures=ARCHITECTURES,
                     seeds=(42, 43, 44), feature_cache=None, manifest=None, device=DEVICE):
    from tqdm import tqdm

    project_root, data_root, output_root = Path(project_root), Path(data_root), Path(output_root)
    runs_root = Path(runs_root) if runs_root is not None else project_root / "runs"
    manifest = Path(manifest) if manifest is not None else data_root / dataset / ("model_matrix_v1" if dataset == "ai2d" else "prepared_v1") / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["split"] == "test"]
    if not rows:
        raise ValueError(f"No test questions in {manifest}")
    ids = [str(row["image_id"]) for row in rows]
    doc_ids = list(dict.fromkeys(ids))
    doc_index = {value: index for index, value in enumerate(doc_ids)}
    q_positive = [{doc_index[value]} for value in ids]
    d_positive = [set() for _ in doc_ids]
    for index, value in enumerate(ids):
        d_positive[doc_index[value]].add(index)
    if feature_cache is not None:
        cache = Path(feature_cache)
    elif dataset == "ai2d":
        cache = runs_root / "ai2d/sam2_dinov2_regularized_sparse_learned_graph/seed42_full" / f"node_features_{SCHEMA}"
    else:
        cache = data_root / "model_matrix_cache/sam2_dinov2_features" / dataset / f"node_features_{SCHEMA}"
    ensure_new_outputs(output_root, (dataset,), architectures, seeds)
    for architecture in architectures:
        for seed in seeds:
            run = runs_root / dataset / architecture / f"seed{seed}_full"
            for name in ("metrics.json", "checkpoint_best.pt"):
                if not (run / name).is_file():
                    raise FileNotFoundError(run / name)
    features = []
    for image_id in tqdm(doc_ids, desc=f"{dataset} cached test documents"):
        payload = torch.load(cache / f"{image_id}.pt", map_location="cpu", weights_only=False)
        assert payload["schema_version"] == SCHEMA
        x = payload["x"].float()
        assert x.ndim == 2 and x.shape[1] == 778 and len(x) > 0 and torch.isfinite(x).all()
        features.append(x)
    questions = text_encoder.encode([row["question"] for row in rows], batch_size=128,
                                    normalize_embeddings=True, convert_to_tensor=True,
                                    show_progress_bar=True).float().to(device)
    for architecture in architectures:
        model = notebook_model(dataset, architecture, project_root=project_root, device=device)
        for seed in seeds:
            run = runs_root / dataset / architecture / f"seed{seed}_full"
            saved = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            assert saved["status"] in ("completed_full", "completed_early_stopped")
            checkpoint_path = run / "checkpoint_best.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            assert checkpoint["epoch"] == saved["best_epoch"]
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            documents = []
            for start in tqdm(range(0, len(features), 32), desc=f"{dataset}/{architecture}/seed{seed}"):
                values = features[start:start + 32]
                max_nodes = max(len(x) for x in values)
                nodes = torch.zeros(len(values), max_nodes, 778, device=device)
                mask = torch.zeros(len(values), max_nodes, dtype=torch.bool, device=device)
                for index, value in enumerate(values):
                    nodes[index, :len(value)] = value.to(device)
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
                    options = torch.zeros(len(values), 1, 384, device=device)
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
            output = output_root / dataset / architecture / f"seed{seed}_full/retrieval_unique_test.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
            print(json.dumps({"dataset": dataset, "architecture": architecture, "seed": seed,
                              "mean_recall_at_k": retrieval["mean_recall_at_k"]}), flush=True)
        del model
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT, help="diagram_vqa directory containing notebooks")
    parser.add_argument("--data-root", type=Path, help="Parent of ai2d/docvqa/infographicvqa; default: project root's parent")
    parser.add_argument("--runs-root", type=Path, help="Checkpoint tree; default: PROJECT_ROOT/runs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Separate output tree; existing retrieval JSON files are never replaced")
    parser.add_argument("--datasets", "--dataset", nargs="+", choices=("ai2d", "docvqa", "infographicvqa"), default=("ai2d", "docvqa", "infographicvqa"))
    parser.add_argument("--architectures", "--architecture", nargs="+", choices=ARCHITECTURES, default=ARCHITECTURES)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--feature-cache", type=Path, help="Exact node_features directory; requires a single dataset")
    parser.add_argument("--manifest", type=Path, help="Override manifest.jsonl; requires a single dataset")
    parser.add_argument("--text-encoder", default="sentence-transformers/all-MiniLM-L6-v2", help="Encoder used by the saved model: Hugging Face ID or local path")
    parser.add_argument("--local-files-only", action="store_true", help="Require an already downloaded/local text encoder")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if len(args.datasets) != 1 and (args.feature_cache is not None or args.manifest is not None):
        parser.error("--feature-cache and --manifest require exactly one dataset")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.datasets)) != len(args.datasets) or len(set(args.architectures)) != len(args.architectures):
        parser.error("dataset, architecture and seed selections must not contain duplicates")
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        ensure_new_outputs(args.output_dir, args.datasets, args.architectures, args.seeds)
    except FileExistsError as exc:
        parser.error(str(exc))
    device = DEVICE if args.device == "auto" else torch.device(args.device)
    torch.set_num_threads(args.threads)
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.text_encoder, device=str(device), local_files_only=args.local_files_only)
    for dataset in args.datasets:
        evaluate_dataset(dataset, encoder, project_root=args.project_root,
                         data_root=args.data_root or args.project_root.parent,
                         runs_root=args.runs_root, output_root=args.output_dir,
                         architectures=args.architectures, seeds=args.seeds,
                         feature_cache=args.feature_cache, manifest=args.manifest, device=device)
    print("PRESENTATION_RETRIEVAL_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
