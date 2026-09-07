from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vqa_retrieval.model_matrix import candidate_texts  # noqa: E402

from train_model_matrix_local import TEXT_DIM, knn_edges  # noqa: E402


def load_torch(path: Path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def graph_parts(payload: dict) -> tuple[torch.Tensor, list[str]]:
    source_x = payload["x"].float()
    ocr_nodes = int(payload.get("ocr_nodes", 0))
    sam_nodes = int(payload.get("sam_nodes", 0))
    ocr_count = min(80, ocr_nodes)
    sam_count = min(80, sam_nodes)

    parts = []
    if ocr_count:
        ocr_x = source_x[:ocr_count]
        parts.append(torch.cat([ocr_x[:, :TEXT_DIM], ocr_x[:, -10:]], dim=1))
    if sam_count:
        sam_x = source_x[ocr_nodes : ocr_nodes + sam_count]
        parts.append(torch.cat([torch.zeros((sam_count, TEXT_DIM)), sam_x[:, -10:]], dim=1))
    x = torch.cat(parts, dim=0) if parts else torch.zeros((1, TEXT_DIM + 10))

    texts = [str(value) for value in payload.get("ocr_texts", [])[:ocr_count]]
    candidates = candidate_texts([{"text": value} for value in texts], max_candidates=128) or [""]
    return x, candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["docvqa", "infographicvqa"], required=True)
    args = parser.parse_args()

    source_dir = (
        EXTERNAL_ROOT
        / "model_matrix_cache"
        / "sam2_dinov2_features"
        / args.dataset
        / "node_features_sam2_dinov2_sparse_regularized_v1"
    )
    target_dir = EXTERNAL_ROOT / "model_matrix_cache" / "graphs" / args.dataset / "sam2_graph_transformer"
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(source_dir.glob("*.pt"))
    missing = [path for path in paths if not (target_dir / path.name).exists()]
    print(f"{args.dataset}: source={len(paths)}, missing={len(missing)}", flush=True)
    if not missing:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    chunk_size = 64
    for start in tqdm(range(0, len(missing), chunk_size), desc=f"prebuild {args.dataset}"):
        chunk = missing[start : start + chunk_size]
        prepared = [(path, *graph_parts(load_torch(path))) for path in chunk]
        unique_candidates = list(dict.fromkeys(value for _, _, values in prepared for value in values))
        encoded = encoder.encode(
            unique_candidates,
            batch_size=256,
            convert_to_tensor=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).detach().cpu().float()
        embedding_by_text = dict(zip(unique_candidates, encoded))
        for path, x, candidates in prepared:
            candidate_embeddings = torch.stack([embedding_by_text[value] for value in candidates])
            graph = Data(x=x, edge_index=knn_edges(x), candidate_embeddings=candidate_embeddings)
            graph.candidate_texts = candidates
            torch.save(graph, target_dir / path.name)


if __name__ == "__main__":
    main()
