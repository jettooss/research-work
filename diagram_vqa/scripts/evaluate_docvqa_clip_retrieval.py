from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import clip
import psutil
import torch
from PIL import Image
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vqa_retrieval.metrics import mean_reciprocal_rank_multi_positive, recall_at_k_multi_positive  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CLIP question/document retrieval on DocVQA.")
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "docvqa" / "prepared_v1" / "manifest.jsonl")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "docvqa_clip_retrieval")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [row for row in load_jsonl(args.manifest) if row.get("split") == args.split]
    if args.max_samples:
        rows = rows[: args.max_samples]
    unique_documents: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique_documents.setdefault(str(row["image_id"]), row)
    documents = list(unique_documents.values())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, preprocess = clip.load(args.model, device=device)
    model.eval()
    image_embeddings = []
    with torch.no_grad():
        for start in tqdm(range(0, len(documents), args.batch_size), desc="CLIP DocVQA images"):
            batch = torch.stack([preprocess(Image.open(row["image_path"]).convert("RGB")) for row in documents[start : start + args.batch_size]]).to(device)
            image_embeddings.append(torch.nn.functional.normalize(model.encode_image(batch), dim=1).cpu())
        text_embeddings = []
        for start in tqdm(range(0, len(rows), args.batch_size * 2), desc="CLIP DocVQA questions"):
            tokens = clip.tokenize([str(row["question"]) for row in rows[start : start + args.batch_size * 2]], truncate=True).to(device)
            text_embeddings.append(torch.nn.functional.normalize(model.encode_text(tokens), dim=1).cpu())
    images = torch.cat(image_embeddings)
    questions = torch.cat(text_embeddings)
    similarities = (questions @ images.T).tolist()
    document_index = {str(row["image_id"]): idx for idx, row in enumerate(documents)}
    q2d_positive = [{document_index[str(row["image_id"])]} for row in rows]
    questions_by_document: dict[str, set[int]] = defaultdict(set)
    for idx, row in enumerate(rows):
        questions_by_document[str(row["image_id"])].add(idx)
    d2q_positive = [questions_by_document[str(row["image_id"])] for row in documents]
    transposed = list(map(list, zip(*similarities)))
    q2d = recall_at_k_multi_positive(similarities, q2d_positive)
    d2q = recall_at_k_multi_positive(transposed, d2q_positive)
    elapsed = time.perf_counter() - started
    metrics = {
        "dataset": "docvqa",
        "mode": "clip_retrieval",
        "model": args.model,
        "metric_family": "retrieval",
        "split": args.split,
        "status": "available_full" if args.max_samples is None else "available_partial",
        "num_samples": len(rows),
        "num_documents": len(documents),
        "question_to_document": {"recall_at_k": {str(k): v for k, v in q2d.items()}, "mrr": mean_reciprocal_rank_multi_positive(similarities, q2d_positive)},
        "document_to_question": {"recall_at_k": {str(k): v for k, v in d2q.items()}, "mrr": mean_reciprocal_rank_multi_positive(transposed, d2q_positive)},
        "mean_recall_at_k": {str(k): (q2d[k] + d2q[k]) / 2 for k in q2d},
        "wall_time_seconds": elapsed,
        "latency_ms_per_question": elapsed * 1000 / max(1, len(rows)),
        "throughput_questions_per_second": len(rows) / elapsed if elapsed else None,
        "peak_ram_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    out_dir = args.output_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
