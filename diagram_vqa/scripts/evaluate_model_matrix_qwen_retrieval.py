from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
from qwen_vl_utils import process_vision_info  # noqa: E402
from vqa_retrieval.model_matrix import DATASETS, load_rows, retrieval_metrics, split_rows  # noqa: E402
from vqa_retrieval.vlm_service import QwenVlmRunner  # noqa: E402


class HiddenStateEncoder:
    def __init__(self, adapter_path: Path, cache_dir: Path, max_pixels: int) -> None:
        self.runner = QwenVlmRunner(
            model_path=EXTERNAL_ROOT / "models/Qwen2.5-VL-3B-Instruct",
            adapter_path=adapter_path,
            use_4bit=True,
            device_mode="balanced_low_0",
            max_memory={0: "5GiB", "cpu": "24GiB"},
            offload_folder=ROOT / "runs/model_matrix_qwen_retrieval_offload",
            low_cpu_mem_usage=True,
            max_pixels=max_pixels,
        )
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_pixels = max_pixels

    @torch.no_grad()
    def encode(self, key: str, text: str, image_path: str | None = None) -> torch.Tensor:
        import hashlib

        path = self.cache_dir / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.pt"
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True).float()
        content: list[dict[str, Any]] = []
        if image_path is not None:
            content.append({"type": "image", "image": str(Path(image_path).resolve()), "max_pixels": self.max_pixels})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        rendered = self.runner.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.runner.processor(
            text=[rendered], images=images or None, videos=videos or None,
            padding=True, return_tensors="pt",
        )
        device = next(self.runner.model.parameters()).device
        inputs = {name: value.to(device) if hasattr(value, "to") else value for name, value in inputs.items()}
        outputs = self.runner.model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden = outputs.hidden_states[-1].float()
        mask = inputs.get("attention_mask", torch.ones(hidden.shape[:2], device=hidden.device)).unsqueeze(-1)
        embedding = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).squeeze(0)
        embedding = F.normalize(embedding, dim=0).detach().cpu()
        torch.save(embedding.to(torch.float16), path)
        return embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen adapter hidden-state document retrieval.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=786432)
    args = parser.parse_args()
    evaluate_qwen_retrieval(
        dataset=args.dataset,
        split=args.split,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_pixels=args.max_pixels,
    )


def evaluate_qwen_retrieval(
    *,
    dataset: str,
    split: str,
    adapter_path: Path,
    output_dir: Path,
    max_samples: int | None = None,
    max_pixels: int = 786432,
) -> dict[str, Any]:
    rows = split_rows(load_rows(dataset, EXTERNAL_ROOT), split, max_samples)
    unique_docs: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique_docs.setdefault(row["image_id"], row)
    docs = list(unique_docs.values())
    encoder = HiddenStateEncoder(adapter_path, output_dir / "features", max_pixels)
    document_embeddings = torch.stack([
        encoder.encode(f"doc:{row['image_id']}", "Represent this document image for question retrieval.", row["image_path"])
        for row in docs
    ])
    question_embeddings = torch.stack([
        encoder.encode(f"question:{row['sample_id']}", f"Represent this document question for retrieval: {row['question']}")
        for row in rows
    ])
    similarity = (question_embeddings @ document_embeddings.T).numpy().astype(np.float32)
    metrics = retrieval_metrics(similarity, [row["image_id"] for row in rows], [row["image_id"] for row in docs])
    metrics.update({"dataset": dataset, "split": split, "num_samples": len(rows), "num_documents": len(docs)})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    main()
