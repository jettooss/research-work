from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = (
    "random", "ocr_text", "clip", "siglip", "gatv2_knn", "graph_transformer",
    "sam2_graph_transformer", "hybrid_gatv2_knn", "graphcolbert", "graphcolbert_film", "qwen25_vl_qlora",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute and verify model-matrix notebooks sequentially.")
    parser.add_argument("--notebook-root", type=Path, default=ROOT / "notebooks/model_matrix")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dataset", choices=["ai2d", "infographicvqa", "docvqa"], default=None)
    parser.add_argument("--model", choices=MODEL_ORDER, default=None)
    parser.add_argument("--skip-qlora", action="store_true")
    parser.add_argument("--timeout", type=int, default=7 * 24 * 3600)
    args = parser.parse_args()

    previous = os.environ.get("MODEL_MATRIX_MAX_SAMPLES")
    if args.max_samples is None:
        os.environ.pop("MODEL_MATRIX_MAX_SAMPLES", None)
    else:
        os.environ["MODEL_MATRIX_MAX_SAMPLES"] = str(args.max_samples)
    try:
        notebooks = []
        datasets = [args.dataset] if args.dataset else ["ai2d", "infographicvqa", "docvqa"]
        for model in MODEL_ORDER:
            if args.model and model != args.model:
                continue
            if args.skip_qlora and model == "qwen25_vl_qlora":
                continue
            for dataset in datasets:
                matches = list((args.notebook_root / dataset).glob("*.ipynb"))
                notebooks.extend(
                    path
                    for path in matches
                    if not path.name.endswith(".executed.ipynb") and path.stem.split("_", 1)[-1] == model
                )
        for index, path in enumerate(notebooks, start=1):
            executed_path = path.with_name(path.stem + ".executed.ipynb")
            print(f"[NOTEBOOK {index}/{len(notebooks)}] {path}", flush=True)
            notebook = nbformat.read(path, as_version=4)
            client = NotebookClient(notebook, timeout=args.timeout, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
            client.execute(cwd=str(ROOT))
            nbformat.write(notebook, executed_path)
            print(f"[SAVED] {executed_path}", flush=True)
    finally:
        if previous is None:
            os.environ.pop("MODEL_MATRIX_MAX_SAMPLES", None)
        else:
            os.environ["MODEL_MATRIX_MAX_SAMPLES"] = previous


if __name__ == "__main__":
    main()
