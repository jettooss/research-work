from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ai2d", "infographicvqa", "docvqa")
MODELS = (
    ("00", "random", "Random baseline"),
    ("01", "clip", "CLIP frozen backbone + trained heads"),
    ("02", "siglip", "SigLIP frozen backbone + trained heads"),
    ("03", "ocr_text", "OCR text retrieval"),
    ("04", "gatv2_knn", "OCR+kNN GATv2"),
    ("05", "graph_transformer", "Graph Transformer"),
    ("06", "sam2_graph_transformer", "SAM2 + Graph Transformer"),
    ("07", "hybrid_gatv2_knn", "Hybrid GATv2+kNN"),
    ("08", "graphcolbert", "GraphColBERT"),
    ("09", "graphcolbert_film", "GraphColBERT+FiLM"),
    ("10", "qwen25_vl_qlora", "Qwen2.5-VL-3B QLoRA"),
)


def cell(cell_type: str, source: str) -> dict:
    payload = {
        "cell_type": cell_type,
        "id": hashlib.sha1(f"{cell_type}:{source}".encode("utf-8")).hexdigest()[:12],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        payload.update({"execution_count": None, "outputs": []})
    return payload


def notebook(dataset: str, model: str, title: str) -> dict:
    seeds = [42]
    source = f'''from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
while not (ROOT / "scripts" / "run_model_matrix.py").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("diagram_vqa root not found")
    ROOT = ROOT.parent

DATASET = "{dataset}"
MODEL = "{model}"
SEEDS = {seeds!r}
MAX_SAMPLES = int(os.environ["MODEL_MATRIX_MAX_SAMPLES"]) if os.environ.get("MODEL_MATRIX_MAX_SAMPLES") else None
OUTPUT_DIR = ROOT / "runs" / ("model_matrix_smoke" if MAX_SAMPLES else "model_matrix")
print({{"dataset": DATASET, "model": MODEL, "seeds": SEEDS, "max_samples": MAX_SAMPLES, "output_dir": str(OUTPUT_DIR)}})
'''
    run = '''for seed in SEEDS:
    command = [sys.executable, str(ROOT / "scripts" / "run_model_matrix.py"),
               "--dataset", DATASET, "--model", MODEL, "--split", "val",
               "--seed", str(seed), "--output-dir", str(OUTPUT_DIR), "--resume"]
    if MAX_SAMPLES is not None:
        command += ["--max-samples", str(MAX_SAMPLES)]
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
'''
    report = '''rows = []
for seed in SEEDS:
    path = OUTPUT_DIR / DATASET / MODEL / f"seed{seed}" / "val" / "metrics.json"
    if path.exists():
        rows.append(json.loads(path.read_text(encoding="utf-8")))
assert len(rows) == len(SEEDS), f"missing metrics: {len(rows)}/{len(SEEDS)}"
summary = {
    "dataset": DATASET,
    "model": MODEL,
    "runs": len(rows),
    "statuses": [row["status"] for row in rows],
    "vqa_scores": [row.get("vqa", {}).get("score") for row in rows],
    "mean_r1": [row.get("retrieval", {}).get("mean_recall_at_k", {}).get("1") for row in rows],
    "resources": [row.get("resources", {}) for row in rows],
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
'''
    return {
        "cells": [
            cell("markdown", f"# {title} — {dataset}\n\nОтдельный воспроизводимый эксперимент матрицы 11×3. Retrieval и VQA публикуются раздельно."),
            cell("code", source),
            cell("markdown", "## Manifest audit и конфигурация"),
            cell("code", '''manifest_map = {
    "ai2d": ROOT.parent / "ai2d/model_matrix_v1/audit.json",
    "infographicvqa": ROOT.parent / "diagram_vqa/reports/infographicvqa_defense_audit.json",
    "docvqa": ROOT.parent / "diagram_vqa/reports/docvqa_defense_audit.json",
}
audit = json.loads(manifest_map[DATASET].read_text(encoding="utf-8"))
print(json.dumps(audit.get("dataset", audit), ensure_ascii=False, indent=2)[:12000])
'''),
            cell("markdown", "## Обучение / оценка"),
            cell("code", run),
            cell("markdown", "## Метрики, ресурсы и статусы"),
            cell("code", report),
            cell("markdown", "## Ошибки\n\nПодробные predictions и error slices находятся в каталоге запуска, указанном выше."),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "model_matrix": {"dataset": dataset, "model": model, "seeds": seeds},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 33 thin model-matrix notebooks.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "notebooks/model_matrix")
    args = parser.parse_args()
    written = []
    for dataset in DATASETS:
        directory = args.output_root / dataset
        directory.mkdir(parents=True, exist_ok=True)
        for prefix, model, title in MODELS:
            path = directory / f"{prefix}_{model}.ipynb"
            path.write_text(json.dumps(notebook(dataset, model, title), ensure_ascii=False, indent=1), encoding="utf-8")
            written.append(str(path.resolve()))
    index = {"datasets": list(DATASETS), "models": [model for _, model, _ in MODELS], "source_notebooks": written, "count": len(written)}
    (args.output_root / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(written), "output_root": str(args.output_root.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
