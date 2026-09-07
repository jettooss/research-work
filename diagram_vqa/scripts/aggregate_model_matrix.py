from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ai2d", "infographicvqa", "docvqa")
MODELS = (
    "random", "clip", "siglip", "ocr_text", "gatv2_knn", "graph_transformer", "sam2_graph_transformer",
    "hybrid_gatv2_knn", "graphcolbert", "graphcolbert_film", "qwen25_vl_qlora",
)


def nested(payload: dict[str, Any], *keys: str) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the controlled 11x3 model matrix.")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs/model_matrix")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/model_matrix_aggregate.json")
    parser.add_argument("--allow-smoke", action="store_true", help="Aggregate available_smoke runs without marking them full.")
    args = parser.parse_args()
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            expected = [42]
            runs = []
            for seed in expected:
                path = args.runs_root / dataset / model / f"seed{seed}" / "val/metrics.json"
                if path.exists():
                    run = json.loads(path.read_text(encoding="utf-8"))
                    run["metrics_file"] = str(path.resolve())
                    runs.append(run)
            vqa = [value for run in runs if (value := nested(run, "vqa", "score")) is not None]
            r1 = [value for run in runs if (value := nested(run, "retrieval", "mean_recall_at_k", "1")) is not None]
            metrics_complete = len(vqa) == len(expected) and len(r1) == len(expected)
            all_full = all(run.get("status") == "available_full" for run in runs)
            all_smoke = all(run.get("status") in {"available_full", "available_smoke"} for run in runs)
            status = "available_full" if len(runs) == len(expected) and metrics_complete and all_full else (
                "available_smoke" if args.allow_smoke and len(runs) == len(expected) and metrics_complete and all_smoke else "incomplete"
            )
            rows.append({
                "dataset": dataset, "model": model, "expected_seeds": expected, "runs": runs,
                "status": status,
                "vqa_mean": statistics.mean(vqa) if vqa else None, "vqa_std": statistics.pstdev(vqa) if len(vqa) > 1 else 0.0 if vqa else None,
                "mean_r1_mean": statistics.mean(r1) if r1 else None, "mean_r1_std": statistics.pstdev(r1) if len(r1) > 1 else 0.0 if r1 else None,
            })
    payload = {"datasets": list(DATASETS), "models": list(MODELS), "cells": rows, "available_full": sum(row["status"] == "available_full" for row in rows), "available_smoke": sum(row["status"] in {"available_full", "available_smoke"} for row in rows), "expected": 33}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"available_full": payload["available_full"], "available_smoke": payload["available_smoke"], "expected": 33, "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
