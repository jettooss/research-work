from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(payload: Any, keys: list[str]) -> Any:
    for key in keys:
        payload = payload[key]
    return payload


def build_infographicvqa_defense_audit(protocol_path: Path) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    protocol = read_json(protocol_path)
    base = protocol_path.parent
    split_path = (base / protocol["split_file"]).resolve()
    split_payload = read_json(split_path)

    model_rows = []
    for model in protocol["models"]:
        item = {"name": model["name"], "role": model["role"], "status": model.get("status", "available")}
        metrics_rel = model.get("metrics_file")
        if metrics_rel:
            metrics_path = (base / metrics_rel).resolve()
            item["metrics_file"] = str(metrics_path)
            item["metrics_exists"] = metrics_path.exists()
            if metrics_path.exists() and model.get("metric_path"):
                metrics = read_json(metrics_path)
                required_metrics = model.get("required_metrics", protocol["required_metrics"])
                metric_keys = set(protocol["required_metrics"]) | set(required_metrics)
                item.update({key: metrics[key] for key in metric_keys if key in metrics})
                item.update(
                    {
                        key: metrics[key]
                        for key in (
                            "num_samples",
                            "max_samples",
                            "total_eval_rows",
                            "epochs_completed",
                            "metric_family",
                            "primary_metric",
                            "resource_notes",
                        )
                        if key in metrics
                    }
                )
                try:
                    item["score"] = float(nested(metrics, model["metric_path"]))
                except (KeyError, TypeError, ValueError):
                    item["score"] = None
                    item["score_missing"] = True
        checkpoint_rel = model.get("checkpoint")
        if checkpoint_rel:
            checkpoint = (base / checkpoint_rel).resolve()
            item["checkpoint"] = str(checkpoint)
            item["checkpoint_exists"] = checkpoint.exists()
        status = str(item.get("status", "")).lower()
        requires_rerun = any(token in status for token in ("partial", "smoke", "rerun_required"))
        required_metrics = model.get("required_metrics", protocol["required_metrics"])
        item["required_metrics"] = required_metrics
        item["missing_required_metrics"] = [
            metric for metric in required_metrics if metric not in item or item.get(metric) is None
        ]
        item["complete_for_defense"] = not item["missing_required_metrics"] and not requires_rerun
        model_rows.append(item)

    return {
        "research_question": protocol["research_question"],
        "dataset": {
            "manifest": str((base / protocol["dataset_manifest"]).resolve()),
            "split_file": str(split_path),
            "splits": {
                name: {
                    "questions": payload["questions"],
                    "images": payload["images"],
                    "missing_images": payload["missing_images"],
                }
                for name, payload in split_payload["splits"].items()
            },
            "image_leakage": split_payload.get("image_leakage", {}),
        },
        "models": model_rows,
        "required_metrics": protocol["required_metrics"],
        "ablation_matrix": protocol["ablation_matrix"],
        "defense_ready": not any(split_payload.get("image_leakage", {}).values()) and all(row["complete_for_defense"] for row in model_rows),
    }


def write_infographicvqa_defense_audit(protocol_path: Path, output_path: Path) -> dict[str, Any]:
    audit = build_infographicvqa_defense_audit(protocol_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducibility audit for InfographicVQA defense experiments.")
    parser.add_argument("--protocol", type=Path, default=Path("experiments/infographicvqa_defense_protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/infographicvqa_defense_audit.json"))
    args = parser.parse_args()

    audit = write_infographicvqa_defense_audit(args.protocol, args.output)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
