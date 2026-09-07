from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def nested(payload: Any, keys: list[str]) -> Any:
    for key in keys:
        payload = payload[key]
    return payload


def question_type(text: str) -> str:
    value = text.lower()
    if any(token in value for token in ("how many", "number of", "percentage")):
        return "counting"
    if any(token in value for token in ("where", "above", "below", "left", "right", "inside")):
        return "spatial"
    if any(token in value for token in ("what happens", "would happen", "if ", "cause")):
        return "reasoning"
    if any(token in value for token in ("label", "letter", "represents", "indicated")):
        return "label/OCR"
    return "recognition/knowledge"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducibility audit for the AI2D defense.")
    parser.add_argument("--protocol", type=Path, default=Path("experiments/defense_protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/defense_audit.json"))
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    base = protocol_path.parent
    manifest_path = (base / protocol["dataset_manifest"]).resolve()
    split_path = (base / protocol["split_file"]).resolve()
    rows = read_jsonl(manifest_path)
    split = read_json(split_path)

    split_counts = Counter(str(row.get("split", "unknown")) for row in rows)
    split_images: dict[str, set[str]] = {}
    for row in rows:
        split_images.setdefault(str(row.get("split", "unknown")), set()).add(str(row.get("image_id", "")))
    leakage = {
        f"{a}-{b}": sorted(split_images.get(a, set()) & split_images.get(b, set()))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    qtypes = Counter(question_type(str(row.get("question", ""))) for row in rows if row.get("split") == "test")

    model_rows = []
    for model in protocol["models"]:
        item = {"name": model["name"], "role": model["role"], "status": model.get("status", "available")}
        metrics_rel = model.get("metrics_file")
        if metrics_rel:
            metrics_path = (base / metrics_rel).resolve()
            item["metrics_file"] = str(metrics_path)
            item["metrics_exists"] = metrics_path.exists()
            if metrics_path.exists() and model.get("accuracy_path"):
                item["accuracy"] = float(nested(read_json(metrics_path), model["accuracy_path"]))
        checkpoint_rel = model.get("checkpoint")
        if checkpoint_rel:
            checkpoint = (base / checkpoint_rel).resolve()
            item["checkpoint"] = str(checkpoint)
            item["checkpoint_exists"] = checkpoint.exists()
        item["complete_for_defense"] = all(metric in item for metric in protocol["required_metrics"])
        model_rows.append(item)

    audit = {
        "research_question": protocol["research_question"],
        "dataset": {
            "manifest": str(manifest_path),
            "split_file": str(split_path),
            "split_seed": split.get("seed"),
            "validation_ratio": split.get("val_ratio"),
            "images_total": len({str(row.get("image_id", "")) for row in rows}),
            "questions_total": len(rows),
            "questions_by_split": dict(sorted(split_counts.items())),
            "images_by_split": {key: len(value) for key, value in sorted(split_images.items())},
            "missing_official_test_ids": len(split.get("missing_test_ids", [])),
            "image_leakage": leakage,
            "test_question_types": dict(sorted(qtypes.items()))
        },
        "models": model_rows,
        "required_metrics": protocol["required_metrics"],
        "ablation_matrix": protocol["ablation_matrix"],
        "defense_ready": not any(leakage.values()) and all(row["complete_for_defense"] for row in model_rows)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
