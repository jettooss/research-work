from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit reproducibility and completeness of the DocVQA experiment.")
    parser.add_argument("--protocol", type=Path, default=Path("experiments/docvqa_defense_protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/docvqa_defense_audit.json"))
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    base = protocol_path.parent
    split_path = (base / protocol["split_file"]).resolve()
    split = read_json(split_path)

    def audit_model(model: dict[str, Any], family: str) -> dict[str, Any]:
        path = (base / model["metrics_file"]).resolve()
        row: dict[str, Any] = {"name": model["name"], "metric_family": family, "metrics_file": str(path), "metrics_exists": path.exists()}
        if not path.exists():
            row["complete_for_defense"] = False
            return row
        metrics = read_json(path)
        row["status"] = metrics.get("status")
        if family == "open_answer":
            required = ["anls", "accuracy", *protocol["required_resource_metrics"]]
        elif model["name"] == "OCR graph ablations":
            required = ["variants"]
        else:
            required = ["question_to_document", "document_to_question", "mean_recall_at_k", *protocol["required_resource_metrics"]]
        row["missing_required_metrics"] = [key for key in required if metrics.get(key) is None]
        row["complete_for_defense"] = not row["missing_required_metrics"] and metrics.get("status") == "available_full"
        if model["name"] == "OCR graph ablations":
            variants = {item.get("variant"): item for item in metrics.get("variants", [])}
            row["variants"] = [
                {
                    "variant": variant,
                    "seed_count": variants.get(variant, {}).get("seed_count", 0),
                    "complete": variants.get(variant, {}).get("status") == "available_full" and variants.get(variant, {}).get("seed_count") == len(protocol["seeds"]),
                }
                for variant in protocol["ablation_matrix"]
            ]
            row["complete_for_defense"] = row["complete_for_defense"] and all(item["complete"] for item in row["variants"])
        return row

    models = [audit_model(model, "open_answer") for model in protocol["open_answer_models"]]
    models += [audit_model(model, "retrieval") for model in protocol["retrieval_models"]]
    submission_path = (base / protocol["test_submission"]).resolve()
    submission_count = None
    unique_question_ids = None
    if submission_path.exists():
        submission = read_json(submission_path)
        submission_count = len(submission)
        unique_question_ids = len({str(item.get("questionId")) for item in submission})
    expected_test = int(split["splits"]["test"]["questions"])
    data_complete = all(
        payload.get("missing_images") == 0 and payload.get("missing_ocr") == 0
        for payload in split["splits"].values()
    )
    leakage_free = not any(split.get("image_leakage", {}).values()) and not any(split.get("document_page_leakage", {}).values())
    submission_complete = submission_count == expected_test and unique_question_ids == expected_test
    audit = {
        "research_question": protocol["research_question"],
        "dataset": {
            "manifest": str((base / protocol["dataset_manifest"]).resolve()),
            "split_file": str(split_path),
            "splits": {name: {key: payload[key] for key in ("questions", "images", "missing_images", "missing_ocr", "questions_without_answers")} for name, payload in split["splits"].items()},
            "image_leakage": split.get("image_leakage", {}),
            "document_page_leakage": split.get("document_page_leakage", {}),
            "source_document_overlap_counts": {key: len(value) for key, value in split.get("source_document_overlap", {}).items()},
        },
        "models": models,
        "test_submission": {"path": str(submission_path), "exists": submission_path.exists(), "rows": submission_count, "unique_question_ids": unique_question_ids, "expected_rows": expected_test, "complete": submission_complete},
        "legacy_result": {"path": str((base / "../../runs/mlflow_checkpoints/docvqa-auto/summary.json").resolve()), "mean_r_at_1": 0.0415610745549202, "included_in_primary_table": False},
        "defense_ready": data_complete and leakage_free and submission_complete and all(row["complete_for_defense"] for row in models),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
