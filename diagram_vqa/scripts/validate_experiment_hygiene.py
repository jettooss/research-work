"""Validate current notebook sources and independently supplied run artifacts."""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vqa_retrieval.notebook_runs import notebook_paths
from vqa_retrieval.experiment_matrix import ARCHITECTURE_SPECS

REQUIRED_SECTIONS = ("Dataset Format", "Data Example", "Parsed Image", "Architecture", "Training And Validation", "Prediction", "Metrics")
PLACEHOLDERS = ("preflight_existing_metrics", "full_training_backend_pending", "fallback_preview_prediction")


def read_json(path: Path, failures: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        failures.append(f"Cannot read JSON {path}: {error}")
        return None


def notebook_stats(path: Path) -> dict[str, Any]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    code_cells = executed = outputs = 0
    sources = []
    rendered = []
    functions = set()
    variables = set()
    has_model_class = False
    for index, cell in enumerate(notebook["cells"]):
        value = cell.get("source", "")
        source = "".join(value) if isinstance(value, list) else value
        sources.append(source)
        if cell["cell_type"] != "code":
            continue
        code_cells += 1
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.add(node.name)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    variables.add(node.id)
                if isinstance(node, ast.ClassDef) and any(ast.unparse(base) == "nn.Module" for base in node.bases):
                    has_model_class = True
        except SyntaxError as error:
            errors.append(f"cell {index}: {error}")
        executed += cell.get("execution_count") is not None
        for output in cell.get("outputs", []):
            outputs += 1
            rendered.append(json.dumps(output, ensure_ascii=False))
            if output.get("output_type") == "error":
                errors.append(f"cell {index}: {output.get('ename')}: {output.get('evalue')}")
    source_text = "\n".join(sources)
    output_text = "\n".join(rendered)
    # The standalone notebooks use Russian headings. Check the same stages by
    # their actual definitions/outputs instead of requiring English titles.
    structural_stages = {
        "Dataset Format": "load_manifest" in functions,
        "Data Example": "format_table" in variables,
        "Parsed Image": {"draw_processing", "load_ocr_words"} <= functions,
        "Architecture": has_model_class,
        "Training And Validation": {"train_full", "evaluate_loader"} <= functions,
        "Prediction": "predict_one" in functions,
        "Metrics": "test_metrics" in variables,
    }
    return {"code_cells": code_cells, "executed_cells": executed, "outputs": outputs,
            "missing_sections": [name for name in REQUIRED_SECTIONS
                                 if name.lower() not in source_text.lower() and not structural_stages[name]],
            "placeholder_outputs": [name for name in PLACEHOLDERS if name in output_text], "errors": errors}


def _positive_integer(value) -> bool:
    return type(value) is int and value > 0


def _finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_run(run_dir: Path, dataset: str, architecture: str, seed: int) -> list[str]:
    failures: list[str] = []
    for name in ("metrics.json", "history.json", "checkpoint_best.pt"):
        if not (run_dir / name).is_file():
            failures.append(f"Missing run artifact (supply externally or regenerate): {run_dir / name}")
    if not (run_dir / "metrics.json").is_file():
        return failures
    metrics = read_json(run_dir / "metrics.json", failures)
    if not isinstance(metrics, dict):
        failures.append(f"metrics must be an object: {run_dir}")
        return failures
    for key, expected in (("dataset", dataset), ("architecture", architecture), ("seed", seed)):
        actual = metrics.get(key)
        if key == "architecture" and actual is None:
            backend = ARCHITECTURE_SPECS[architecture].backend_model
            if backend is not None and metrics.get("model") == backend:
                actual = architecture
        if actual != expected:
            failures.append(f"{key} mismatch: {run_dir}: expected {expected!r}, got {metrics.get(key)!r}")
    status = metrics.get("status")
    if status not in {"completed_full", "completed_early_stopped"}:
        failures.append(f"Run is not completed: {run_dir}: {status!r}")
    training = metrics.get("training_execution") or {}
    if not isinstance(training, dict):
        training = {}
    if metrics.get("full_training_was_run") is not True and training.get("full_training_was_run") is not True:
        failures.append(f"Missing full-data training evidence: {run_dir}")
    config = metrics.get("config") or {}
    if not isinstance(config, dict):
        config = {}
    for name in ("config.json", "notebook_run.json"):
        if (run_dir / name).is_file():
            extra = read_json(run_dir / name, failures)
            if isinstance(extra, dict):
                config = {**extra, **config}
    if metrics.get("max_samples") is not None or config.get("max_samples") is not None or config.get("full_data") is False:
        failures.append(f"Limited data run cannot establish full-data completion: {run_dir}")
    requested = metrics.get("epochs_requested", metrics.get("epochs", config.get("epochs")))
    completed = metrics.get("epochs_completed")
    best = metrics.get("best_epoch")
    if not _positive_integer(requested):
        failures.append(f"Missing positive requested epoch budget in metrics/config: {run_dir}")
    if not _positive_integer(completed):
        failures.append(f"Invalid epochs_completed: {run_dir}")
    elif _positive_integer(requested):
        if completed > requested or (status == "completed_full" and completed != requested):
            failures.append(f"Epoch count contradicts completion status/budget: {run_dir}")
        if status == "completed_early_stopped" and completed >= requested:
            failures.append(f"Early-stop status contradicts epoch budget: {run_dir}")
    if not _positive_integer(best) or (_positive_integer(completed) and best > completed):
        failures.append(f"Invalid selected best_epoch: {run_dir}")
    if (run_dir / "history.json").is_file():
        history = read_json(run_dir / "history.json", failures)
        if not isinstance(history, list) or not history:
            failures.append(f"History must contain training epochs: {run_dir}")
        else:
            if len(history) != completed:
                failures.append(f"History length does not match epochs_completed: {run_dir}")
            for index, epoch in enumerate(history, 1):
                if (not isinstance(epoch, dict) or epoch.get("epoch") != index
                        or not _finite_number(epoch.get("train_loss", epoch.get("loss")))
                        or not _finite_number(epoch.get("val_loss"))):
                    failures.append(f"Invalid real train/validation history row {index}: {run_dir}")
    test = metrics.get("test")
    if (run_dir / "test/metrics.json").is_file():
        test = read_json(run_dir / "test/metrics.json", failures)
    if not isinstance(test, dict) or not test:
        failures.append(f"Missing test evaluation in metrics or test/metrics.json: {run_dir}")
        test = {}
    predictions = test.get("predictions")
    for name in ("test/predictions.jsonl", "predictions.json", "predictions.jsonl"):
        path = run_dir / name
        if path.is_file():
            if path.suffix == ".jsonl":
                try:
                    predictions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                except (OSError, ValueError) as error:
                    failures.append(f"Invalid predictions {path}: {error}")
                    predictions = None
            else:
                predictions = read_json(path, failures)
            break
    if not isinstance(predictions, list) or not predictions or any(not isinstance(row, dict) for row in predictions):
        failures.append(f"Missing nonempty per-example predictions (external artifact required): {run_dir}")
    return failures


def validate_project(root: Path, runs_root: Path, seeds: list[int]) -> dict[str, Any]:
    failures: list[str] = []
    expected = notebook_paths(root)
    actual = set((root / "notebooks/experiments").rglob("*.ipynb"))
    for path in sorted(actual - set(expected)):
        failures.append(f"Unexpected active notebook: {path}")
    notebook_rows = []
    run_rows = []
    for path in expected:
        dataset = path.parent.name
        architecture = path.stem.removeprefix(dataset + "_")
        row: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if not path.is_file():
            failures.append(f"Missing active notebook: {path}")
        else:
            try:
                stats = notebook_stats(path)
                row["stats"] = stats
                for issue in stats["errors"]:
                    failures.append(f"Notebook error: {path}: {issue}")
                if stats["missing_sections"]:
                    failures.append(f"Missing notebook sections: {path}: {stats['missing_sections']}")
                if stats["placeholder_outputs"]:
                    failures.append(f"Placeholder notebook outputs: {path}: {stats['placeholder_outputs']}")
                if stats["code_cells"] == 0 or stats["executed_cells"] != stats["code_cells"]:
                    failures.append(f"Notebook has unexecuted code cells: {path}")
            except (OSError, ValueError, KeyError) as error:
                failures.append(f"Invalid notebook: {path}: {error}")
        notebook_rows.append(row)
        for seed in seeds:
            run_dir = runs_root / dataset / architecture / f"seed{seed}_full"
            run_failures = validate_run(run_dir, dataset, architecture, seed)
            failures.extend(run_failures)
            run_rows.append({"run_dir": str(run_dir), "status": "failed" if run_failures else "passed", "failures": run_failures})
    return {"status": "failed" if failures else "passed", "generated_at": datetime.now(timezone.utc).isoformat(),
            "project_root": str(root), "runs_root": str(runs_root), "seeds": seeds,
            "counts": {"notebooks": len(notebook_rows), "runs": len(run_rows), "failures": len(failures)},
            "notebooks": notebook_rows, "runs": run_rows, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--json-report", type=Path, default=Path("reports/experiment_hygiene_report.json"))
    parser.add_argument("--md-report", type=Path, default=Path("reports/experiment_hygiene_report.md"))
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be unique nonnegative integers")
    root = args.project_root.resolve()
    report = validate_project(root, (args.runs_root or root / "runs").resolve(), args.seeds)
    if not args.no_write_report:
        json_path = root / args.json_report
        md_path = root / args.md_report
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        lines = ["# Experiment validation", "", f"Status: **{report['status']}**", "",
                 f"Checked {len(report['notebooks'])} notebooks and {len(report['runs'])} runs.", "",
                 "Missing external artifacts remain failures; source inspection does not prove training reproducibility.", ""]
        lines += [f"- {failure}" for failure in report["failures"]] or ["- No validation failures."]
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "counts", "failures")}, ensure_ascii=False, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
