from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from vqa_retrieval.experiment_matrix import ARCHITECTURES, ARCHITECTURE_SPECS, DATASETS


DEFAULT_JSON_REPORT = Path("reports/experiment_hygiene_report.json")
DEFAULT_MD_REPORT = Path("reports/experiment_hygiene_report.md")
NOTEBOOK_ROOT = Path("notebooks/experiments")
EXECUTED_SUFFIX = ".executed.ipynb"
IGNORED_DIR_NAMES = {".git", ".venv", "archive", "outputs", "runs"}
CACHE_DIR_NAMES = {".ipynb_checkpoints", "__pycache__", ".pytest_cache"}
REQUIRED_SECTIONS = (
    "Dataset Format",
    "Data Example",
    "Parsed Image",
    "Architecture",
    "Training And Validation",
    "Prediction",
    "Metrics",
)
FORBIDDEN_SOURCE_PATTERNS = (
    "full_commands",
    "benchmark_command.py",
    "canonical_experiments.json",
    "import sys",
    "sys.",
    "subprocess",
    "audit_infographicvqa_defense.py",
    "preflight_existing_metrics",
    "full_training_backend_pending",
    "fallback_preview_prediction",
    "_empty_history",
)
LEGACY_CANDIDATES = (
    {
        "path": "archive/wrong_single_dataset_notebooks_20260816",
        "reason": "wrong universal dataset notebooks archived out of the active path",
    },
    {
        "path": "archive/experiment_matrix_3x9_superseded_20260816",
        "reason": "previous generated matrix snapshot",
    },
    {
        "path": "notebooks/model_matrix",
        "reason": "legacy notebook root is not part of the active experiments tree",
    },
    {
        "path": "notebooks/model_matrix_clean",
        "reason": "legacy notebook root is not part of the active experiments tree",
    },
)


def read_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def notebook_stats(path: Path) -> dict[str, Any]:
    notebook = read_notebook(path)
    cells = notebook.get("cells", [])
    outputs = 0
    executed_cells = 0
    code_cells = 0
    text_parts: list[str] = []
    mimes: set[str] = set()
    errors: list[str] = []
    output_text: list[str] = []
    for cell in cells:
        text_parts.append(source_text(cell))
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        for output in cell.get("outputs", []):
            outputs += 1
            mimes.update((output.get("data") or {}).keys())
            output_text.append(json.dumps(output, ensure_ascii=False, default=str))
            if output.get("output_type") == "error":
                errors.append(f"{output.get('ename')}: {output.get('evalue')}")
        if cell.get("execution_count") is not None:
            executed_cells += 1
    text = "\n".join(text_parts)
    rendered_outputs = "\n".join(output_text)
    return {
        "cells": len(cells),
        "code_cells": code_cells,
        "outputs": outputs,
        "executed_cells": executed_cells,
        "sections": [section for section in REQUIRED_SECTIONS if section in text],
        "missing_sections": [section for section in REQUIRED_SECTIONS if section not in text],
        "forbidden_patterns": [pattern for pattern in FORBIDDEN_SOURCE_PATTERNS if pattern in text],
        "forbidden_output_patterns": [
            pattern
            for pattern in ("preflight_existing_metrics", "full_training_backend_pending", "fallback_preview_prediction")
            if pattern in rendered_outputs
        ],
        "errors": errors,
        "all_code_cells_executed": executed_cells == code_cells,
        "completed_full_output": "completed_full" in rendered_outputs,
        "full_training_output": "full_training_was_run" in rendered_outputs and "True" in rendered_outputs,
        "has_visual_output": bool({"image/png", "image/jpeg", "text/html"} & mimes),
        "has_progress_output": "application/vnd.jupyter.widget-view+json" in mimes or outputs > 0,
    }


def expected_notebooks() -> list[dict[str, str]]:
    notebooks: list[dict[str, str]] = []
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            notebook = NOTEBOOK_ROOT / dataset / f"{dataset}_{architecture}.ipynb"
            notebooks.append(
                {
                    "id": f"{dataset}_{architecture}",
                    "dataset": dataset,
                    "architecture": architecture,
                    "notebook": notebook.as_posix(),
                }
            )
    return notebooks


def validate_notebooks(root: Path, failures: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected = {item["notebook"] for item in expected_notebooks()}
    actual = {
        rel(path, root)
        for path in (root / NOTEBOOK_ROOT).rglob("*.ipynb")
        if not path.name.endswith(EXECUTED_SUFFIX)
    }
    executed_copies = {
        rel(path, root)
        for path in (root / NOTEBOOK_ROOT).rglob(f"*{EXECUTED_SUFFIX}")
    }
    for extra in sorted(actual - expected):
        failures.append(f"extra active notebook: {extra}")
    for missing in sorted(expected - actual):
        failures.append(f"missing notebook: {missing}")
    for duplicate in sorted(executed_copies):
        failures.append(f"executed duplicate notebook must be removed: {duplicate}")

    for item in expected_notebooks():
        notebook = root / item["notebook"]
        row: dict[str, Any] = {**item, "exists": notebook.exists()}
        if not notebook.exists():
            failures.append(f"missing notebook: {item['notebook']}")
        else:
            stats = notebook_stats(notebook)
            row["stats"] = stats
            if not (stats["outputs"] or stats["executed_cells"]):
                failures.append(f"notebook is not executed in place: {item['notebook']}")
            if stats["missing_sections"]:
                failures.append(
                    f"notebook is not end-to-end: {item['notebook']} "
                    f"missing {', '.join(stats['missing_sections'])}"
                )
            if stats["forbidden_patterns"]:
                failures.append(
                    f"notebook contains forbidden patterns: {item['notebook']} "
                    f"({', '.join(stats['forbidden_patterns'])})"
                )
            if stats["forbidden_output_patterns"]:
                failures.append(f"notebook contains stale/fallback outputs: {item['notebook']}")
            if stats["errors"]:
                failures.append(f"notebook contains execution errors: {item['notebook']} ({'; '.join(stats['errors'])})")
            if not stats["all_code_cells_executed"]:
                failures.append(f"not all code cells were executed: {item['notebook']}")
            if not stats["completed_full_output"] or not stats["full_training_output"]:
                failures.append(f"notebook has no confirmed full training output: {item['notebook']}")
            if not stats["has_visual_output"]:
                failures.append(f"notebook has no visual output: {item['notebook']}")
            if not stats["has_progress_output"]:
                failures.append(f"notebook has no training/progress output: {item['notebook']}")
        rows.append(row)

        run_dir = root / "runs" / item["dataset"] / item["architecture"] / "seed42_full"
        metrics_path = run_dir / "metrics.json"
        history_path = run_dir / "history.json"
        artifacts = (
            metrics_path,
            history_path,
            run_dir / "checkpoint_best.pt",
            run_dir / "predictions.jsonl",
            run_dir / "test" / "metrics.json",
        )
        for artifact in artifacts:
            if not artifact.exists():
                failures.append(f"missing full-run artifact: {rel(artifact, root)}")
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("status") != "completed_full":
                failures.append(f"full-run status is not completed_full: {rel(metrics_path, root)}")
            expected_epochs = 2 if architecture == "qwen_vlm_qlora_training" else 40
            epochs_completed = int(metrics.get("epochs_completed") or 0)
            valid_epoch_count = 1 <= epochs_completed <= expected_epochs if expected_epochs == 2 else epochs_completed == 40
            if not valid_epoch_count:
                failures.append(f"unexpected full-run epoch count: {rel(metrics_path, root)}")
            if (metrics.get("training_execution") or {}).get("full_training_was_run") is not True:
                failures.append(f"full training flag is missing: {rel(metrics_path, root)}")
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            expected_epochs = 2 if architecture == "qwen_vlm_qlora_training" else 40
            valid_history_count = 1 <= len(history) <= expected_epochs if expected_epochs == 2 else len(history) == 40
            if not valid_history_count or any(epoch.get("loss") is None or epoch.get("val_loss") is None for epoch in history):
                failures.append(f"history has an invalid number of real train/val rows: {rel(history_path, root)}")
    return rows


def scan_executed_notebook_duplicates(root: Path) -> list[str]:
    duplicates: list[str] = []
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        dirs[:] = [name for name in dirs if name not in {".git", ".venv"}]
        for name in files:
            if name.endswith(EXECUTED_SUFFIX):
                duplicates.append(rel(current_path / name, root))
    return sorted(duplicates)


def scan_cache_dirs(root: Path) -> list[str]:
    found: list[str] = []
    for current, dirs, _files in os.walk(root):
        current_path = Path(current)
        dirs[:] = [name for name in dirs if name not in IGNORED_DIR_NAMES]
        for name in list(dirs):
            if name in CACHE_DIR_NAMES:
                found.append(rel(current_path / name, root))
    return sorted(found)


def legacy_inventory(root: Path) -> list[dict[str, Any]]:
    return [{**item, "exists": (root / item["path"]).exists()} for item in LEGACY_CANDIDATES]


def write_reports(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Experiment Hygiene Report",
        "",
        f"Status: **{report['status']}**",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Counts",
        "",
        "| Area | Count |",
        "| --- | ---: |",
        f"| Active notebooks | {report['counts']['notebooks']} |",
        f"| Executed duplicate notebooks | {report['counts']['executed_duplicates']} |",
        f"| Legacy candidates | {report['counts']['legacy_candidates']} |",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        lines.extend(f"- {failure}" for failure in report["failures"])
    else:
        lines.append("- None")
    lines.extend(["", "## Legacy Candidates", "", "| Path | Exists | Reason |", "| --- | --- | --- |"])
    for row in report["legacy_candidates"]:
        lines.append(f"| {row['path']} | {row['exists']} | {row['reason']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate active end-to-end experiment notebooks.")
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--md-report", type=Path, default=DEFAULT_MD_REPORT)
    parser.add_argument("--no-write-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    failures: list[str] = []
    backends = [ARCHITECTURE_SPECS[name].backend_model for name in ARCHITECTURES]
    if len(set(backends)) != len(backends):
        failures.append("architecture slots must use nine distinct training backends")
    notebooks = validate_notebooks(root, failures)
    executed_duplicates = scan_executed_notebook_duplicates(root)
    for duplicate in executed_duplicates:
        failures.append(f"executed duplicate notebook must be removed project-wide: {duplicate}")
    cache_dirs = scan_cache_dirs(root)
    if cache_dirs:
        failures.append(f"generated cache directories remain outside archive: {', '.join(cache_dirs)}")
    notebook_count = sum(1 for row in notebooks if row["exists"])
    report = {
        "status": "passed" if not failures else "failed",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {
            "notebooks": notebook_count,
            "executed_duplicates": len(executed_duplicates),
            "legacy_candidates": len(LEGACY_CANDIDATES),
        },
        "failures": failures,
        "notebooks": notebooks,
        "cache_dirs": cache_dirs,
        "legacy_candidates": legacy_inventory(root),
    }
    if not args.no_write_report:
        write_reports(report, root / args.json_report, root / args.md_report)
    print(json.dumps({"status": report["status"], "failures": failures, "counts": report["counts"]}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
