"""Execute only missing presentation seeds using the existing notebook code."""

from __future__ import annotations

import ast
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/presentation_metrics"
HYBRID = "hybrid_v4_multipos_training"
HETERO = "sam2_dinov2_heterogeneous_balanced_evidence_graph"
QUEUE = [
    ("ai2d", HYBRID, 44),
    ("docvqa", HETERO, 44),
    ("infographicvqa", HETERO, 43),
    ("infographicvqa", HETERO, 44),
    ("infographicvqa", HYBRID, 43),
    ("infographicvqa", HYBRID, 44),
    ("docvqa", HYBRID, 43),
    ("docvqa", HYBRID, 44),
]


def log(message):
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    with (REPORT / "seed_queue.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def completed(run):
    metrics = run / "metrics.json"
    if not metrics.exists() or not (run / "checkpoint_best.pt").exists():
        return False
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    return payload.get("status") in {"completed_full", "completed_early_stopped"} and bool(payload.get("test"))


def configure_seed(notebook, seed):
    changed = {"SEED": 0, "RUN_DIR": 0}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        lines = cell.source.splitlines(keepends=True)
        replacements = []
        for node in ast.parse(cell.source).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id not in changed:
                continue
            if target.id == "SEED":
                value = f"SEED = {seed}\n"
            else:
                value = "".join(lines[node.lineno - 1:node.end_lineno])
                assert "seed42_full" in value
                value = value.replace("seed42_full", f"seed{seed}_full")
            replacements.append((node.lineno - 1, node.end_lineno, value))
            changed[target.id] += 1
        for start, end, value in reversed(replacements):
            lines[start:end] = [value]
        cell.source = "".join(lines)
        cell.outputs = []
        cell.execution_count = None
    assert changed == {"SEED": 1, "RUN_DIR": 1}, changed
    return notebook


def execute_run(dataset, architecture, seed):
    run = ROOT / "runs" / dataset / architecture / f"seed{seed}_full"
    if completed(run):
        log(f"SKIP completed {dataset}/{architecture}/seed{seed}")
        return
    source = ROOT / "notebooks/experiments" / dataset / f"{dataset}_{architecture}.ipynb"
    notebook = configure_seed(nbformat.read(source, as_version=4), seed)
    run.mkdir(parents=True, exist_ok=True)

    def persist(cell=None, cell_index=None, **kwargs):
        # Preserve per-seed cell outputs without adding duplicate active notebooks.
        snapshot = run / "notebook_execution.json"
        temporary = snapshot.with_suffix(".tmp")
        temporary.write_text(nbformat.writes(notebook), encoding="utf-8")
        temporary.replace(snapshot)
        if cell_index is not None:
            log(f"CELL {dataset}/{architecture}/seed{seed} index={cell_index}")

    log(f"START {dataset}/{architecture}/seed{seed}")
    client = NotebookClient(notebook, timeout=None, kernel_name="data-cu124",
                            resources={"metadata": {"path": str(ROOT)}},
                            allow_errors=False, on_cell_executed=persist)
    try:
        client.execute()
    finally:
        persist()
    assert completed(run), f"Missing completed test metrics: {run}"
    log(f"COMPLETED {dataset}/{architecture}/seed{seed}")


def main():
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.pop("NOTEBOOK_SMOKE", None)
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    (REPORT / "seed_queue.pid").write_text(str(os.getpid()), encoding="ascii")
    for dataset, architecture, seed in QUEUE:
        execute_run(dataset, architecture, seed)
    log("ALL_MISSING_PRESENTATION_SEEDS_COMPLETED")


if __name__ == "__main__":
    main()
