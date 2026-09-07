from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ai2d", "docvqa", "infographicvqa")
ARCHITECTURES = (
    "sam2_dinov2_dual_branch_evidence_graph",
    "sam2_dinov2_heterogeneous_balanced_evidence_graph",
    "sam2_dinov2_option_conditioned_sparse_graph",
    "graphcolbert_film",
    "hybrid_v4_multipos_training",
    "sam2_dinov2_regularized_sparse_learned_graph",
)


def test_active_tree_has_one_complete_notebook_per_dataset_approach() -> None:
    notebooks = list((ROOT / "notebooks" / "experiments").rglob("*.ipynb"))
    assert len(notebooks) == len(DATASETS) * len(ARCHITECTURES) == 18
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            assert (ROOT / "notebooks" / "experiments" / dataset / f"{dataset}_{architecture}.ipynb").exists()


def test_project_has_no_executed_notebook_duplicates() -> None:
    executed_copies = [
        path
        for path in ROOT.rglob("*.executed.ipynb")
        if ".git" not in path.parts and ".venv" not in path.parts
    ]
    assert executed_copies == []


def test_each_source_notebook_contains_full_experiment_sections() -> None:
    required_markers = ("Image.open", "architecture", "tqdm", "validation", "prediction", "metrics")
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            source = ROOT / "notebooks" / "experiments" / dataset / f"{dataset}_{architecture}.ipynb"
            text = source.read_text(encoding="utf-8")
            assert all(marker.lower() in text.lower() for marker in required_markers), source


def test_source_notebooks_do_not_use_legacy_launch_patterns() -> None:
    forbidden = {
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
    }
    for source in (ROOT / "notebooks" / "experiments").rglob("*.ipynb"):
        text = source.read_text(encoding="utf-8")
        assert not (forbidden & {pattern for pattern in forbidden if pattern in text}), source


def test_each_notebook_is_executed_in_place() -> None:
    for notebook_path in (ROOT / "notebooks" / "experiments").rglob("*.ipynb"):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        outputs = [
            output
            for cell in notebook.get("cells", [])
            for output in cell.get("outputs", [])
        ]
        executed_cells = [
            cell
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code" and cell.get("execution_count") is not None
        ]
        assert outputs, notebook_path
        code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
        assert len(executed_cells) == len(code_cells), notebook_path
        rendered = json.dumps(outputs, ensure_ascii=False)
        assert "completed_full" in rendered or "completed_early_stopped" in rendered, notebook_path
        assert not any(output.get("output_type") == "error" for output in outputs), notebook_path


def test_architectures_are_distinct() -> None:
    assert len(ARCHITECTURES) == len(set(ARCHITECTURES)) == 6


def test_each_experiment_has_full_artifacts() -> None:
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            run_dir = ROOT / "runs" / dataset / architecture / "seed42_full"
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
            assert metrics["status"] in {"completed_full", "completed_early_stopped"}
            epochs_completed = metrics["epochs_completed"]
            epochs_requested = metrics.get("epochs_requested", metrics.get("epochs", epochs_completed))
            assert 1 <= epochs_completed <= epochs_requested
            full_training_was_run = metrics.get(
                "full_training_was_run",
                metrics.get("training_execution", {}).get("full_training_was_run"),
            )
            assert full_training_was_run is True or metrics["status"] == "completed_early_stopped"
            assert len(history) == epochs_completed
            assert all(
                (row.get("loss") is not None or row.get("train_loss") is not None)
                and row.get("val_loss") is not None
                for row in history
            )
            assert (run_dir / "checkpoint_best.pt").exists()
