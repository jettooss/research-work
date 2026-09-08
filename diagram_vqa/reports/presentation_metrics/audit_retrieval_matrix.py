"""Audit independent seed evidence and aggregate the presentation retrieval table."""

from __future__ import annotations

import hashlib
import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("docvqa", "infographicvqa", "ai2d")
ARCHITECTURES = (
    "clip", "hybrid_v4_multipos_training",
    "sam2_dinov2_heterogeneous_balanced_evidence_graph",
    "sam2_dinov2_dual_branch_evidence_graph",
)
NAMES = ("CLIP", "GATv2 + kNN", "Граф с отбором связей по типам",
         "Граф с общим отбором связей")
SEEDS = (42, 43, 44)
PROTOCOL = "bidirectional_mean_unique_documents_v1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        assert isinstance(tensor, torch.Tensor)
        assert torch.isfinite(tensor).all(), name
        digest.update(name.encode("utf-8"))
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_retrieval(retrieval):
    for key in ("question_to_document", "document_to_question"):
        values = [retrieval[key]["recall_at_k"][str(k)] for k in (1, 5, 10)]
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in values)
        assert values == sorted(values)
    for k in ("1", "5", "10"):
        expected = sum(retrieval[d]["recall_at_k"][k] for d in
                       ("question_to_document", "document_to_question")) / 2
        assert math.isclose(retrieval["mean_recall_at_k"][k], expected, abs_tol=1e-12)


def audit_run(dataset, architecture, seed, test_rows, *, project_root=None, runs_root=None, retrieval_root=None):
    project_root = Path(project_root) if project_root is not None else ROOT
    runs_root = Path(runs_root) if runs_root is not None else project_root / "runs"
    if architecture == "clip":
        run = runs_root / "model_matrix" / dataset / architecture / f"seed{seed}" / "val"
        test_path = run.parent / "test/metrics.json"
    else:
        run = runs_root / dataset / architecture / f"seed{seed}_full"
        test_path = run / "test/metrics.json"
    record = {"dataset": dataset, "architecture": architecture, "seed": seed,
              "run": str(run), "verified": False}
    metrics_path = run / "metrics.json"
    if not metrics_path.exists():
        history_path = run / "history.json"
        record.update(reason="training_incomplete",
                      epochs=len(read_json(history_path)) if history_path.exists() else 0)
        return record
    metrics = read_json(metrics_path)
    assert metrics["status"] in {"available_full", "completed_full", "completed_early_stopped"}
    assert metrics["dataset"] == dataset and metrics["seed"] == seed
    history = read_json(run / "history.json")
    assert history and len(history) == metrics["epochs_completed"]
    assert [row["epoch"] for row in history] == list(range(1, len(history) + 1))
    assert all(math.isfinite(row.get("train_loss", row.get("loss", float("nan"))))
               and math.isfinite(row["composite"]) for row in history)
    best_epoch = metrics["best_epoch"]
    assert math.isclose(history[best_epoch - 1]["composite"],
                        max(row["composite"] for row in history), abs_tol=1e-10)
    checkpoint_path = run / "checkpoint_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == best_epoch
    state = checkpoint["heads"] if architecture == "clip" else checkpoint["model"]
    assert state
    weight_hash = state_digest(state)
    expected_count = len(test_rows)
    if architecture in ARCHITECTURES[:2]:
        config = read_json(run / "config.json")
        assert config["max_samples"] is None and config["seed"] == seed
        test = read_json(test_path)
        assert test["split"] == "test" and test["num_samples"] == expected_count
        assert test["dataset"] == dataset and test["seed"] == seed
        predictions = test_path.parent / "predictions.jsonl"
        with predictions.open(encoding="utf-8") as stream:
            assert sum(bool(line.strip()) for line in stream) == expected_count
        retrieval = test["retrieval"]
        source = test_path
    else:
        if metrics["status"] == "completed_early_stopped":
            last = torch.load(run / "checkpoint_last.pt", map_location="cpu", weights_only=False)
            assert last["epoch"] == len(history) and last["history"] == history
            assert last["patience"] >= 5 and len(history) < metrics["epochs_requested"]
        else:
            assert metrics["full_training_was_run"] is True
        # AI2D notebooks store aggregate VQA only; retrieval has its own full-test evidence.
        if dataset != "ai2d":
            assert len(metrics["test"]["predictions"]) == expected_count
        source = (Path(retrieval_root) / dataset / architecture / f"seed{seed}_full/retrieval_unique_test.json"
                  if retrieval_root is not None else run / "retrieval_unique_test.json")
        if not source.exists():
            record.update(reason="retrieval_evaluation_missing", epochs=len(history))
            return record
        report = read_json(source)
        assert report["status"] == "evaluated_full_test" and report["protocol"] == PROTOCOL
        assert report["dataset"] == dataset and report["seed"] == seed
        assert report["architecture"] == architecture and report["best_epoch"] == best_epoch
        assert report["num_samples"] == expected_count
        assert report["num_documents"] == len({str(row["image_id"]) for row in test_rows})
        assert report["checkpoint_sha256"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        retrieval = report["retrieval"]
    validate_retrieval(retrieval)
    record.update(verified=True, epochs=len(history), best_epoch=best_epoch,
                  num_test_samples=expected_count, weight_sha256=weight_hash,
                  source=str(source), recall=retrieval["mean_recall_at_k"])
    return record


def build_audit(*, project_root=ROOT, data_root=None, runs_root=None, retrieval_root=None,
                datasets=DATASETS, architectures=ARCHITECTURES, seeds=SEEDS, manifest=None):
    torch.set_num_threads(2)
    project_root = Path(project_root)
    data_root = Path(data_root) if data_root is not None else project_root.parent
    records = []
    groups = []
    for dataset in datasets:
        manifest_path = Path(manifest) if manifest is not None else data_root / dataset / ("model_matrix_v1" if dataset == "ai2d" else "prepared_v1") / "manifest.jsonl"
        with manifest_path.open(encoding="utf-8") as stream:
            test_rows = [row for line in stream if line.strip()
                         if (row := json.loads(line))["split"] == "test"]
        if not test_rows:
            raise ValueError(f"No test questions in {manifest_path}")
        for architecture in architectures:
            name = NAMES[ARCHITECTURES.index(architecture)]
            current = []
            for seed in seeds:
                try:
                    record = audit_run(dataset, architecture, seed, test_rows,
                                       project_root=project_root, runs_root=runs_root, retrieval_root=retrieval_root)
                except (AssertionError, KeyError, ValueError, OSError, RuntimeError) as error:
                    record = {"dataset": dataset, "architecture": architecture, "seed": seed,
                              "verified": False, "reason": f"{type(error).__name__}: {error}"}
                records.append(record)
                if record["verified"]:
                    current.append(record)
            assert len({r["weight_sha256"] for r in current}) == len(current), "Duplicate seed weights"
            stats = {}
            for k in ("1", "5", "10"):
                values = [r["recall"][k] * 100 for r in current]
                stats[k] = {"mean": statistics.mean(values) if values else None,
                            "sd": statistics.pstdev(values) if len(values) > 1 else None}
            groups.append({"dataset": dataset, "architecture": architecture, "name": name,
                           "seeds": [r["seed"] for r in current], "n": len(current), "percent": stats})
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "protocol": PROTOCOL,
            "sd_ddof": 0, "required_runs": len(datasets) * len(architectures) * len(seeds),
            "verified_runs": sum(r["verified"] for r in records),
            "completed": all(g["n"] == len(seeds) for g in groups), "groups": groups, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT, help="diagram_vqa project directory")
    parser.add_argument("--data-root", type=Path, help="Parent of dataset directories; default: project root's parent")
    parser.add_argument("--runs-root", type=Path, help="Checkpoint/history/config tree; default: PROJECT_ROOT/runs")
    parser.add_argument("--retrieval-root", type=Path, help="Read graph retrieval reports from a separate evaluator output tree")
    parser.add_argument("--output", type=Path, required=True, help="New audit JSON path; existing files are never replaced")
    parser.add_argument("--datasets", "--dataset", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--architectures", "--architecture", nargs="+", choices=ARCHITECTURES, default=ARCHITECTURES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--manifest", type=Path, help="Override test manifest; requires a single dataset")
    args = parser.parse_args()
    if args.manifest is not None and len(args.datasets) != 1:
        parser.error("--manifest requires exactly one dataset")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.datasets)) != len(args.datasets) or len(set(args.architectures)) != len(args.architectures):
        parser.error("dataset, architecture and seed selections must not contain duplicates")
    if args.output.exists():
        parser.error(f"Refusing to replace existing audit: {args.output}; choose a new --output")
    try:
        report = build_audit(project_root=args.project_root, data_root=args.data_root,
                             runs_root=args.runs_root, retrieval_root=args.retrieval_root,
                             datasets=args.datasets, architectures=args.architectures,
                             seeds=args.seeds, manifest=args.manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    except (AssertionError, OSError, ValueError) as error:
        parser.exit(2, f"Retrieval audit failed: {error}\n")
    print(json.dumps({"verified_runs": report["verified_runs"], "required_runs": report["required_runs"],
                      "completed": report["completed"],
                      "pending": [r for r in report["records"] if not r["verified"]]}, indent=2))
    if not report["completed"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
