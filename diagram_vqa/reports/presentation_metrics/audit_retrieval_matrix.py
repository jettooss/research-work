"""Audit independent seed evidence and aggregate the presentation retrieval table."""

from __future__ import annotations

import hashlib
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
NAMES = ("CLIP", "Hybrid GATv2 + kNN", "Heterogeneous Graph + Attention",
         "Dual-Branch + top-k Attention")
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


def audit_run(dataset, architecture, seed, test_rows):
    if architecture == "clip":
        run = ROOT / "runs/model_matrix" / dataset / architecture / f"seed{seed}" / "val"
        test_path = run.parent / "test/metrics.json"
    else:
        run = ROOT / "runs" / dataset / architecture / f"seed{seed}_full"
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
        source = run / "retrieval_unique_test.json"
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


def main():
    torch.set_num_threads(2)
    records = []
    groups = []
    for dataset in DATASETS:
        manifest = ROOT.parent / dataset / ("model_matrix_v1" if dataset == "ai2d" else "prepared_v1") / "manifest.jsonl"
        with manifest.open(encoding="utf-8") as stream:
            test_rows = [row for line in stream if line.strip()
                         if (row := json.loads(line))["split"] == "test"]
        for architecture, name in zip(ARCHITECTURES, NAMES):
            current = []
            for seed in SEEDS:
                try:
                    record = audit_run(dataset, architecture, seed, test_rows)
                except (AssertionError, KeyError, ValueError, FileNotFoundError) as error:
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
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "protocol": PROTOCOL,
              "sd_ddof": 0, "required_runs": 36, "verified_runs": sum(r["verified"] for r in records),
              "completed": all(g["n"] == 3 for g in groups), "groups": groups, "records": records}
    destination = ROOT / "reports/presentation_metrics/retrieval_matrix_audit.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"verified_runs": report["verified_runs"], "required_runs": 36,
                      "completed": report["completed"],
                      "pending": [r for r in records if not r["verified"]]}, indent=2))


if __name__ == "__main__":
    main()
