"""Test portable saved-result entrypoints without feature extraction or training."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1] / "reports/presentation_metrics"
sys.path.insert(0, str(TOOLS))
import audit_retrieval_matrix as audit
import evaluate_saved_retrieval as evaluate


class SavedRetrievalToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_notebook_ast_finds_model_after_cell_reordering_without_running_training(self):
        path = self.root / "notebooks/experiments/ai2d/ai2d_custom.ipynb"
        path.parent.mkdir(parents=True)
        cells = [
            {"cell_type": "markdown", "source": ["New introduction"]},
            {"cell_type": "code", "source": ["raise RuntimeError('training must not run')"]},
            {"cell_type": "markdown", "source": ["New explanation"]},
            {"cell_type": "code", "source": ["WIDTH: int = 3\n"]},
            {"cell_type": "code", "source": ["def unused(value: MissingAnnotation):\n    return value\n"]},
            {"cell_type": "code", "source": ["def offset(value):\n    return value + 1\n"]},
            {"cell_type": "code", "source": [
                "class DualBranchEvidenceGraph(nn.Module):\n"
                "    def __init__(self):\n        super().__init__()\n        self.bias = nn.Parameter(torch.zeros(WIDTH))\n"
                "    def forward(self, values):\n        return offset(values) + self.bias\n"]},
            {"cell_type": "code", "source": ["raise RuntimeError('evaluation cells must not run either')"]},
        ]
        path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
        model = evaluate.notebook_model("ai2d", "custom", project_root=self.root, device="cpu")
        self.assertEqual(tuple(model.bias.shape), (3,))
        torch.testing.assert_close(model(torch.zeros(3)), torch.ones(3))
        self.assertFalse(model.training)

    def test_rank_metrics_preserve_stable_candidate_tie_order(self):
        report = evaluate.rank_metrics(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]), [{1}, {1}])
        self.assertEqual(report["recall_at_k"], {"1": 0.5, "5": 1.0, "10": 1.0})
        self.assertEqual(report["mrr"], 0.75)

    def test_existing_evaluation_output_is_protected(self):
        output = self.root / "ai2d/model/seed42_full/retrieval_unique_test.json"
        output.parent.mkdir(parents=True)
        output.write_text("historical record", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "Refusing to replace"):
            evaluate.ensure_new_outputs(self.root, ["ai2d"], ["model"], [42])
        self.assertEqual(output.read_text(encoding="utf-8"), "historical record")

    def test_selected_audit_counts_and_roots_are_not_hardcoded_to_36(self):
        manifest = self.root / "input.jsonl"
        manifest.write_text(json.dumps({"split": "test", "image_id": "a"}) + "\n", encoding="utf-8")

        def checked(dataset, architecture, seed, rows, **kwargs):
            self.assertEqual(kwargs["runs_root"], self.root / "custom_runs")
            self.assertEqual(kwargs["retrieval_root"], self.root / "new_eval")
            return {"dataset": dataset, "architecture": architecture, "seed": seed, "verified": True,
                    "weight_sha256": str(seed), "recall": {"1": 0.25, "5": 0.5, "10": 0.75}}

        with patch.object(audit, "audit_run", side_effect=checked):
            report = audit.build_audit(project_root=self.root, runs_root=self.root / "custom_runs",
                                       retrieval_root=self.root / "new_eval", datasets=["ai2d"],
                                       architectures=["clip", "hybrid_v4_multipos_training"], seeds=[7, 8], manifest=manifest)
        self.assertEqual(report["required_runs"], 4)
        self.assertEqual(report["verified_runs"], 4)
        self.assertTrue(report["completed"])
        self.assertEqual(report["groups"][0]["seeds"], [7, 8])
        self.assertEqual(report["groups"][0]["percent"]["1"], {"mean": 25.0, "sd": 0.0})

    def test_graph_audit_reads_separate_retrieval_output(self):
        architecture = "sam2_dinov2_dual_branch_evidence_graph"
        run = self.root / "custom_runs/ai2d" / architecture / "seed99_full"
        run.mkdir(parents=True)
        metrics = {"dataset": "ai2d", "seed": 99, "status": "completed_full", "epochs_completed": 1,
                   "best_epoch": 1, "full_training_was_run": True}
        (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (run / "history.json").write_text(json.dumps([{"epoch": 1, "train_loss": 1, "composite": 0.5}]), encoding="utf-8")
        checkpoint = run / "checkpoint_best.pt"
        torch.save({"epoch": 1, "model": {"weight": torch.ones(1)}}, checkpoint)
        retrieval = {"question_to_document": {"recall_at_k": {"1": 1, "5": 1, "10": 1}},
                     "document_to_question": {"recall_at_k": {"1": 1, "5": 1, "10": 1}},
                     "mean_recall_at_k": {"1": 1, "5": 1, "10": 1}}
        output = self.root / "new_eval/ai2d" / architecture / "seed99_full/retrieval_unique_test.json"
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps({"status": "evaluated_full_test", "protocol": audit.PROTOCOL,
                                     "dataset": "ai2d", "seed": 99, "architecture": architecture,
                                     "best_epoch": 1, "num_samples": 1, "num_documents": 1,
                                     "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                     "retrieval": retrieval}), encoding="utf-8")
        result = audit.audit_run("ai2d", architecture, 99, [{"image_id": "a"}],
                                 project_root=self.root, runs_root=self.root / "custom_runs",
                                 retrieval_root=self.root / "new_eval")
        self.assertTrue(result["verified"])
        self.assertEqual(result["source"], str(output))
        self.assertFalse((run / "retrieval_unique_test.json").exists())

    def test_audit_cli_refuses_existing_file_before_auditing(self):
        output = self.root / "existing.json"
        output.write_text("preserve", encoding="utf-8")
        with patch.object(sys, "argv", ["audit", "--output", str(output)]), patch.object(audit, "build_audit") as build:
            with self.assertRaises(SystemExit) as error:
                audit.main()
            self.assertEqual(error.exception.code, 2)
            build.assert_not_called()
        self.assertEqual(output.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
