from __future__ import annotations

import ast
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

from vqa_retrieval.experiment_matrix import ARCHITECTURES, ExperimentConfig, train_experiment
from vqa_retrieval.notebook_runs import notebook_paths, prepare_notebook, reserve_run

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("notebook_hygiene", ROOT / "scripts/validate_experiment_hygiene.py")
hygiene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hygiene)


def code(source):
    return {"cell_type": "code", "source": source, "metadata": {}, "outputs": [], "execution_count": None}


class NotebookReproductionTests(unittest.TestCase):
    def test_registry_covers_exactly_the_committed_eighteen_notebooks(self):
        self.assertEqual(len(ARCHITECTURES), 6)
        self.assertEqual(set(notebook_paths(ROOT)), set((ROOT / "notebooks/experiments").rglob("*.ipynb")))
        with tempfile.TemporaryDirectory() as temp:
            for path in notebook_paths(ROOT):
                before = path.read_bytes()
                notebook, config = prepare_notebook(path, Path(temp), seed=44)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(config["seed"], 44)
                self.assertIn("seed44_full", config["run_dir"])
                self.assertTrue(config["full_data"])
                self.assertEqual(hygiene.notebook_stats(path)["missing_sections"], [])
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        ast.parse("".join(cell["source"]))
                        self.assertEqual(cell["outputs"], [])

    def test_configuration_executes_before_seeding_and_changes_external_root_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            path = base / "project/notebooks/experiments/ai2d/ai2d_hybrid_v4_multipos_training.ipynb"
            path.parent.mkdir(parents=True)
            original = {"cells": [code("from pathlib import Path\nimport random\nSEED = 42\nSMOKE = True\nEPOCHS = 1 if SMOKE else 40\nROOT = find_project_root()\nEXTERNAL_ROOT = ROOT.parent\nRUN_DIR = ROOT / 'old_run'\nFORCE_RETRAIN = False\nrandom.seed(SEED)\nobserved = random.random()\n"),
                                  code("def train_local_experiment(**kwargs): return kwargs\nresult = train_local_experiment(seed=SEED, resume=True)\n")], "metadata": {}}
            original["cells"].append(code("class MatrixFeatureStore:\n    def __init__(self, dataset, model, cache_root, device): self.cache_root = cache_root\nstore = MatrixFeatureStore('ai2d', 'hybrid', EXTERNAL_ROOT / 'model_matrix_cache', 'cpu')\n"))
            path.write_text(json.dumps(original), encoding="utf-8")
            prepared, config = prepare_notebook(path, base / "runs", seed=43, external_root=base / "data")
            scope = {}
            for cell in prepared["cells"]:
                exec("".join(cell["source"]), scope)
            self.assertEqual(scope["observed"], random.Random(43).random())
            self.assertEqual(scope["EPOCHS"], 40)
            self.assertEqual(scope["EXTERNAL_ROOT"], (base / "data").resolve())
            self.assertEqual(scope["ROOT"], (base / "project").resolve())
            self.assertEqual(scope["RUN_DIR"], Path(config["run_dir"]))
            self.assertTrue(scope["FORCE_RETRAIN"])
            self.assertFalse(scope["result"]["resume"])
            self.assertEqual(scope["store"].cache_root, Path(config["feature_cache_dir"]) / "matrix")

    def test_refuse_existing_run_or_changed_source_and_accept_explicit_matching_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            manifest = data / "ai2d/model_matrix_v1/manifest.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{}\n')
            _, config = prepare_notebook(notebook_paths(ROOT)[0], Path(temp) / "runs", external_root=data)
            reserve_run(config)
            with self.assertRaises(FileExistsError):
                reserve_run(config)
            reserve_run({**config, "resume": True})
            with self.assertRaises(ValueError):
                reserve_run({**config, "resume": True, "epochs": config["epochs"] + 1})

    def test_changed_manifest_refuses_resume_and_separates_feature_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            manifest = data / "ai2d/model_matrix_v1/manifest.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{}\n')
            path = notebook_paths(ROOT, "ai2d", "sam2_dinov2_dual_branch_evidence_graph")[0]
            _, config = prepare_notebook(path, Path(temp) / "runs", external_root=data)
            run_dir = reserve_run(config)
            record = (run_dir / "notebook_run.json").read_bytes()
            manifest.write_text('{"changed": true}\n')
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                reserve_run({**config, "resume": True})
            _, next_config = prepare_notebook(path, Path(temp) / "runs", external_root=data, resume=True)
            self.assertNotEqual(config["feature_cache_key"], next_config["feature_cache_key"])
            with self.assertRaisesRegex(ValueError, "different notebook/configuration"):
                reserve_run(next_config)
            self.assertEqual((run_dir / "notebook_run.json").read_bytes(), record)

    def test_missing_manifest_can_be_planned_but_not_executed(self):
        with tempfile.TemporaryDirectory() as temp:
            _, config = prepare_notebook(notebook_paths(ROOT)[0], Path(temp) / "runs", external_root=Path(temp) / "absent-data")
            self.assertTrue(config["input_manifest"]["missing"])
            with self.assertRaisesRegex(FileNotFoundError, "manifest is required"):
                reserve_run(config)
            self.assertFalse(Path(config["run_dir"]).exists())

    def test_portable_docvqa_ocr_paths_in_both_actual_notebook_loaders(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            manifest = data / "docvqa/prepared_v1/manifest.jsonl"
            manifest.parent.mkdir(parents=True)
            ocr = data / "docvqa/train/ocr_results/q.json"
            ocr.parent.mkdir(parents=True)
            ocr.write_text('{}')
            manifest.write_text(json.dumps({"image_path": "docvqa/train/documents/q.png", "image_id": "q", "ocr_path": "docvqa/train/ocr_results/q.json", "answers": ["yes"]}))
            for architecture, function in (("hybrid_v4_multipos_training", "load_rows"),
                                           ("sam2_dinov2_dual_branch_evidence_graph", "load_manifest")):
                notebook, _ = prepare_notebook(notebook_paths(ROOT, "docvqa", architecture)[0], Path(temp) / "runs", external_root=data)
                definitions = [node for cell in notebook["cells"] if cell["cell_type"] == "code"
                               for node in ast.parse("".join(cell["source"])).body
                               if isinstance(node, ast.FunctionDef) and node.name == function]
                self.assertEqual(len(definitions), 1)
                module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), definitions[0]], type_ignores=[])
                scope = {"Path": Path, "json": json, "DATA_ROOT": data, "manifest_for": lambda *_: manifest}
                exec(compile(ast.fix_missing_locations(module), "<prepared-loader>", "exec"), scope)
                rows = scope[function](manifest) if function == "load_manifest" else scope[function]("docvqa", data)
                self.assertEqual(Path(rows[0]["ocr_path"]), ocr.resolve())
                self.assertTrue(Path(rows[0]["ocr_path"]).is_file())

    def test_staged_run_metadata_is_compatible_with_embedded_training_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            manifest = data / "ai2d/model_matrix_v1/manifest.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{}\n')
            notebook, config = prepare_notebook(notebook_paths(ROOT, "ai2d", "hybrid_v4_multipos_training")[0], Path(temp) / "runs", external_root=data)
            run_dir = reserve_run(config)
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                for node in ast.parse("".join(cell["source"])).body:
                    if not isinstance(node, ast.FunctionDef) or node.name != "run_local_experiment_config":
                        continue
                    # Execute the actual pre-training run-directory/config stage,
                    # stopping before seeding, dataset loading, models or training.
                    function = copy.deepcopy(node)
                    stop = next(i for i, part in enumerate(function.body)
                                if isinstance(part, ast.Expr) and isinstance(part.value, ast.Call)
                                and isinstance(part.value.func, ast.Name) and part.value.func.id == "seed_everything")
                    function.body = function.body[:stop]
                    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
                    scope = {"Path": Path, "json": json, "ensure_free_space": lambda *_: None}
                    exec(compile(ast.fix_missing_locations(module), "<training-prefix>", "exec"), scope)
                    args = argparse.Namespace(model="hybrid_gatv2_knn", dataset="ai2d", split="val", seed=42,
                                              run_dir_override=run_dir, output_dir=run_dir.parent, max_samples=None,
                                              epochs=40, early_stopping_patience=0, batch_size=32, eval_batch_size=64, resume=False)
                    scope["run_local_experiment_config"](args)
            self.assertTrue((run_dir / "config.json").is_file())
            self.assertEqual(json.loads((run_dir / "notebook_run.json").read_text()), config)

    def test_standalone_architecture_cannot_be_dispatched_as_generic_graph_model(self):
        config = ExperimentConfig("ai2d", "sam2_dinov2_dual_branch_evidence_graph", seed=44, epochs=20)
        with self.assertRaisesRegex(ValueError, "standalone notebook"):
            train_experiment(config)


class NotebookArtifactValidationTests(unittest.TestCase):
    def make_run(self, root: Path, *, early=False):
        root.mkdir(parents=True, exist_ok=True)
        metrics = {"dataset": "ai2d", "architecture": "sam2_dinov2_dual_branch_evidence_graph", "seed": 44,
                   "epochs_requested": 20 if early else 2, "epochs_completed": 2, "best_epoch": 1,
                   "status": "completed_early_stopped" if early else "completed_full",
                   "full_training_was_run": True, "test": {"accuracy": 0.5}}
        (root / "metrics.json").write_text(json.dumps(metrics))
        (root / "history.json").write_text(json.dumps([{"epoch": i, "train_loss": 1.0, "val_loss": 1.1} for i in (1, 2)]))
        (root / "predictions.json").write_text('[{"question_id": "q", "prediction": "a"}]')
        (root / "checkpoint_best.pt").write_bytes(b"fixture-presence-only")
        return metrics

    def check(self, root):
        return hygiene.validate_run(root, "ai2d", "sam2_dinov2_dual_branch_evidence_graph", 44)

    def test_real_schema_accepts_early_stopping_and_non42_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_run(root, early=True)
            self.assertEqual(self.check(root), [])

    def test_missing_external_checkpoint_and_predictions_are_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_run(root)
            (root / "checkpoint_best.pt").unlink()
            (root / "predictions.json").unlink()
            failures = self.check(root)
            self.assertTrue(any("checkpoint_best.pt" in value for value in failures))
            self.assertTrue(any("per-example predictions" in value for value in failures))

    def test_mismatched_history_and_fabricated_full_completion_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metrics = self.make_run(root)
            metrics.update(epochs_completed=1, epochs_requested=20)
            (root / "metrics.json").write_text(json.dumps(metrics))
            failures = self.check(root)
            self.assertTrue(any("Epoch count" in value for value in failures))
            self.assertTrue(any("History length" in value for value in failures))

    def test_malformed_history_does_not_crash_validator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_run(root)
            (root / "history.json").write_text('[null, {"epoch": 2, "train_loss": "invented"}]')
            self.assertEqual(sum("Invalid real" in value for value in self.check(root)), 2)

    def test_generic_model_identifier_is_supported_but_not_used_for_standalone_graphs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metrics = self.make_run(root)
            metrics.pop("architecture")
            metrics["model"] = "hybrid_gatv2_knn"
            (root / "metrics.json").write_text(json.dumps(metrics))
            self.assertEqual(hygiene.validate_run(root, "ai2d", "hybrid_v4_multipos_training", 44), [])
            self.assertTrue(any("architecture mismatch" in value for value in self.check(root)))


if __name__ == "__main__":
    unittest.main()
